import os
import sys
import torch
import torch_geometric.datasets as GeoData
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn.data_parallel import DataParallel
import random
import numpy as np

from opt import * 
from EV_GCN import EV_GCN
from utils.metrics import accuracy, auc, prf
#from dataloader import dataloader

#import data.ABIDEParser as Reader

from dataloader_mdd import dataloader
import data.MDDParser as Reader

from utils.loso import resolve_sites_for_subjects, build_loso_splits_from_sites
from collections import Counter

from utils.dir_core import EdgeResidual, posterior_pi, gumbel_topk_mask, MemoryBankS



def build_masks_and_tensors(edge_index_t, edgenet_input_t, hard_eval: bool, topk_ratio: float):
    """
    Given a graph (edge_index_t, edgenet_input_t), compute:
      - w_pae using model_c.edge_net (reusing existing PAE)
      - residual ψ
      - posterior π
      - Top-k mask for C; complement S
      - Return masked (edge_index_C/S, edgenet_input_C/S) and π (for logging)
    """
    with torch.set_grad_enabled(not hard_eval):
        w_pae = model_c.edge_net(edgenet_input_t)[:, 0] if model_c.edge_net(edgenet_input_t).dim()==2 \
                else model_c.edge_net(edgenet_input_t).squeeze(-1)
        psi   = residual(edgenet_input_t)
        pi, pi_logits = posterior_pi(w_pae, psi, tau=opt.dir_tau, alpha=opt.dir_alpha)

        E = pi.shape[0]
        k = max(1, int(topk_ratio * E))
        if hard_eval:
            # hard top-k by π
            topk = torch.topk(pi, k=k, dim=0).indices
            mask_c = torch.zeros_like(pi); mask_c[topk] = 1.0
        else:
            # soft/diff top-k during training
            mask_c = gumbel_topk_mask(pi_logits, k=k, tau_g=opt.dir_tau_gumbel, hard=False)

        mask_s = 1.0 - mask_c
        # Build subsets
        idx_c = mask_c > 0.5 if hard_eval else (mask_c > 0.0)  # keep soft>0 for richer grads
        idx_s = mask_s > 0.5 if hard_eval else (mask_s > 0.0)

        ei_c = edge_index_t[:, idx_c]
        ei_s = edge_index_t[:, idx_s]
        fe_c = edgenet_input_t[idx_c]
        fe_s = edgenet_input_t[idx_s]
    return (ei_c, fe_c, ei_s, fe_s, pi)

def soft_gate_from_pi_logits(pi_logits, topk_ratio: float, tau: float):
    """
    Build a differentiable gate in [0,1] for edges from pi logits.
    We approximate top-k by centering a sigmoid at a detached threshold.
    """
    E = pi_logits.shape[0]
    if E == 0:
        return pi_logits.new_zeros(0)

    # choose k edges to be ~1; rest ~0 (approx)
    k = max(1, int(topk_ratio * E)) if 0.0 < topk_ratio < 1.0 else E
    # detached threshold so gate remains differentiable wrt pi_logits
    th = torch.topk(pi_logits.detach(), k=k, dim=0).values.min()
    gate = torch.sigmoid((pi_logits - th) / max(tau, 1e-6))
    return gate

@torch.no_grad()
def hard_split_S(edge_index_t, edgenet_input_t, pi, topk_ratio: float):
    """
    Hard C/S split (for memory bank & logging only).
    """
    E = pi.shape[0]
    if E == 0:
        return edge_index_t[:, :0], edgenet_input_t[:0]
    k = max(1, int(topk_ratio * E)) if 0.0 < topk_ratio < 1.0 else E
    topk_idx = torch.topk(pi, k=k, dim=0).indices
    mask_c = torch.zeros(E, dtype=torch.bool, device=pi.device)
    mask_c[topk_idx] = True
    mask_s = ~mask_c
    return edge_index_t[:, mask_s], edgenet_input_t[mask_s]

def keep_mask_with_min_deg(edge_index, scores, base_keep_ratio: float,
                           focus_nodes: torch.Tensor, min_deg: int) -> torch.Tensor:
    E = scores.numel()
    k = max(1, int(base_keep_ratio * E)) if 0.0 < base_keep_ratio < 1.0 else E
    keep = torch.zeros(E, dtype=torch.bool, device=scores.device)
    topk_idx = torch.topk(scores, k=k).indices
    keep[topk_idx] = True

    # ensure min degree on focus nodes (e.g., test nodes)
    for u in focus_nodes.tolist():
        inc = (edge_index[0] == u) | (edge_index[1] == u)
        have = (keep & inc).sum().item()
        if have < min_deg:
            cand = inc & (~keep)
            if cand.any():
                cand_idx = torch.where(cand)[0]
                need = min(min_deg - have, cand_idx.numel())
                add_local = torch.topk(scores[cand_idx], k=need).indices
                keep[cand_idx[add_local]] = True
    return keep


if __name__ == '__main__':
    opt = OptInit().initialize()

    import data.MDDParser as Reader
    Reader.set_root(opt.mdd_root)

    print('  Loading dataset ...')
    dl = dataloader() 
    raw_features, y, nonimg = dl.load_data()
    #n_folds = 10
    #cv_splits = dl.data_split(n_folds)

    subject_IDs = Reader.get_ids()  # may be numeric-only, e.g., "50003"

    if opt.mode == 'loso':
        '''
        phenotype_csv = os.path.join(opt.abide_root, "Phenotypic_V1_0b_preprocessed1.csv")
        sites = resolve_sites_for_subjects(
            subject_ids=subject_IDs,
            phenotype_csv=phenotype_csv,
            abide_root=opt.abide_root,
            pipeline=opt.pipeline,
            filt=opt.filt,
            derivative=opt.derivative
        )
        loso_splits = build_loso_splits_from_sites(subject_IDs, sites)
        '''
        sites_dict = Reader.get_subject_score(subject_IDs, score='SITE_ID')
        sites = [sites_dict[sid] for sid in subject_IDs]
        loso_splits = build_loso_splits_from_sites(subject_IDs, sites)

        if not loso_splits:
            raise RuntimeError("No valid LOSO splits were produced.")
        cv_splits = [(tr, te) for (tr, te, _site) in loso_splits]
        n_folds = len(cv_splits)
        # --- Debug / visibility: show which site names are used ---
        from collections import Counter

        # 1) Collapsed site inventory and counts
        cnt = Counter(sites)
        print("\n[LOSO] Collapsed sites detected:")
        for s in sorted(cnt):
            print(f"[LOSO]   {s:<12}  n={cnt[s]}")

        # 2) Example mapping from subject_id -> site (first 12 for brevity)
        pairs = list(zip(subject_IDs, sites))
        print("\n[LOSO] Example subject -> site (first 12):")
        for sid, st in pairs[:12]:
            print(f"[LOSO]   {sid:<12} -> {st}")

        # 3) Planned folds: test site per fold and its test size
        print("\n[LOSO] Planned folds (test site -> #test subjects):")
        for k, (tr_idx, te_idx, site) in enumerate(loso_splits):
            print(f"[LOSO]   fold {k}: {site:<12} (test_n={len(te_idx)})")
        print()

    else:
        n_folds = opt.n_folds
        cv_splits = dl.data_split(n_folds)


    corrects = np.zeros(n_folds, dtype=np.int32) 
    accs = np.zeros(n_folds, dtype=np.float32) 
    aucs = np.zeros(n_folds, dtype=np.float32)
    prfs = np.zeros([n_folds,3], dtype=np.float32)

    for fold in range(n_folds):
        #print("\r\n========================== Fold {} ==========================".format(fold)) 
        if opt.mode == 'loso':
            test_site = loso_splits[fold][2]
            print(f"\n========================== Fold {fold} (test site: {test_site}) ==========================")
        else:
            print(f"\n========================== Fold {fold} ==========================")
        
        train_ind = cv_splits[fold][0] 
        test_ind = cv_splits[fold][1] 
        
        
        # ---- Build a small validation split from the training set (stratified) ----
        # e.g., 10% of train as val; adjust the ratio to your preference
        VAL_RATIO = 0.10

        train_labels = np.array(y)[train_ind]
        pos_idx = [i for i, lab in enumerate(train_labels) if lab == 1]
        neg_idx = [i for i, lab in enumerate(train_labels) if lab == 0]

        n_val_pos = max(1, int(len(pos_idx) * VAL_RATIO))
        n_val_neg = max(1, int(len(neg_idx) * VAL_RATIO))

        # pick the FIRST chunk for reproducibility (you can shuffle with a fixed seed if you prefer)
        val_local_pos = pos_idx[:n_val_pos]
        val_local_neg = neg_idx[:n_val_neg]
        val_local = np.array(val_local_pos + val_local_neg, dtype=int)

        # Map back to global indices
        val_ind  = np.array(train_ind)[val_local].tolist()
        train_core_local = [i for i in range(len(train_ind)) if i not in set(val_local)]
        train_core_ind   = np.array(train_ind)[train_core_local].tolist()

        print(f"  Train={len(train_core_ind)} | Val={len(val_ind)} | Test={len(test_ind)}")
        print("  Train label counts:", Counter(np.array(y)[train_core_ind]))
        print("  Val   label counts:", Counter(np.array(y)[val_ind]))
        print("  Test  label counts:", Counter(np.array(y)[test_ind]))
        

        print('  Constructing graph data...')
        '''
        # extract node features  
        node_ftr = dl.get_node_features(train_ind)
        # get PAE inputs
        edge_index, edgenet_input = dl.get_PAE_inputs(nonimg) 
        # normalization for PAE
        edgenet_input = (edgenet_input- edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)
        '''
        # --- extract node features (feature selection uses train_ind internally) ---
        node_ftr = dl.get_node_features(train_ind)   # returns features for ALL nodes, selected based on train
        n = node_ftr.shape[0]

        # --- full PAE edges over all nodes ---
        edge_index_full, edgenet_input_full = dl.get_PAE_inputs(nonimg)  # full graph edges

        # ---- masks ----
        train_mask = np.zeros(n, dtype=bool); train_mask[train_core_ind] = True
        val_mask   = np.zeros(n, dtype=bool); val_mask[val_ind]          = True
        test_mask  = np.zeros(n, dtype=bool); test_mask[test_ind]         = True
        trainval_mask = train_mask | val_mask

        ei = edge_index_full

        # ---- edge subsets ----
        edge_mask_train    = train_mask[ei[0]]    & train_mask[ei[1]]
        edge_mask_trainval = trainval_mask[ei[0]] & trainval_mask[ei[1]]
        # (optional, for logging only)
        edge_mask_test     = test_mask[ei[0]]     & test_mask[ei[1]]

        edge_index_train    = ei[:, edge_mask_train]
        edge_index_trainval = ei[:, edge_mask_trainval]
        # we will use FULL graph for evaluation (no need to compute edge_index_test unless you want to log it)

        edgenet_input_train    = edgenet_input_full[edge_mask_train]
        edgenet_input_trainval = edgenet_input_full[edge_mask_trainval]
        # FULL graph edgenet input (for evaluation)
        edgenet_input_full_use = edgenet_input_full

        print(f"  Train edges: {edge_index_train.shape[1]} | Train+Val edges: {edge_index_trainval.shape[1]} | Full edges: {edge_index_full.shape[1]}  (Test-only edges: {edge_mask_test.sum()})")

        # ---- normalize PAE inputs using TRAIN EDGES ONLY ----
        mu  = edgenet_input_train.mean(axis=0)
        std = edgenet_input_train.std(axis=0)
        std[std < 1e-6] = 1.0

        #edgenet_input_train    = (edgenet_input_train    - edgenet_input_train.mean(axis=0)) / edgenet_input_train.std(axis=0)
        #edgenet_input_trainval = (edgenet_input_trainval - edgenet_input_trainval.mean(axis=0)) / edgenet_input_trainval.std(axis=0)
        #edgenet_input_full_use = (edgenet_input_full_use - edgenet_input_full_use.mean(axis=0)) / edgenet_input_full_use.std(axis=0)

        edgenet_input_train    = (edgenet_input_train    - mu) / std
        edgenet_input_trainval = (edgenet_input_trainval - mu) / std
        edgenet_input_full_use = (edgenet_input_full_use - mu) / std
        
        # build network architecture (unchanged)
        model = EV_GCN(node_ftr.shape[1], opt.num_classes, dropout=opt.dropout, 
                    edgenet_input_dim=2*nonimg.shape[1], edge_dropout=opt.edropout, 
                    hgc=opt.hgc, lg=opt.lg).to(opt.device)
        model = model.to(opt.device)

        # loss/opt (unchanged)
        loss_fn  = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)

        # tensors
        features_cuda = torch.tensor(node_ftr, dtype=torch.float32).to(opt.device)
        labels        = torch.tensor(y,        dtype=torch.long).to(opt.device)

        # ---- tensors ----
        edge_index_train_t       = torch.tensor(edge_index_train,       dtype=torch.long).to(opt.device)
        edgenet_input_train_t    = torch.tensor(edgenet_input_train,    dtype=torch.float32).to(opt.device)

        edge_index_trainval_t    = torch.tensor(edge_index_trainval,    dtype=torch.long).to(opt.device)
        edgenet_input_trainval_t = torch.tensor(edgenet_input_trainval, dtype=torch.float32).to(opt.device)

        edge_index_full_t        = torch.tensor(edge_index_full,        dtype=torch.long).to(opt.device)
        edgenet_input_full_t     = torch.tensor(edgenet_input_full_use, dtype=torch.float32).to(opt.device)

        '''
        DIR
        '''
        # features_cuda, labels already defined; edges built: edge_index_train/_trainval/_full
        in_dim = node_ftr.shape[1]
        ed_in_dim = edgenet_input_train.shape[1]  # equals 2 * nonimg_dim

        # Causal head (on C)
        model_c = EV_GCN(in_dim, opt.num_classes, dropout=opt.dropout,
                        edgenet_input_dim=ed_in_dim, edge_dropout=opt.edropout,
                        hgc=opt.hgc, lg=opt.lg).to(opt.device)

        # Shortcut head (on S or intervened S_j)
        model_s = EV_GCN(in_dim, opt.num_classes, dropout=opt.dropout,
                        edgenet_input_dim=ed_in_dim, edge_dropout=opt.edropout,
                        hgc=opt.hgc, lg=opt.lg).to(opt.device)

        # Residual scorer ψ (DIR)
        residual = EdgeResidual(in_dim=ed_in_dim, hidden=256, dropout=opt.dropout).to(opt.device)

        loss_fn  = torch.nn.CrossEntropyLoss()
        opt_c = torch.optim.Adam(list(model_c.parameters()) + list(residual.parameters()),
                                lr=opt.lr, weight_decay=opt.wd)
        opt_s = torch.optim.Adam(model_s.parameters(), lr=opt.lr, weight_decay=opt.wd)

        bankS = MemoryBankS(capacity=opt.dir_bank)

        '''
        # build network architecture  
        model = EV_GCN(node_ftr.shape[1], opt.num_classes, opt.dropout, edge_dropout=opt.edropout, hgc=opt.hgc, lg=opt.lg, edgenet_input_dim=2*nonimg.shape[1]).to(opt.device)
        model = model.to(opt.device)

        # build loss, optimizer, metric 
        loss_fn =torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        features_cuda = torch.tensor(node_ftr, dtype=torch.float32).to(opt.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(opt.device)
        '''

        labels = torch.tensor(y, dtype=torch.long).to(opt.device)
        fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)

        '''
        def train(): 
            print("  Number of training samples %d" % len(train_ind))
            print("  Start training...\r\n")
            acc = 0
            for epoch in range(opt.num_iter):
                model.train()  
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(True):
                    node_logits, edge_weights = model(features_cuda, edge_index, edgenet_input)
                    loss = loss_fn(node_logits[train_ind], labels[train_ind])
                    loss.backward()
                    optimizer.step()
                correct_train, acc_train = accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])  
                
                model.eval()
                with torch.set_grad_enabled(False):
                    node_logits, _ = model(features_cuda, edge_index, edgenet_input)
                logits_test = node_logits[test_ind].detach().cpu().numpy()
                correct_test, acc_test = accuracy(logits_test, y[test_ind])
                auc_test = auc(logits_test,y[test_ind])
                prf_test = prf(logits_test,y[test_ind])

                print("Epoch: {},\tce loss: {:.5f},\ttrain acc: {:.5f}".format(epoch, loss.item(), acc_train.item()))
                if acc_test > acc and epoch >9:
                    acc = acc_test
                    correct = correct_test 
                    aucs[fold] = auc_test
                    prfs[fold]  = prf_test  
                    if opt.ckpt_path !='':
                        if not os.path.exists(opt.ckpt_path): 
                            #print("Checkpoint Directory does not exist! Making directory {}".format(opt.ckpt_path))
                            os.makedirs(opt.ckpt_path)
                        torch.save(model.state_dict(), fold_model_path)

            accs[fold] = acc 
            corrects[fold] = correct
            print("\r\n => Fold {} test accuacry {:.5f}".format(fold, acc))

        def evaluate():
            print("  Number of testing samples %d" % len(test_ind))
            print('  Start testing...')
            model.load_state_dict(torch.load(fold_model_path)) 
            model.eval()
            node_logits, _ = model(features_cuda, edge_index, edgenet_input)

            logits_test = node_logits[test_ind].detach().cpu().numpy()
            corrects[fold], accs[fold] = accuracy(logits_test, y[test_ind])
            aucs[fold] = auc(logits_test,y[test_ind]) 
            prfs[fold]  = prf(logits_test,y[test_ind])  

            print("  Fold {} test accuracy {:.5f}, AUC {:.5f}".format(fold, accs[fold], aucs[fold]))
        
        '''
        
        def train():
            print("  Number of training samples %d" % len(train_core_ind))
            print("  Start training...\n")

            best_val = -1.0
            best_snapshot = None

            for epoch in range(opt.num_iter):
                model.train()
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    # ---- TRAIN: train-only subgraph ----
                    logits_train, _ = model(features_cuda, edge_index_train_t, edgenet_input_train_t)
                    loss = loss_fn(logits_train[train_core_ind], labels[train_core_ind])
                    loss.backward()
                    optimizer.step()

                # Train accuracy (optional)
                _, acc_train = accuracy(
                    logits_train[train_core_ind].detach().cpu().numpy(),
                    y[train_core_ind]
                )

                # ---- VALIDATION: (train ∪ val) subgraph ----
                model.eval()
                with torch.set_grad_enabled(False):
                    logits_trainval, _ = model(features_cuda, edge_index_trainval_t, edgenet_input_trainval_t)
                correct_val, acc_val = accuracy(logits_trainval[val_ind].detach().cpu().numpy(), y[val_ind])

                print(f"Epoch: {epoch},\tce loss: {loss.item():.5f},\ttrain acc: {acc_train.item():.5f}\tval acc: {acc_val.item():.5f}")

                # ---- checkpoint by validation only ----
                if acc_val > best_val and epoch > 9:   # warmup optional
                    best_val = acc_val
                    best_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    if opt.ckpt_path != '':
                        if not os.path.exists(opt.ckpt_path):
                            os.makedirs(opt.ckpt_path)
                        torch.save(best_snapshot, fold_model_path)
            
            accs[fold] = best_val 
            corrects[fold] = correct_val
            print("\r\n => Fold {} test accuacry {:.5f}".format(fold, best_val))

            # Load best snapshot (from memory if not saved)
            if best_snapshot is not None and (opt.ckpt_path == '' or not os.path.exists(fold_model_path)):
                model.load_state_dict(best_snapshot, strict=True)


        def evaluate():
            print("  Number of testing samples %d" % len(test_ind))
            print('  Start testing...')

            if opt.ckpt_path != '' and os.path.exists(fold_model_path):
                model.load_state_dict(torch.load(fold_model_path), strict=True)

            model.eval()
            with torch.set_grad_enabled(False):
                # ---- EVAL: FULL graph (train ∪ val ∪ test) ----
                logits_full, _ = model(features_cuda, edge_index_full_t, edgenet_input_full_t)

            pred_logits = logits_full[test_ind].detach().cpu().numpy()
            corrects[fold], accs[fold] = accuracy(pred_logits, y[test_ind])
            aucs[fold] = auc(pred_logits, y[test_ind])
            prfs[fold] = prf(pred_logits, y[test_ind])
            print("  Fold {} test accuracy {:.5f}, AUC {:.5f}".format(fold, accs[fold], aucs[fold]))

        '''
        def train_dir():
            print("  [DIR] Training with full DIR ...\n")
            best_val = -1.0
            best_snapshot = None

            for epoch in range(opt.num_iter):
                model_c.train(); model_s.train(); residual.train()

                # === 1) Build C/S on the TRAIN graph with NO GRAD ===
                with torch.no_grad():
                    # hard top-k for stability during selection (non-diff selection anyway)
                    ei_c_tr, fe_c_tr, ei_s_tr, fe_s_tr, _ = build_masks_and_tensors(
                        edge_index_train_t, edgenet_input_train_t,
                        hard_eval=True,         # hard top-k selection
                        topk_ratio=opt.dir_topk_ratio
                    )

                # === 2) CAUSAL UPDATE (train-only subgraph) ===
                opt_c.zero_grad()
                logits_c, _ = model_c(features_cuda, ei_c_tr, fe_c_tr)
                loss_c = loss_fn(logits_c[train_core_ind], labels[train_core_ind])

                # ---- variance term from SHORTCUT head, computed with NO GRAD ----
                with torch.no_grad():
                    # push current S to the bank for future steps
                    bankS.push(ei_s_tr.cpu(), fe_s_tr.cpu())

                    samples = bankS.sample(opt.dir_J) or [(ei_s_tr.cpu(), fe_s_tr.cpu())]
                    ce_s_vals = []
                    for (s_ei_cpu, s_feat_cpu) in samples:
                        s_ei   = s_ei_cpu.to(opt.device)
                        s_feat = s_feat_cpu.to(opt.device)
                        logits_s_ng, _ = model_s(features_cuda, s_ei, s_feat)
                        ce_s_ng = loss_fn(logits_s_ng[train_core_ind], labels[train_core_ind])
                        ce_s_vals.append(ce_s_ng.detach())
                    loss_s_var = torch.stack(ce_s_vals).var(unbiased=False)

                # ---- DIR main loss (variance detached) ----
                loss_main = loss_c + opt.dir_lambda_var * loss_s_var
                loss_main.backward()
                opt_c.step()

                # === 3) SHORTCUT UPDATE (fresh forward; disjoint graph) ===
                opt_s.zero_grad()
                samples = bankS.sample(opt.dir_J) or [(ei_s_tr.cpu(), fe_s_tr.cpu())]
                ce_s_list = []
                for (s_ei_cpu, s_feat_cpu) in samples:
                    s_ei   = s_ei_cpu.to(opt.device)
                    s_feat = s_feat_cpu.to(opt.device)
                    logits_s, _ = model_s(features_cuda, s_ei, s_feat)
                    ce_s = loss_fn(logits_s[train_core_ind], labels[train_core_ind])
                    ce_s_list.append(ce_s)
                loss_s_mean = torch.stack(ce_s_list).mean()
                loss_s_mean.backward()
                opt_s.step()

                # === 4) VALIDATION on TRAIN+VAL combined graph (hard top-k; no grad) ===
                model_c.eval(); residual.eval()
                with torch.no_grad():
                    ei_c_tv, fe_c_tv, _, _, _ = build_masks_and_tensors(
                        edge_index_trainval_t, edgenet_input_trainval_t,
                        hard_eval=True, topk_ratio=opt.dir_topk_ratio
                    )
                    logits_tv, _ = model_c(features_cuda, ei_c_tv, fe_c_tv)
                    _, acc_val = accuracy(logits_tv[val_ind].detach().cpu().numpy(), y[val_ind])

                if epoch % 10 == 0:
                    print(f"Epoch {epoch:03d} | lossC {loss_c.item():.4f} | varS {loss_s_var.item():.4f} | val {acc_val.item():.4f}")

                if acc_val > best_val and epoch > 9:
                    best_val = acc_val
                    best_snapshot = {k: v.detach().cpu().clone() for k, v in model_c.state_dict().items()}
                    if opt.ckpt_path != '':
                        os.makedirs(opt.ckpt_path, exist_ok=True)
                        torch.save(best_snapshot, fold_model_path)

            if best_snapshot is not None and (opt.ckpt_path == '' or not os.path.exists(fold_model_path)):
                model_c.load_state_dict(best_snapshot, strict=True)


        def evaluate_dir():
            print("  [DIR] Evaluating on FULL graph ...")
            if opt.ckpt_path != '' and os.path.exists(fold_model_path):
                model_c.load_state_dict(torch.load(fold_model_path), strict=True)
            model_c.eval(); residual.eval()
            with torch.no_grad():
                (ei_c_full, fe_c_full, _, _, _) = build_masks_and_tensors(
                    edge_index_full_t, edgenet_input_full_t, hard_eval=True,
                    topk_ratio=opt.dir_topk_ratio
                )
                logits_full, _ = model_c(features_cuda, ei_c_full, fe_c_full)

            logits_test = logits_full[test_ind].detach().cpu().numpy()
            corrects[fold], accs[fold] = accuracy(logits_test, y[test_ind])
            aucs[fold] = auc(logits_test, y[test_ind])
            prfs[fold] = prf(logits_test, y[test_ind])
            print("  Fold {} test accuracy {:.5f}, AUC {:.5f}".format(fold, accs[fold], aucs[fold]))
        '''

        def train_dir():
            print("  [DIR] Training with full DIR (gated) ...\n")
            best_val = -1.0
            best_snapshot = None

            for epoch in range(opt.num_iter):
                model_c.train(); model_s.train(); residual.train()

                # ===== Causal gate on TRAIN graph (with grad) =====
                # π from PAE + residual
                w_pae_tr = model_c.edge_net(edgenet_input_train_t).squeeze(-1)  # (E_tr,)
                #psi_tr   = residual(edgenet_input_train_t)                      # (E_tr,)
                if opt.dir_no_residual:
                    psi_tr = torch.zeros_like(w_pae_tr)      # posterior = prior
                else:
                    psi_tr = residual(edgenet_input_train_t)
                pi_tr, pi_logits_tr = posterior_pi(w_pae_tr, psi_tr,
                                                tau=opt.dir_tau, alpha=opt.dir_alpha)

                # differentiable soft gate centered at a detached top-k threshold
                gate_tr = soft_gate_from_pi_logits(pi_logits_tr, opt.dir_topk_ratio, opt.dir_tau_gumbel)

                # ---- NEW: tiny S-floor so S still perturbs outputs a bit (makes varS > 0) ----
                with torch.no_grad():
                    E_tr = pi_tr.numel()
                    kc = max(1, int(opt.dir_topk_ratio * E_tr))
                    idx = torch.topk(pi_tr.detach(), k=kc).indices
                    hardC = torch.zeros_like(pi_tr, dtype=torch.bool); hardC[idx] = True
                epsS = 0.05  # feel free to expose as --dir_s_floor (recommend 0.05)
                gate_tr = torch.where(hardC, gate_tr, torch.full_like(gate_tr, epsS))

                # ===== Causal update on the FULL TRAIN edge set (no subgraph!) =====
                opt_c.zero_grad()
                edge_weight_tr = (w_pae_tr * gate_tr).to(features_cuda.dtype)
                logits_c, _ = model_c(
                    features_cuda,
                    edge_index_train_t,         # full train edges
                    edgenet_input_train_t,
                    edge_weight_override=edge_weight_tr
                )
                loss_c = loss_fn(logits_c[train_core_ind], labels[train_core_ind])


                # ===== Variance term from S interventions (no grad, disjoint) =====
                with torch.no_grad():
                    ei_s_tr, fe_s_tr = hard_split_S(edge_index_train_t, edgenet_input_train_t, pi_tr, opt.dir_topk_ratio)
                    bankS.push(ei_s_tr.cpu(), fe_s_tr.cpu())

                    samples = bankS.sample(opt.dir_J) or [(ei_s_tr.cpu(), fe_s_tr.cpu())]
                    ce_s_vals = []
                    for (s_ei_cpu, s_feat_cpu) in samples:
                        s_ei   = s_ei_cpu.to(opt.device)
                        s_feat = s_feat_cpu.to(opt.device)
                        logits_s_ng, _ = model_s(features_cuda, s_ei, s_feat)
                        ce_s_ng = loss_fn(logits_s_ng[train_core_ind], labels[train_core_ind])
                        ce_s_vals.append(ce_s_ng.detach())
                    loss_s_var = torch.stack(ce_s_vals).var(unbiased=False)

                # ===== Main loss (variance detached) =====
                loss_main = loss_c + opt.dir_lambda_var * loss_s_var
                loss_main.backward()
                opt_c.step()

                # ===== Shortcut update on a fresh forward =====
                opt_s.zero_grad()
                samples = bankS.sample(opt.dir_J) or [(ei_s_tr.cpu(), fe_s_tr.cpu())]
                ce_s_list = []
                for (s_ei_cpu, s_feat_cpu) in samples:
                    s_ei   = s_ei_cpu.to(opt.device)
                    s_feat = s_feat_cpu.to(opt.device)
                    logits_s, _ = model_s(features_cuda, s_ei, s_feat)
                    ce_s = loss_fn(logits_s[train_core_ind], labels[train_core_ind])
                    ce_s_list.append(ce_s)
                loss_s_mean = torch.stack(ce_s_list).mean()
                loss_s_mean.backward()
                opt_s.step()

                # ===== Validation on TRAIN+VAL with HARD gate (no grad) =====
                model_c.eval(); residual.eval()
                with torch.no_grad():
                    w_pae_tv = model_c.edge_net(edgenet_input_trainval_t).squeeze(-1)
                    psi_tv   = residual(edgenet_input_trainval_t)
                    pi_tv, pi_logits_tv = posterior_pi(w_pae_tv, psi_tv, tau=opt.dir_tau, alpha=opt.dir_alpha)
                    # hard gate by top-k
                    E_tv = pi_tv.shape[0]
                    k_tv = max(1, int(opt.dir_topk_ratio * E_tv)) if 0.0 < opt.dir_topk_ratio < 1.0 else E_tv
                    topk_tv = torch.topk(pi_tv, k=k_tv, dim=0).indices
                    gate_tv = torch.zeros_like(pi_tv); gate_tv[topk_tv] = 1.0

                    w_pae_tv = model_c.edge_net(edgenet_input_trainval_t).squeeze(-1)
                    # build gate_tv (hard or soft) same length E_tv
                    edge_weight_tv = (w_pae_tv * gate_tv).to(features_cuda.dtype)
                    logits_tv, _ = model_c(
                        features_cuda, edge_index_trainval_t, edgenet_input_trainval_t,
                        edge_weight_override=edge_weight_tv
                    )
                    _, acc_val = accuracy(logits_tv[val_ind].detach().cpu().numpy(), y[val_ind])

                if epoch % 10 == 0:
                    print(f"Epoch {epoch:03d} | lossC {loss_c.item():.4f} | varS {loss_s_var.item():.4f} | val {acc_val.item():.4f}")
                                                                            #varS {loss_s_var.item():.4f} | 
                if acc_val > best_val and epoch > 9:
                    best_val = acc_val
                    best_snapshot = {k: v.detach().cpu().clone() for k, v in model_c.state_dict().items()}
                    if opt.ckpt_path != '':
                        os.makedirs(opt.ckpt_path, exist_ok=True)
                        torch.save(best_snapshot, fold_model_path)

            if best_snapshot is not None and (opt.ckpt_path == '' or not os.path.exists(fold_model_path)):
                model_c.load_state_dict(best_snapshot, strict=True)
        
        def evaluate_dir():
            print("  [DIR] Evaluating on FULL graph ...")
            if opt.ckpt_path != '' and os.path.exists(fold_model_path):
                model_c.load_state_dict(torch.load(fold_model_path), strict=True)
            model_c.eval(); residual.eval()

            with torch.no_grad():
                # --- 1) Compute π on FULL graph ---
                w_pae_full = model_c.edge_net(edgenet_input_full_t).squeeze(-1)
                psi_full   = residual(edgenet_input_full_t)
                pi_full, pi_logits_full = posterior_pi(w_pae_full, psi_full,
                                                    tau=opt.dir_tau, alpha=opt.dir_alpha)

                # --- 2) Global sparsity + per-test-node min-degree protection ---
                # If you already defined keep_mask_with_min_deg() above, reuse it.
                # Otherwise, paste the helper we used earlier (it ensures each test node keeps >= min_deg edges).
                min_deg_test = 0     # try 3–8 if a site is weak; 3 is a good start
                keep = keep_mask_with_min_deg(
                    edge_index_full_t, pi_full, opt.dir_topk_ratio,
                    focus_nodes=torch.tensor(test_ind, device=opt.device),
                    min_deg=min_deg_test
                )

                # --- 3) Soft-ish final gate with a tiny floor for non-kept edges ---
                eps = 0.0  # try 0.02–0.10
                gate_full = torch.where(keep, torch.ones_like(pi_full), torch.full_like(pi_full, eps))

                # --- 4) Full-graph forward with overridden weights ---
                edge_weight_full = (w_pae_full * gate_full).to(features_cuda.dtype)
                logits_full, _ = model_c(
                    features_cuda,
                    edge_index_full_t, edgenet_input_full_t,
                    edge_weight_override=edge_weight_full
                )

            pred_logits = logits_full[test_ind].detach().cpu().numpy()
            corrects[fold], accs[fold] = accuracy(pred_logits, y[test_ind])
            aucs[fold] = auc(pred_logits, y[test_ind])
            prfs[fold] = prf(pred_logits, y[test_ind])
            print("  Fold {} test accuracy {:.5f}, AUC {:.5f}".format(fold, accs[fold], aucs[fold]))


        if opt.train==1:
            if opt.dir == 'full':
                train_dir()
            else:
                train()

        elif opt.train==0:
            if opt.dir == 'full':
                evaluate_dir()
            else:
                evaluate()

    print("\r\n========================== Finish ==========================") 
    n_samples = raw_features.shape[0]
    acc_nfold = np.sum(corrects)/n_samples
    print("=> Average test accuracy in {}-fold CV: {:.5f}".format(n_folds, acc_nfold))
    print("=> Average test AUC in {}-fold CV: {:.5f}".format(n_folds, np.mean(aucs)))
    se, sp, f1 = np.mean(prfs,axis=0)
    print("=> Average test sensitivity {:.4f}, specificity {:.4f}, F1-score {:.4f}".format(se, sp, f1))

