# Edge Variational Graph Convolutional Networks for Disease Prediction

## About
This is a Pytorch implementation of IER-GCN described in [IER-GCN: Invariant Edge Rationale for Robust Population Graphs in Multisite Neuroimaging] by Tianshu Chu and Youyong Kong.  

## Prerequisites
- `Python 3.7.4+`
- `Pytorch 1.4.0`
- `torch-geometric `
- `scikit-learn`
- `NumPy 1.16.2`

Ensure Pytorch 1.4.0 is installed before installing torch-geometric. 

## Training
```
python train_eval_evgcn.py --train=1
```
To get a detailed description for available arguments, please run
```
python train_eval_evgcn.py --help
```
To download ABIDE dataset, please run the following script in the `data` folder: 
```
python fetch_data.py 
```
If you want to train a new model on your own dataset, please change the data loader functions defined in `dataloader.py` accordingly, then run `python train_eval_evgcn.py --train=1`  

## Inference and Evaluation
```
python train_eval_evgcn.py --train=0
```

