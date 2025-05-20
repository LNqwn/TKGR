# Relational Message Passing Neural Network for Temporal Knowledge Graph Reasoning

## Overview

T-RMPN is a temporal knowledge graph reasoning Framework based on graph neural network, by iteratively passing messages between edges to aggregate neighborhood information, effectively capturing the relation patterns of temporal knowledge graphs.

### Dependency
- python==3.8
- torch==1.11.0+cu113
- torchdrug==0.2.1
- numpy==1.24.4
- tqdm==4.67.1
- scipy==1.10.1
- torchvision==0.12.1
- dgl-cu113==0.7.1
- torch-scatter>=2.0.8


### Train models
Train the model with specified or default parameters
```
python main.py -d ICEWS18 --batch_size 32 --n_epoch 20 --history_len 8 
```

### Visualize Interpretations 

You can visualize the path interpretations with the following line. Replace the checkpoint with your own path.

```
python pathview.py --checkpoint /results/model_epoch_20.pth
```


# Two-stage Graph Attention Network for Temporal Knowledge Graph Reasoning

## Overview
T-GATN is a temporal knowledge graph reasoning Framework based on graph neural network, with global propagation and local propagation, corresponding to identifying long-term histor
ical patterns and short-term complex dependencies related to the query, respectively.

### Train models
```
python  main.py  --window_size 15  --d ICEWS14 --n_epoch 20

```
