# WS-UIENet: A Wavelet-Decoupled Semantics-Guided Underwater Image Enhancement Network

This repository contains the official implementation of the following paper:
> **WS-UIENet: A Wavelet-Decoupled Semantics-Guided Underwater Image Enhancement Network**<br>
> Shixuan Xu; Xinghui Dong<sup>*</sup><br>
> Intelligent Marine Technology and System, 2026<br>

# Usage

### Train:

Use this line to train the model

```
    CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 --use_env train.py --exp ./config/xxx.yml
```

### Test:

Use this line to predict results

```
    python3 predict.py --cuda_id 0 --exp ./config/xxx.yml
```
