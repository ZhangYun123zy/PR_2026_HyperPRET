## HyperPRET: Few-shot class incremental learning with precognition and retrospection for hyperspectral imagery

This is the official code of HyperPRET. 
Paper web page: [HyperPRET: Few-shot class incremental learning with precognition and retrospection for hyperspectral imagery](https://www.sciencedirect.com/science/article/pii/S003132032600751X).

## Abstract
Hyperspectral image classification (HSIC) has been widely explored within the scope of Earth observation. However, traditional HSIC models are generally confined to predefined categories, limiting their real-world application. To overcome this, few-shot class-incremental learning (FSCIL) has been developed, allowing models to update with limited samples of new classes and gradually incorporate new knowledge. While aligning the updated model closely with the previous one enhances efficiency, ensuring smooth integration of future classes into the existing framework is equally crucial. Additionally, training on new incremental data without access to earlier datasets often causes catastrophic forgetting of previously learned information. To address these challenges, we propose HyperPRET, an FSCIL framework with precognition and retrospection for hyperspectral imagery. Specifically, we introduce a wavelet-based Mamba (WMamba), which extracts local and global features simultaneously in the frequency domain. We also incorporate virtual prototypes and samples into the embedding space, along with a space reservation loss (SRL) during initial training to allocate sufficient space for future classes. Furthermore, a knowledge aggregation module (KAM) is designed to adaptively fuse predictions from base and novel branches, helping to retain knowledge of base classes while learning new concepts. Extensive experiments on three benchmark hyperspectral datasets validate the effectiveness and superiority of our proposed approach.

## Citation
```
@article{ZHANG2026113786,
title = {HyperPRET: Few-shot class incremental learning with precognition and retrospection for hyperspectral imagery},
journal = {Pattern Recognition},
volume = {179},
pages = {113786},
year = {2026},
issn = {0031-3203},
author = {Yun Zhang and Bobo Xi and Tie Zheng and Jiaojiao Li and Shou Feng and Yunsong Li},
keywords = {Hyperspectral image classification, Few-shot class incremental learning, Mamba, Catastrophic forgetting},
}
```

## Training script
- Train Houston
    python train.py  -dataset Houston -net mamba -f 144 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 100 -schedule Cosine  -gpu 0 -temperature 16 -dataroot /mnt/HDD/data/ZY/HSI_DATA/Houston -alpha 0.5 -balance 0.005 -loss_iter 20 -eta 0 -w_BC_cls 5 -w_BC_binary 10  -w_d 10

 
## Acknowledgment
We thank the following repos providing helpful components/functions in our work.

- [CEC](https://github.com/icoz69/CEC-CVPR2021)

