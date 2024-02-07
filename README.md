# StableMask: Refining Causal Masking in Decoder-only Transformer

![Mamba](sm.pdf "StableMask Architecture")
> **StableMask: Refining Causal Masking in Decoder-only Transformer**\
> Qingyu Yin, Xuzheng He, Xiang Zhuang, Yu Zhao, Jianhua Yao, Xiaoyu Shen, Qiang Zhang\
> Paper: https://arxiv.org/abs/

## Abstract

The decoder-only Transformer architecture with causal masking and relative position encoding (RPE) has become the de facto choice in language modeling. Despite its exceptional performance across various tasks, we have identified two limitations: First, it prevents all attended tokens from having zero weights during the softmax stage, even if the current embedding has sufficient self-contained information. This compels the model to assign disproportional excessive attention to specific tokens. Second, RPE-based Transformers are not universal approximators due to their limited capacity atencoding absolute positional information, which limits their application in position-critical tasks. In this work, we propose StableMask: a parameter-free method to address both limitations by refining the causal mask. It introduces pseudo-attention values to balance attention distributions and encodes absolute positional information via a progressively decreasing mask ratio. StableMask’s effectiveness is validated both theoretically and empirically, showing significant enhancements in language models with parameter sizes ranging from 71M to 1.4B across diverse datasets and encoding methods. We further show that it supports integration with existing optimization techniques, making it easily usable in practical applications.

## Installation

### Pre-requirement

Python >= 3.8

### Required Package

```
cd StableMask
pip install -r requirements.txt
```

### Other requirements

- Linux
- NVIDIA A100/H100 GPU
- PyTorch 2.0+
- CUDA 12.0+

## Pretraining
## Finetuning
## Inference

## Citation
Please cite:
```
@article{stablemask,
  title={},
  author={},
  journal={},
  year={2023}
}
```
