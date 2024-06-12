<img src="sm_cover.png" alt="sm_cover" width="500" height="500">

> **StableMask: Refining Causal Masking in Decoder-only Transformer**\
> Qingyu Yin, Xuzheng He, Xiang Zhuang, Yu Zhao, Jianhua Yao, Xiaoyu Shen, Qiang Zhang\
> Paper: [https://arxiv.org/abs/2402.04779](https://arxiv.org/abs/2402.04779)

## News

**2024/06/11** Talk at AITIME, check [here](https://live.bilibili.com/21813994) for livestream! 

**2024/05/02** 🔥 Our paper has been accepted by ICML'24! See you in Vienna!

**2024/02/10** 📖 We have uploaded our preprint to ArXiv!

## Abstract

The decoder-only Transformer architecture with causal masking and relative position encoding (RPE) has become the de facto choice in language modeling. Despite its exceptional performance across various tasks, we have identified two limitations: First, it prevents all attended tokens from having zero weights during the softmax stage, even if the current embedding has sufficient self-contained information. This compels the model to assign disproportional excessive attention to specific tokens. Second, RPE-based Transformers are not universal approximators due to their limited capacity atencoding absolute positional information, which limits their application in position-critical tasks. In this work, we propose StableMask: a parameter-free method to address both limitations by refining the causal mask. It introduces pseudo-attention values to balance attention distributions and encodes absolute positional information via a progressively decreasing mask ratio. StableMask’s effectiveness is validated both theoretically and empirically, showing significant enhancements in language models with parameter sizes ranging from 71M to 1.4B across diverse datasets and encoding methods. We further show that it supports integration with existing optimization techniques, making it easily usable in practical applications.
![sm](sm.png "StableMask Architecture")
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
Here we provide a guide for pretraining a toy example using the wikitext-103 dataset.

### 1. Dataset Preparation
   
First make sure that you are under the stablemask folder. We create a folder for our dataset.
```
mkdir dataset
cd dataset
```
Then download wikitext-103 dataset. Here we choose the huggingface-cli. See [this link](https://huggingface.co/docs/huggingface_hub/guides/cli) for further instruction. 

For users who fail to directly visit huggingface, we recommend to use hf-mirror:
```
export HF_ENDPOINT=https://hf-mirror.com
```

Download wikitext-103 dataset with huggingface-cli:

```
huggingface-cli download --repo-type dataset --resume-download wikitext --local-dir wikitext --local-dir-use-symlinks False
```

### 2. Pretraining

First run a simple test for checking the environment availability:

```
python test_environment.py
```

Run the shell script to start training:

```
bash run.sh
```


## Citation
Please cite:
```
@misc{yin2024stablemask,
      title={StableMask: Refining Causal Masking in Decoder-only Transformer}, 
      author={Qingyu Yin and Xuzheng He and Xiang Zhuang and Yu Zhao and Jianhua Yao and Xiaoyu Shen and Qiang Zhang},
      year={2024},
      eprint={2402.04779},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
