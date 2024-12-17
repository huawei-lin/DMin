# DMin: Scalable Training Data Influence Estimation for Diffusion Models

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/huawei-lin/LLMsEasyFinetune/blob/master/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

The implementation for paper "[DMin: Scalable Training Data Influence Estimation for Diffusion Models](https://arxiv.org/abs/2412.08637)".

DMin is a scalable framework adapting to large diffusion models for estimating the influence of each training data sample. 

**Keywords: Influence Function, Influence Estimation, Training Data Attribution**

![example](assets/examples.jpg)

## Quick Start
Clone this repo to your local device.
```
git clone https://github.com/huawei-lin/DMin.git
cd DMin
```

Create a new environment by [anaconda](https://www.anaconda.com/download) and install the dependencies.
```
conda create -n DMin python=3.10
conda activate DMin
pip install -r requirements.txt
```

Once you have a config file, you can run:
```
python MP_main.py --config='config.json' --stage {caching/retrieval}
```

We also provide two examples in `./examples`, including 1) stabale diffusion with LoRA and 2) unconditional diffusion model on MNIST.

## Example 1: Stable Diffusion with LoRA

We provide a LoRA adaptor in [huaweilin/DMin_sd3_medium_lora_r4](https://huggingface.co/huaweilin/DMin_sd3_medium_lora_r4), which is trained on mixed datasets [huaweilin/DMin_mixed_datasets_8846](https://huggingface.co/datasets/huaweilin/DMin_mixed_datasets_8846). Additionally, we also provide the cached gradients with $K=2^{16}$ in [huaweilin/DMin_sd3_medium_lora_r4_caching_8846](https://huggingface.co/datasets/huaweilin/DMin_sd3_medium_lora_r4_caching_8846).

```
# Make sure you have git-lfs installed before cloning the caching repo (https://git-lfs.com)
git lfs install

cd ./examples/sd3_medium_lora
git clone https://huggingface.co/datasets/huaweilin/DMin_sd3_medium_lora_r4_caching_8846
```

Caching stage: skip caching stage if you clone the caching repo seccessfully. This stage will write the compressed gradients to the `caching_path` of the config.
```
python ../../main.py --config_path config.json --stage caching
```

Retrieval stage: This stage will calculate the influece score for each training data sample, and write the results to the `output_path` of the config.
```
python ../../main.py --config_path config.json --stage retrieval
```

After retrieval stage, you can visualize the result by `visual_infl.ipynb`. For KNN, we also provide the hsnw index in [huaweilin/DMin_sd3_medium_lora_r4_caching_8846](https://huggingface.co/datasets/huaweilin/DMin_sd3_medium_lora_r4_caching_8846). Since the retrieval stage will save the compressed gradient vectors for test data samples, you can do retrieval through the index after retrieval stage by `visual_knn.ipynb`.

For the jsonl file of the test data samples, you can also include the image as `{"prompt": "xxx", "image": "path/to/the/image"}`.

**More information will be uploaded soon.**

## Example 2: Unconditional Diffusion Model (MNIST)

**More information will be uploaded soon.**


## Citation

```
@article{dmin,
  author       = {Huawei Lin and
                  Yingjie Lao and
                  Weijie Zhao},
  title        = {DMin: Scalable Training Data Influence Estimation for Diffusion Models},
  journal      = {CoRR},
  volume       = {abs/2412.08637},
  year         = {2024},
}

@inproceedings{rapidin,
  author       = {Huawei Lin and
                  Jikai Long and
                  Zhaozhuo Xu and
                  Weijie Zhao},
  title        = {Token-wise Influential Training Data Retrieval for Large Language Models},
  booktitle    = {Proceedings of the 62nd Annual Meeting of the Association for Computational
                  Linguistics, {ACL}},
  address      = {Bangkok, Thailand},
  year         = {2024},
}
```
