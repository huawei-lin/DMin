import torch
import torch.nn.functional as F
from PIL import Image
import json
import collections.abc
import io


class Struct:
    """The recursive class for building and representing objects with."""

    def __init__(self, obj={}, **kwargs):
        for k, v in obj.items():
            if isinstance(v, dict):
                setattr(self, k, Struct(v))
            else:
                setattr(self, k, v)
        for k, v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, Struct(v))
            else:
                setattr(self, k, v)

    def __getitem__(self, val):
        return self.__dict__[val]

    def __repr__(self):
        return "{%s}" % str(
            ", ".join("%s : %s" % (k, repr(v)) for (k, v) in self.__dict__.items())
        )


def get_default_config():
    """Returns a default config file"""
    config = {
        "stage": "caching",
        "data": {
            "train_data_path": None,
            "test_data_path": None,
            "image_column": "image",
            "image_ROI_column": "image_ROI",
            "prompt_column": "prompt",
            "num_inference_step": 28,
            "num_estimate_step": 5,
            "height": 512,
            "width": 512,
        },
        "model": {
            "model_name_or_path": None,
            "lora_path": None,
        },
        "influence": {
            "result_output_path": "output_dir",
            "cache_path": "cache_dir",
            "seed": 42,
            "n_threads": 1,
            "cpu_offload": False,
        },
        "compression": {"enable": False, "save_original_grad": False, "K": 65536},
    }
    return config


def sanity_check(config):
    if not config.data.train_data_path:
        raise Exception('Missing "train_data_path"')
    if config.stage == "retrieval" and not config.data.test_data_path:
        raise Exception('Missing "test_data_path"')
    if not config.model.model_name_or_path:
        raise Exception('Missing "model_name_or_path"')
    if config.stage not in ["retrieval", "caching"]:
        raise Exception(
            f'Incorrect stage value: {config.stage}, should be "retrieval" or "caching"'
        )


def get_config(config_path):
    """Returns a  config file"""

    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    config = get_default_config()
    config = update(config, json.load(open(config_path)))
    config = Struct(config)
    sanity_check(config)
    return config


def print_model_summary(model):
    total_trainable_params = 0  # To track total trainable params
    total_params = 0  # Track all params
    total_trainable_size = 0
    total_size = 0

    print(f"{'Module':<60} {'Precision':<10} {'Trainable':<10} {'# Parameters':<15}")

    for name, module in model.named_modules():
        # Check if the module has parameters
        for param in module.parameters(recurse=False):
            # Determine the precision (dtype) of parameters
            precision = param.dtype

            # Check if the parameter is trainable
            trainable = param.requires_grad
            num_params = param.numel()
            num_size = num_params * param.element_size()

            # Track totals
            total_params += num_params
            total_size += num_size
            if trainable:
                total_trainable_params += num_params
                total_trainable_size += num_size

            # Print details for each module
            print(
                f"{name:<60} {str(param.device):<10} {str(precision):<10} {str(trainable):<10} {num_params:<15} {num_size:<15}"
            )

    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Total Trainable Parameters: {total_trainable_params:,}")
    print(f"\nTotal Size: {total_size:,}")
    print(f"Total Trainable Size: {total_trainable_size:,}")
    print(f"\nTotal Size: {total_size/1024/1024/1024:,}GB")
    print(f"Total Trainable Size: {total_trainable_size/1024/1024/1024:,}GB")


def print_mem(msg=None):
    print("=" * 50)
    if msg is not None:
        print(msg + ":")
    print(
        "torch.cuda.memory_allocated: %fGB"
        % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    )
    print(
        "torch.cuda.memory_reserved: %fGB"
        % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
    )
    print(
        "torch.cuda.max_memory_reserved: %fGB"
        % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024)
    )
    print("=" * 50)


def bytes_to_image(byte_data):
    return Image.open(io.BytesIO(byte_data)).convert("RGB")


def image_to_bytes(img):
    if not isinstance(img, Image.Image):
        return img
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def normalize(x):
    return F.normalize(x, p=2, dim=-1)
