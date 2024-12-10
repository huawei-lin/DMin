from datasets import load_dataset
import pandas as pd
import random
from tqdm import tqdm
import io
import os
import pyarrow as pa
from PIL import Image, ImageDraw
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from copy import deepcopy

from imgcat import imgcat
import matplotlib

matplotlib.use("module://imgcat")

random.seed(42)

dataset_list = [
    {
        "name": "clint-greene/magic-card-captions",
        "image_column": "image",
        "prompt_column": "text",
    },
    {
        "name": "MohamedRashad/midjourney-detailed-prompts",
        "image_column": "image",
        "prompt_column": "short_prompt",
    },
    {
        "name": "HighCWu/diffusiondb_2m_first_5k_canny",
        "image_column": "image",
        "prompt_column": "text",
    },
    {
        "name": "merve/lego_sets_latest",
        "image_column": "image",
        "prompt_column": "prompt",
    },
    {
        "name": "svjack/pokemon-blip-captions-en-ja",
        "image_column": "image",
        "prompt_column": "en_text",
    },
    {
        "name": "Albe-njupt/gesang_flowers",
        "image_column": "image",
        "prompt_column": "text",
    },
]


def image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


total_data_list = []

for dataset_dict in dataset_list:
    dataset = load_dataset(dataset_dict["name"], split="train")

    for i, data_dict in enumerate(tqdm(dataset, desc=dataset_dict["name"])):
        new_data_dict = {
            "image": data_dict[dataset_dict["image_column"]].resize((512, 512)),
            "prompt": data_dict[dataset_dict["prompt_column"]],
            "source_dataset": dataset_dict["name"],
        }
        total_data_list.append(new_data_dict)

random.shuffle(total_data_list)

random.shuffle(total_data_list)

num_train = int(len(total_data_list) * 0.8)

train_data_list = total_data_list[:num_train]
test_data_list = total_data_list[num_train:]
print(
    f"total: {len(total_data_list)}, train: {len(train_data_list)}, test: {len(test_data_list)}"
)

random.shuffle(train_data_list)
random.shuffle(test_data_list)

df_train = pd.DataFrame(train_data_list)
with ThreadPoolExecutor() as executor:
    results = list(
        tqdm(
            executor.map(image_to_bytes, df_train["image"]),
            total=len(df_train),
            desc="Encoding images",
        )
    )
df_train["image"] = results

df_test = pd.DataFrame(test_data_list)
with ThreadPoolExecutor() as executor:
    results = list(
        tqdm(
            executor.map(image_to_bytes, df_test["image"]),
            total=len(df_test),
            desc="Encoding images",
        )
    )
df_test["image"] = results

if not os.path.exists("train"):
    os.makedirs("train")
df_train.to_parquet("train/mixed_dataset.parquet", index=False)

if not os.path.exists("test"):
    os.makedirs("test")
df_test.to_parquet("test/mixed_dataset.parquet", index=False)
