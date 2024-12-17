import sys
sys.path.append("../../")
import hnswlib
import numpy as np
import torch
from tqdm import tqdm
from DMIN.utils import get_config
from datasets import load_dataset

config = get_config("config.json")
cache_path = f"./{config.influence.cache_path}/K{config.compression.K}/"
df = load_dataset("huaweilin/DMin_mixed_datasets_8846", split="train").to_pandas()

# Initialize HNSW index
dim = None  # Placeholder for dimensionality
p = None

# Batch parameters
batch_data = []
batch_ids = []

# Load Training data
for idx in tqdm(range(df.shape[0]), desc="Loading and adding data in batches"):
    # Load and normalize data
    file_path = f"{cache_path}/loss_grad_{idx:08d}.pt"
    loss_grad = torch.flatten(torch.load(file_path, map_location=torch.device("cpu"))).tolist()

    # Initialize the index with the first item's dimensionality
    if p is None:
        dim = len(loss_grad)
        p = hnswlib.Index(space='ip', dim=dim)
        p.init_index(max_elements=df.shape[0], ef_construction=200, M=16)
        p.set_num_threads(8)

    # Add to batch
    batch_data.append(loss_grad)
    batch_ids.append(idx)

p.add_items(np.array(batch_data, dtype=np.float32), np.array(batch_ids))

# Finalize index
p.set_ef(100)
p.save_index(f"./{config.influence.cache_path}/index_K{config.compression.K}.bin")
