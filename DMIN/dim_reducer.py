from tqdm import tqdm
import os
import numpy as np
import torch.nn.functional as F
from copy import copy
import torch
import pickle
import time


class DimReducer:
    def __init__(self, config, seed=42):
        self.is_init = False
        self.config = config
        self.D = None
        self.K = None
        self.random_mat = None
        self.M = 1
        self.perm_mat = None
        self.seed = seed

    def pad(self, x, K):
        ori_D = x.shape[-1]
        max_K = max(K)
        new_D = ((ori_D - 1) // max_K + 1) * max_K
        x = F.pad(x, (0, new_D - ori_D), "constant", 0)
        return x

    def __call__(self, vec, K):
        if isinstance(K, int):
            K = [K]
        begin_time = time.time()
        vec = self.pad(vec, K)

        if self.is_init == False:
            D = vec.shape[-1]
            self.init(D)

        self.random_mat = self.random_mat.to(vec.device)
        self.perm_mat = self.perm_mat.to(vec.device)

        vec = vec[:, self.perm_mat.squeeze()]
        vec = vec * self.random_mat

        vec_list = []
        for k in K:
            step = self.D // k
            vec_list.append(torch.sum(vec.reshape((vec.shape[0], -1, step)), axis=-1))

#         if len(vec_list) == 1:
#             return vec_list[0]
        return vec_list

    def init(self, D):
        self.is_init = True
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.D = D
        self.file_name = os.path.join(
            self.config.influence.cache_path, f"DimReducer_D{self.D}.obj"
        )
        if not self.load():
            print("Creating random and shuffling matrices. It may take a few minutes.")
            self.create_random_mat(D)
            self.create_perm_mat(D)
            self.save()

    def create_random_mat(self, D):
        self.random_mat = torch.randint(0, 2, (D,))
        self.random_mat[self.random_mat < 1e-8] = -1

    def create_perm_mat(self, D):
        self.perm_mat = torch.randperm(D).unsqueeze(0)

    def save(self):
        if os.path.exists(self.file_name):
            return
        with open(self.file_name, "wb") as f:
            pickle.dump(self, f)

    def load(self):
        if not os.path.exists(self.file_name):
            return False
        with open(self.file_name, "rb") as f:
            new_obj = pickle.load(f)
        self.__dict__ = copy(new_obj.__dict__)
        self.random_mat = self.random_mat.to(torch.int8)
        return True
