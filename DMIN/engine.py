from torch.multiprocessing import Queue, Value, Lock, Barrier, Manager, Array, Pipe
from typing import List, Optional, Tuple, Union
import torch.multiprocessing as mp
from ctypes import c_bool, c_int
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import time
import json
import torch.nn.functional as F
from pathlib import Path
from copy import copy
import logging
import datetime
import os
import gc
from dataclasses import dataclass, asdict
from DMIN.pipelines.auto_pipeline import IFAutoPipeline
from DMIN.utils import bytes_to_image, image_to_bytes, normalize
from DMIN.dim_reducer import DimReducer

from imgcat import imgcat
import matplotlib
from PIL import Image

matplotlib.use("module://imgcat")

MAX_CAPACITY = 2048
MAX_DATASET_SIZE = int(1e8)


@dataclass
class InfluenceEstimationOutput:
    idx: int
    test_id: Optional[int] = None
    influence_list: Optional[List[float]] = None
    influence_mean: Optional[float] = None
    compressed_influence_list: Optional[Union[List[float], float]] = None
    K: Optional[Union[List[int], int]] = None


def postprocess(loss_grad):
    return normalize(loss_grad)


def read_data(path):
    if "parquet" in path:
        return pd.read_parquet(path).to_dict("records")
    elif "jsonl" in path:
        dataset = []
        with open(path, "r") as fr:
            for line in fr.readlines():
                dataset.append(json.loads(line.strip()))
        return dataset
    else:
        from datasets import load_dataset
        dataset = load_dataset(path, split="train")
        return dataset
    # raise Exception("Unspported data format.")


def MP_run_subprocess(
    rank, world_size, process_id, config, mp_engine, stage="caching", restart=False
):
    pipeline = IFAutoPipeline.from_pretrained(config, config.model.model_name_or_path)
    if config.model.lora_path:
        pipeline.load_lora_weights(config.model.lora_path)

    image_column = config.data.image_column
    image_ROI_column = config.data.image_ROI_column
    prompt_column = config.data.prompt_column
    cache_path = config.influence.cache_path
    num_estimate_steps = config.data.num_estimate_steps
    num_inference_steps = config.data.num_inference_steps
    is_main_process = True if process_id == 0 else False
    height = config.data.height
    width = config.data.width

    train_dataset = read_data(config.data.train_data_path)

    save_original_grad = True
    dim_reducer = None
    K = None
    if config.compression.enable:
        if is_main_process:
            with mp_engine.has_dim_reducer.get_lock():
                dim_reducer = DimReducer(config)
                mp_engine.has_dim_reducer.value = True
        else:
            while True:
                with mp_engine.has_dim_reducer.get_lock():
                    if mp_engine.has_dim_reducer.value == True:
                        break
                time.sleep(0.2)
            dim_reducer = DimReducer(config)

        save_original_grad = config.compression.save_original_grad
        K = config.compression.K

    test_dataset = []
    test_loss_grad_list = []
    test_loss_grad_compressed_list = []
    if stage == "retrieval" and config.data.test_data_path:
        if is_main_process:
        # if True:
            with mp_engine.gpu_locks[rank].get_lock():
                pipeline.to(rank)
                non_image_index = []

                with open(config.data.test_data_path, "r") as fr:
                    for test_id, line in enumerate(fr.readlines()):
                        data_row = json.loads(line)
                        if image_column not in data_row.keys():
                            non_image_index.append(test_id)
                        else:
                            data_row[image_column] = Image.open(data_row[image_column])

                            if image_ROI_column in data_row.keys():
                                data_row[image_ROI_column] = Image.open(
                                    data_row[image_ROI_column]
                                )

                        test_dataset.append(data_row)

                prompts = [test_dataset[i][prompt_column] for i in non_image_index]
                if len(prompts) > 0:
                    test_images_list = []
                    batch_size = 8
                    for begin in range(0, len(prompts), batch_size):
                        test_images = pipeline(
                            prompts[begin : begin + batch_size],
                            num_inference_steps=num_inference_steps,
                        ).images
                        test_images_list += test_images
                    for test_id, image in zip(non_image_index, test_images_list):
                        test_dataset[test_id][image_column] = image
                        imgcat(image)

                for test_id, test_row in enumerate(
                    tqdm(test_dataset, desc="Get test loss grad.")
                ):
                    test_loss_grad_compressed = None
                    if image_ROI_column in test_row.keys():
                        test_loss_grad = pipeline.calculate_influence_score_ROI(
                            image=test_row[image_ROI_column],
                            prompt=test_row[prompt_column],
                            height=height,
                            width=width,
                            num_inference_steps=num_estimate_steps,
                        )
                    else:
                        test_loss_grad = pipeline.calculate_influence_score(
                            test_row[image_column],
                            test_row[prompt_column],
                            height=height,
                            width=width,
                            num_inference_steps=num_estimate_steps,
                        )

                    if config.influence.cpu_offload:
                        test_loss_grad = test_loss_grad.cpu()

                    test_loss_grad = postprocess(test_loss_grad)
                    if dim_reducer:
                        test_loss_grad_compressed = dim_reducer(test_loss_grad, K)

                    if save_original_grad == True:
                        test_loss_grad_list.append(test_loss_grad)
                        test_row["loss_grad"] = test_loss_grad.cpu().tolist()
                    if test_loss_grad_compressed is not None:
                        test_loss_grad_compressed_list.append(test_loss_grad_compressed)
                        test_row["loss_grad_compressed_list"] = [
                            x.cpu().tolist() for x in test_loss_grad_compressed
                        ]
                pipeline.to("cpu")

            df_test_dataset = pd.DataFrame(test_dataset)
            df_test_dataset[image_column] = df_test_dataset[image_column].apply(
                image_to_bytes
            )
#             if image_ROI_column in df_test_dataset.columns:
#                 df_test_dataset[image_ROI_column] = df_test_dataset[
#                     image_ROI_column
#                 ].apply(image_to_bytes)
#             df_test_dataset.to_parquet(
#                 f"{config.influence.result_output_path}/test_dataset.parquet",
#                 index=False,
#             )

            for i in range(mp_engine.num_processing):
                if save_original_grad == True and dim_reducer is None:
                    mp_engine.pipe_list[i][0].send(
                        ([x.cpu() for x in test_loss_grad_list], [])
                    )
                elif save_original_grad == True and dim_reducer is not None:
                    mp_engine.pipe_list[i][0].send(
                        (
                            [x.cpu() for x in test_loss_grad_list],
                            [
                                [x.cpu() for x in sub_test_loss_grad_compressed_list]
                                for sub_test_loss_grad_compressed_list in test_loss_grad_compressed_list
                            ],
                        )
                    )
                else:
                    mp_engine.pipe_list[i][0].send(
                        (
                            [],
                            [
                                [x.cpu() for x in sub_test_loss_grad_compressed_list]
                                for sub_test_loss_grad_compressed_list in test_loss_grad_compressed_list
                            ],
                        )
                    )
            print(f"{process_id}: test_grad sent")
        else:
            print(f"{process_id}: waiting for test_grad")
            test_loss_grad_list, test_loss_grad_compressed_list = mp_engine.pipe_list[
                process_id
            ][1].recv()
            test_loss_grad_list = [x.to(rank) for x in test_loss_grad_list]
            test_loss_grad_compressed_list = [
                [x.to(rank) for x in sub_test_loss_grad_compressed_list]
                for sub_test_loss_grad_compressed_list in test_loss_grad_compressed_list
            ]
            print(f"{process_id}: test_grad obtained")

    train_dataset_size = 0
    test_dataset_size = 0
    if is_main_process:
        train_dataset_size = len(train_dataset)
        test_dataset_size = len(test_dataset)
        with mp_engine.train_dataset_size.get_lock():
            mp_engine.train_dataset_size.value = train_dataset_size
        with mp_engine.test_dataset_size.get_lock():
            mp_engine.test_dataset_size.value = test_dataset_size

        if stage == "retrieval" and config.data.test_data_path:
            mp_engine.pipe_test_dataset_content[0].send(df_test_dataset)
    else:
        while train_dataset_size == 0:
            with mp_engine.train_dataset_size.get_lock():
                train_dataset_size = mp_engine.train_dataset_size.value
            with mp_engine.test_dataset_size.get_lock():
                test_dataset_size = mp_engine.test_dataset_size.value
            time.sleep(0.02)

    def check_files_exits(idx):
        if save_original_grad and dim_reducer is None:
            return os.path.exists(f"{cache_path}/loss_grad_{idx:08d}.pt")
        compressed_exist = True
        for i, (path, k) in enumerate(zip(mp_engine.multi_k_save_path_list, K)):
            if os.path.exists(f"{path}/loss_grad_{idx:08d}.pt") == False:
                compressed_exist = False
                break

        if save_original_grad == False and dim_reducer:
            return compressed_exist
        if save_original_grad and dim_reducer:
            return (
                os.path.exists(f"{cache_path}/loss_grad_{idx:08d}.pt")
                and compressed_exist
            )
        return False

    if restart == False:
        mp_engine.start_barrier.wait()

    idx = 0
    while True:

        while True:
            with mp_engine.train_idx.get_lock(), mp_engine.finished_idx.get_lock(), mp_engine.cal_word_infl.get_lock():
                idx = mp_engine.train_idx.value
                mp_engine.train_idx.value = (
                    mp_engine.train_idx.value + 1
                ) % train_dataset_size
                if mp_engine.finished_idx[idx] == False:
                    mp_engine.finished_idx[idx] = True
                    cal_word_infl = mp_engine.cal_word_infl[idx]
                    break
            time.sleep(0.002)

        if idx >= train_dataset_size:
            break

        pipeline.to(rank)
        try:
            loss_grad = None
            loss_grad_compressed_list = []
            if (
                stage == "retrieval"
                and cache_path is not None
                and
                # os.path.exists(f"{cache_path}/loss_grad_{idx:08d}.pt")
                check_files_exits(idx)
            ):
                if save_original_grad:
                    loss_grad = torch.load(
                        f"{cache_path}/loss_grad_{idx:08d}.pt",
                        map_location=f"cuda:{rank}",
                    )
                if dim_reducer:
                    for i, (path, k) in enumerate(
                        zip(mp_engine.multi_k_save_path_list, K)
                    ):
                        loss_grad_compressed_list.append(
                            torch.load(
                                f"{path}/loss_grad_{idx:08d}.pt",
                                map_location=f"cuda:{rank}",
                            )
                        )

            elif stage == "caching" and check_files_exits(idx):
                mp_engine.result_q.put(
                    InfluenceEstimationOutput(idx=idx), block=True, timeout=None
                )
                continue

            else:

                item = train_dataset[idx]
                prompt = item[prompt_column]
                image = None
                if image_column in item.keys():
                    image = bytes_to_image(item[image_column])
                else:
                    image = pipeline(
                        prompt, num_inference_steps=num_inference_steps
                    ).images[0]
                    image.save(f"{cache_path}/image_{idx:08d}.png")

                loss_grad = pipeline.calculate_influence_score(
                    image,
                    prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_estimate_steps,
                )
                #                 if stage == "retrieval":
                #                     pipeline.to("cpu")

                if (
                    cache_path is not None
                    and not os.path.exists(f"{cache_path}/loss_grad_{idx:08d}.pt")
                    and save_original_grad == True
                ):
                    torch.save(loss_grad, f"{cache_path}/loss_grad_{idx:08d}.pt")

                if dim_reducer is not None:
                    loss_grad_compressed_list = dim_reducer(postprocess(loss_grad), K)
                    for i, (path, k) in enumerate(
                        zip(mp_engine.multi_k_save_path_list, K)
                    ):
                        torch.save(
                            loss_grad_compressed_list[i],
                            f"{path}/loss_grad_{idx:08d}.pt",
                        )

            if stage == "caching":
                mp_engine.result_q.put(
                    InfluenceEstimationOutput(idx=idx), block=True, timeout=None
                )
                continue

            if save_original_grad:
                loss_grad = postprocess(loss_grad)

            for test_id in range(test_dataset_size):
                influence_mean = None
                influence_list = None
                compressed_influence_list = None

                if save_original_grad:
                    test_loss_grad = test_loss_grad_list[test_id]
                    influence_list = (test_loss_grad * loss_grad).sum(dim=-1)
                    influence_mean = influence_list.mean().cpu().item()
                    influence_list = influence_list.cpu().tolist()

                if dim_reducer:
                    compressed_influence_list = []
                    for i, (
                        test_loss_grad_compressed,
                        loss_grad_compressed,
                        k,
                    ) in enumerate(
                        zip(
                            test_loss_grad_compressed_list[test_id],
                            loss_grad_compressed_list,
                            K,
                        )
                    ):
                        compressed_influence_list.append(
                            (test_loss_grad_compressed * loss_grad_compressed)
                            .sum(dim=-1)
                            .tolist()
                        )

                mp_engine.result_q.put(
                    InfluenceEstimationOutput(
                        test_id=test_id,
                        idx=idx,
                        influence_list=influence_list,
                        influence_mean=influence_mean,
                        compressed_influence_list=compressed_influence_list,
                        K=K,
                    ),
                    block=True,
                    timeout=None,
                )
        except Exception as e:
            with mp_engine.finished_idx.get_lock():
                mp_engine.finished_idx[idx] = False
                print(e)
            raise e


def MP_run_get_result(config, mp_engine, stage="caching"):
    train_dataset_size = 0

    while train_dataset_size == 0:
        with mp_engine.train_dataset_size.get_lock():
            train_dataset_size = mp_engine.train_dataset_size.value
        with mp_engine.test_dataset_size.get_lock():
            test_dataset_size = mp_engine.test_dataset_size.value
        time.sleep(1)
    total_size = max(test_dataset_size, 1) * train_dataset_size
    print(
        f"total_size: {total_size}, train_dataset_size: {train_dataset_size}, test_dataset_size: {test_dataset_size}"
    )

    save_handler_list = []
    if stage == "retrieval":
        df_test_dataset = mp_engine.pipe_test_dataset_content[1].recv()
        df_test_dataset.to_parquet(
            f"{config.influence.result_output_path}/test_dataset.parquet",
            index=False,
        )
        for test_id in range(test_dataset_size):
            save_handler_list.append(
                open(
                    f"{config.influence.result_output_path}/results_{test_id}.jsonl",
                    "w",
                )
            )

    mp_engine.start_barrier.wait()

    i = 0
    # while True:
    for _ in tqdm(range(total_size)):
        try:
            # result_item = mp_engine.result_q.get(block=True, timeout=300)
            result_item = mp_engine.result_q.get(block=True)
        except Exception as e:
            print("Cal Influence Function Finished!")
            break

        if stage == "caching":
            continue

        result_dict = asdict(result_item)
        save_handler_list[int(result_dict["test_id"])].write(json.dumps(result_dict) + "\n")

    for handler in save_handler_list:
        handler.close()


class MPEngine:
    def __init__(self, world_size, num_processing):
        self.result_q = Queue(maxsize=MAX_CAPACITY)

        self.train_idx = Value(c_int, 0)

        self.pipe_list = [Pipe() for _ in range(num_processing)]
        self.pipe_test_dataset_content = Pipe()
        self.num_processing = num_processing
        self.multi_k_save_path_list = None

        self.start_barrier = Barrier(world_size + 1)
        self.finished_a_test = Value(c_int, 0)
        self.cur_processes_num = Value(c_int, 0)
        self.has_dim_reducer = Value(c_bool, False)

        self.gpu_locks = [Value(c_int, 0) for _ in range(world_size)]

        self.train_dataset_size = Value(c_int, 0)
        self.test_dataset_size = Value(c_int, 0)

        self.finished_idx = Array(c_bool, [False for _ in range(MAX_DATASET_SIZE)])

        self.cal_word_infl = Array(c_int, [-1 for _ in range(MAX_DATASET_SIZE)])

    def action_finished_a_test(self):
        with self.train_idx.get_lock():
            self.train_idx.value = 0


def calc_infl_mp(config):
    gpu_num = torch.cuda.device_count()
    print(f"{gpu_num} GPUs available!")

    threads_per_gpu = int(config.influence.n_threads)

    num_processing = gpu_num * threads_per_gpu
    mp_engine = MPEngine(num_processing, num_processing)

    cache_path = config.influence.cache_path
    if cache_path and not os.path.exists(cache_path):
        os.makedirs(cache_path)
    if config.compression.enable and isinstance(config.compression.K, int):
        config.compression.K = [config.compression.K]
    if config.compression.enable:
        mp_engine.multi_k_save_path_list = []
        for k in config.compression.K:
            path = config.influence.cache_path + f"/K{k}"
            mp_engine.multi_k_save_path_list.append(path)
            os.makedirs(path, exist_ok=True)

    result_output_path = config.influence.result_output_path
    if result_output_path and not os.path.exists(result_output_path):
        os.makedirs(result_output_path)

    mp_handler = []
    mp_args = []
    stage = config.stage
    for i in range(gpu_num):
        for j in range(threads_per_gpu):
            mp_handler.append(
                mp.Process(
                    target=MP_run_subprocess,
                    args=(
                        i,
                        gpu_num,
                        i * threads_per_gpu + j,
                        config,
                        mp_engine,
                        stage,
                    ),
                )
            )
            mp_args.append(mp_handler[-1]._args)
    mp_handler.append(
        mp.Process(target=MP_run_get_result, args=(config, mp_engine, stage))
    )

    for x in mp_handler:
        x.start()

    while mp_handler[-1].is_alive():
        cur_processes_num = len([1 for x in mp_handler if x.is_alive()])
        if cur_processes_num < num_processing + 1:
            print(f"ready to restart processing, {cur_processes_num}/{num_processing}")
            for i, x in enumerate(mp_handler):
                if x.is_alive() != True:
                    print(f"start {mp_args[i]}")
                    mp_handler[i] = mp.Process(
                        target=MP_run_subprocess, args=mp_args[i] + (True,)
                    )
                    mp_handler[i].start()
            continue
        with mp_engine.cur_processes_num.get_lock():
            mp_engine.cur_processes_num.value = cur_processes_num
        time.sleep(1)

    # infl = MP_run_get_result(config, mp_engine)
    for x in mp_handler:
        x.terminate()


#         x.join()

# return infl
