import sys
import os
import time
import configparser
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from labMLlib.datasets.collapseVITDataset import collapseVITDataset

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
import asyncio
from PIL import Image
import numpy as np
import os
import tqdm
import re

from torch.utils.data import Dataset
from labMLlib.utils.functions import sort_by_sub_dirname
from labMLlib.utils.imgPreprocess import imgPreprocessCollapse

dataDir=""
padding = (1, 1, 1, 1)
image_size=(718, 998)
dim=3
padding=(0, 0)
padding_mode="constant"
padding_value=0
num_workers=16
mode="train"
npz_list = os.path.join(dataDir, mode + ".npz")
class Data:
    def __init__(
        self,
        label: str,
        data: list | np.ndarray | torch.Tensor,
        answer: np.ndarray | torch.Tensor,
    ):
        self.label = label
        self.data = data  # shape: (b,c,w,h)
        self.answer = answer
def pad(value: list | np.ndarray | torch.Tensor):
    if isinstance(value, list):
        return [pad(x) for x in value]
    else:
        value = torch.from_numpy(value)
        value = transforms.CenterCrop(image_size)(value)
        value = F.pad(
            value,
            padding,
            mode=padding_mode,
            value=padding_value,
        )
        while len(value.shape) < dim:
            value.unsqueeze_(0)
        # print(f"dimension of value is {len(value.shape)} so don't need to unsqueeze.")
        return value

def genDataset(npz_list):
    dataset = {}
    with np.load(npz_list) as data:
        data_path = sort_by_sub_dirname(
            data.files, subdirname_pattern="(WHIRAM_c384_.*_.*_.*?_V3.3)"
        )
        for key, p_list in data_path.items():
            d = []
            ans = []
            for p in p_list:
                if "final_max_collapse" in p:
                    ans.append(data[p])
                else:
                    d.append(data[p])
            assert len(ans) == 1, f"{key}: final_max_collapse more than one!"
            dataset[key] = (d, ans[0])
    return dataset

def genBasic(basic_list: str):
    basic = []
    with np.load(basic_list) as data:
        for i in data.files:
            value = torch.from_numpy(data[i])
            value = transforms.CenterCrop(image_size)(value)
            value = F.pad(
                value,
                padding,
                mode=padding_mode,
                value=padding_value,
            )
            while len(value.shape) < dim:
                value.unsqueeze_(0)
            # print(f"dimension of data is {len(data[i].shape)} so don't need to unsqueeze.")
            basic.append(value)
    return torch.concat(basic)
async def worker(key, value):
    data_list, answer = value
    answer, data_list = pad(answer), pad(data_list)
    data = torch.concat(data_list)
    return Data(key, data, answer)

async def workerPool(dataset):
    queue = asyncio.Queue(maxsize=num_workers)
    tasks = []
    for key, value in dataset.items():
        await queue.put(worker(key, value))
        while queue.empty() is False:
            tasks.append(await queue.get())
    result = await asyncio.gather(*tasks, return_exceptions=True)
    merge_result = []
    for i in result:
        merge_result.append(i)
    return merge_result

def asc2npz(
    asc_list,
    target_folder,
    fn="dataset.npz",
    remove_pattern=".*?0.asc",
    verbose=None,
    dtype=np.float16,
    num_workers=16,
):
    if verbose is None:
        verbose = False
    if type(asc_list) != list:
        asc_list = [asc_list]
    async def worker(p,verbose):
        with open(p, "r") as f:
            if verbose:
                print("path: " + p)
            ncols = f.readline().split()[1]
            nrows = f.readline().split()[1]
            xllcorner = f.readline().split()[1]
            yllcorner = f.readline().split()[1]
            cellsize = f.readline().split()[1]
            NODATA_value = f.readline().split()[1]
            if verbose:
                print("ncols: ", ncols)
                print("nrows: ", nrows)
                print("xllcorner: ", xllcorner)
                print("yllcorner: ", yllcorner)
                print("cellsize: ", cellsize)
                print("NODATA_value: ", NODATA_value)
            # print(f"{nrows=}")
            imgData = []
            for i in f.readlines():
                imgData.append(
                    [float(j) if j != NODATA_value else 0 for j in i.split()]
                )
                # print(t)
            imgData = np.array(imgData, dtype=dtype)
            if verbose:
                print("shape:", imgData.shape)
                print("dtype:", imgData.dtype)
                print("min:", imgData.min(), "max", imgData.max())
            # print(imgData.min(), imgData.max(), imgData.mean(), imgData[0][0])
            print(p)
            # compressed_data[p] = imgData
            f.close()
        return p,imgData
    async def workerPool():
        queue = asyncio.Queue(maxsize=num_workers)
        tasks = []
        compressed_data = {}
        pbar = tqdm(asc_list,desc="Progress")
        for p in pbar:
            if re.findall(remove_pattern, p) == []:
                # print(p)
                await queue.put(worker(p, verbose))
                while queue.empty() is False:
                    tasks.append(await queue.get())
        result = await asyncio.gather(*tasks, return_exceptions=True)
        for p,imgData in result:
            compressed_data[p] = imgData
        return compressed_data
    compressed_data = asyncio.run(workerPool())
    np.savez_compressed(os.path.join(target_folder, fn), **compressed_data)

dataset = []
print("generate dataset...")
for key, value in genDataset(npz_list).items():
    data_list, answer = value
    answer, data_list = pad(answer), pad(data_list)
    data = torch.concat(data_list)
    data = Data(key, data, answer)
    dataset.append(data)
print("generate dataset done.")
np.savez("train.npz", **dataset)
