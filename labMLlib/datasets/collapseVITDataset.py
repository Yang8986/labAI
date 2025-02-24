import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
import asyncio
from PIL import Image
import re
import numpy as np
import os

from torch.utils.data import Dataset
from ..utils.functions import sort_by_sub_dirname
from ..utils.imgPreprocess import imgPreprocessCollapse


class collapseVITDataset(Dataset):
    def __init__(
        self,
        mode,
        dataDir,
        image_size=(718, 998),
        dim=3,
        padding=(1, 1, 1, 1),
        padding_mode="constant",
        padding_value=0,
        is_basic=True,
    ) -> None:
        start = time.time()
        self.mode = mode
        self.dataDir = dataDir
        self.image_size = image_size
        self.dim = dim
        self.dataset = []
        self.return_task = False
        self.padding = padding
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.npz_list = os.path.join(self.dataDir, mode + ".npz")
        self.basic_list = os.path.join(self.dataDir, "Basic.npz")
        self.is_basic = is_basic
        if self.is_basic:
            self.basic = self._genBasic()
        self.dataset = self._genDataset()
        self.time_used = time.time() - start
        # print("time used in initializing dataset: ", self.time_used)
        # torchvision.utils.save_image(list(self.dataset[0][1]),"test.png",nrow=10)

    async def _init(self):
        if self.return_task:
            await self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # with np.load(self.npz_list) as data:
        key,d,ans = self.dataset[index]
        d = self._pad(torch.Tensor(d))
        ans = torch.Tensor(ans)
        self.fitDim(d)
        self.fitDim(ans)
        if self.is_basic:
            return (key,torch.concat([self.basic,d]),ans)
        else:
            return (key,d,ans)

    def _pad(self, value: list | np.ndarray | torch.Tensor):
        if isinstance(value, list):
            return [self._pad(x) for x in value]
        else:
            if isinstance(value,np.ndarray):value = torch.from_numpy(value)
            value = transforms.CenterCrop(self.image_size)(value)
            value = F.pad(
                value,
                self.padding,
                mode=self.padding_mode,
                value=self.padding_value,
            )
            # while len(value.shape) < self.dim:
            #     value.unsqueeze_(0)
            # print(f"dimension of value is {len(value.shape)} so don't need to unsqueeze.")
            return value

    def _genDataset(self):
        dataset = []
        with np.load(self.npz_list) as data:
            data_path = sort_by_sub_dirname(
                data.files, subdirname_pattern="(WHIRAM_c384_.*_.*_.*?_V3.3)"
            )
            for key,p_list in data_path.items():
                # print(key)
                data_list = []
                d_list = [p for p in p_list if "Sec_output" not in p and "Basic" not in p]
                ans_list = [p for p in p_list if "Sec_output" in p]
                try:
                    d_list = sorted(d_list,key=lambda x:int(re.findall("(\d+).asc",x)[0]))
                    ans_list = sorted(ans_list,key=lambda x:int(re.findall("(\d+).asc",x)[0]))
                except IndexError:
                    print(key)
                    print(d_list)
                    print(ans_list)
                    raise IndexError("IndexError in sorting")
                assert len(ans_list)==len(d_list),f"{key}: length of ans and d mismatch!"
                for d,ans in zip(d_list,ans_list):
                    assert int(re.findall("(\d+).asc",d)[0])==int(re.findall("(\d+).asc",ans)[0]),f"{key}: d and ans mismatch!"
                    data_list.append((key,torch.concat([torch.ones((1,*data[d].shape))*int(re.findall("(\d+).asc",d)[0]),torch.tensor(data[d]).unsqueeze(0)]),data[ans]))
                dataset+=data_list
        return dataset

    def _genBasic(self):
        basic = []
        with np.load(self.basic_list) as data:
            for i in data.files:
                value = torch.from_numpy(data[i])
                value = transforms.CenterCrop(tuple(self.image_size))(value)
                value = F.pad(
                    value,
                    self.padding,
                    mode=self.padding_mode,
                    value=self.padding_value,
                )
                self.fitDim(value)
                # while len(value.shape) < self.dim:
                #     value.unsqueeze_(0)
                # print(f"dimension of data is {len(data[i].shape)} so don't need to unsqueeze.")
                basic.append(value)
        return torch.concat(basic)
    
    def fitDim(self,tensor):
        while len(tensor.shape) < self.dim:
            tensor.unsqueeze_(0)


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
