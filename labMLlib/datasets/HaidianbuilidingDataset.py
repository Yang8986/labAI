import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
import asyncio
from PIL import Image
import numpy as np
import os

from torch.utils.data import Dataset
from ..utils.functions import sort_by_sub_dirname
from ..utils.imgPreprocess import imgPreprocessAnnchung_npz_with_filepath


class HaidianbuilidingDataset(Dataset):
    def __init__(
        self,
        mode,
        dataDir,
        geoData_fn="海佃四hy2correct.asc",
        image_size=(256, 256),
    ) -> None:
        start = time.time()
        self.mode = mode
        self.dataDir = dataDir
        self.image_size = image_size
        self.dataset = []
        self.dataset,self.datasetlabel = imgPreprocessAnnchung_npz_with_filepath(image_size=tuple(image_size),root_folder=dataDir,mode=mode,geoData_fn=geoData_fn)
        self.time_used = time.time() - start

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return str(self.datasetlabel[index]),*self.dataset[index]

