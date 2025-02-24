#python vit_collapse_predict.py --model_dir "models\CaseC\models+MSELoss+Adam(2024-05-23 16-00-48.867039)" --input "dataTest" --basic "./datasets/XDS/Basic.npz" --output output2

import os
import time
from pathlib import Path
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir",
    type=str,
    default="./models/CaseC/models+MSELoss+Adam(2024-05-23 16-00-48.867039)",
    help="model root dir path",
)
parser.add_argument(
    "--input",
    type=str,
    # default="./datasets/XDS/data_sample/1.asc",
    required=True,
    help="input asc path(or dir)",
)
parser.add_argument(
    "--basic",
    type=str,
    default="./datasets/XDS/Basic.npz",
    help="basic npz file path",
)
parser.add_argument(
    "--output",
    type=str,
    default="output",
    help="output directory",
)
opt = parser.parse_args()
if Path(opt.input).is_file():
    opt.input = [opt.input]
    mode = "file"
elif Path(opt.input).is_dir():
    print("detect input is not a file, change to folder mode.")
    mode = "folder"
    files = []
    for dirpath,dirnames,filenames in os.walk(opt.input):
        for filename in filenames:
            files.append(os.path.join(dirpath,filename))
    root = str(Path(files[0]).parent).split(os.sep)
    for file in files[1:]:
        p = str(Path(file).parent).split(os.sep)
        root = root if len(root) < len(p) else p
    if root[0] == "":
        root[0] = os.sep
    root = os.path.join(*root)+os.sep
    print("root:",root)
    opt.input = files
    print("input files:",opt.input)
else:
    raise ValueError("input is not a file or a folder")

os.makedirs(opt.output,exist_ok=True)
s = time.time()

import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from netCDF4 import Dataset

def mask_center(img_size,mask_size):
    mask = torch.zeros(img_size)
    crop_top, crop_left = (img_size[0] - mask_size[0])//2, (img_size[1] - mask_size[1])//2
    mask[crop_top:crop_top+mask_size[0],crop_left:crop_left+mask_size[1]] = 1
    return mask

def get_offset(img_size,mask_size):
    crop_top, crop_left = (img_size[0] - mask_size[0])//2, (img_size[1] - mask_size[1])//2
    crop_bottom, crop_right = img_size[0] -mask_size[0] - crop_top, img_size[1] -mask_size[1] - crop_left
    return ((crop_top,crop_bottom),(crop_left,crop_right))

def get_asc_header_text(asc_file_path: str)->dict:
    with open(asc_file_path, "r") as f:
        lines = f.readlines()
        header = lines[:6]
    return header

def get_asc_header_dict(asc_file_path: str, verbose=None)->dict:
    header_dict = {}
    with open(asc_file_path, "r") as f:
        lines = f.readlines()
        header = lines[:6]
        for i in header:
            header_dict[i.split()[0]] = i.split()[1]
    return header_dict

def get_asc_mask(asc_file_path: str, verbose=None, dtype=np.float16)->np.ndarray:
    header_dict = {}
    with open(asc_file_path, "r") as f:
        lines = f.readlines()
        header, array = lines[:6],lines[6:]
        for i in header:
            header_dict[i.split()[0]] = i.split()[1]
        imgData = []
        for i in array:
            imgData.append([1 if j != header_dict["NODATA_value"] else np.nan for j in i.split()])
        imgData = np.array(imgData, dtype=dtype)
    return imgData

def ascGenMaskAndPredictClean(file_path, predict):
    arr = get_asc_mask(file_path)
    arr[np.isnan(arr)] = 0
    arr = 1 - arr
    try:
        mask*=arr
    except NameError:
        mask=arr
    mask = 1 - mask
    mask = mask.astype(np.float64)
    
    # debug
    print("predict.type: ",predict.dtype, "mask.type: ",mask.dtype, "file path:" ,file_path)
    
    predict = np.where(mask == 0, 99999.123, predict)
    return predict  

def npArray2asc(header, imgData, file_path):
    header_dict = {}
    for i in header:
        header_dict[i.split()[0]] = i.split()[1]
    ncols = header_dict["ncols"]
    nrows = header_dict["nrows"]
    nodata_value = header_dict["NODATA_value"]
    # print(nodata_value)
    offset = get_offset((int(nrows),int(ncols)),imgData.shape)
    header = "".join(header)
    imgData = np.round(imgData,decimals=4)
    imgData = imgData.astype(str)
    imgData = np.where(imgData == "99999.123", nodata_value, imgData)
    imgData = np.pad(imgData,offset,mode="constant",constant_values=nodata_value)
    # print(imgData[0][0])
    imgData = np.array([[" " + j for j in i] for i in imgData])
    imgData = np.array([" ".join(i) for i in imgData])
    imgData = "\n".join(imgData)
    with open(file_path, "w") as f:
        f.write(header + imgData)



def asc2NPArray(p,verbose=False,dtype=np.float16):
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
        # print(p)
        # compressed_data[p] = imgData
        f.close()
    return p,imgData


def fitDim(tensor,dim=3):
    while len(tensor.shape) < dim:
        tensor.unsqueeze_(0)

# print("change dir to ",Path(__file__).parent.parent)
# os.chdir(Path(__file__).parent.parent)

print("current model root dir:",opt.model_dir)
image_size = (998,718)  #不同區域大小可能不同
device = "cpu"
dtype = float
loss = nn.MSELoss()
basic = []
with np.load(opt.basic) as data:
    for i in data.files:
        value = torch.from_numpy(data[i])
        value = transforms.CenterCrop(tuple(image_size))(value)
        value = F.pad(
            value,
            (1,1,1,1),
            mode="constant",
            value=0,
        )
        fitDim(value)
        basic.append(value)
basic =  torch.concat(basic)
model_path = os.path.join(opt.model_dir,"VITNystromCollapse.pth")
struct_path = os.path.join(opt.model_dir,"structure.pt")
model = torch.load(struct_path,map_location=device)
model.load_state_dict(torch.load(model_path,map_location=device)["model_state_dict"])
model = model.to(device,dtype=dtype)

# label,answer_sample = asc2NPArray(answer_sample_path)
# original_size = answer_sample.shape[-2:]
# print(original_size)
# plist = [*Path(label).parts]
for file in files:
    input_sample = torch.tensor(asc2NPArray(file)[1]).unsqueeze(0)
    input_sample = F.pad(
        input_sample,
        (1,1,1,1),
        mode="constant",
        value=0,
    )
    input_data = torch.cat((torch.ones(input_sample.shape)*int(re.findall("(\d+).asc",file)[0]), basic,input_sample),dim=0).unsqueeze(0).to(device,dtype=dtype)
    print("input_data.shape: ",input_data.shape)
    model.eval()
    predict = model(input_data).cpu().detach()
    predict = ascGenMaskAndPredictClean(file,predict.numpy().reshape(image_size))
    target =os.path.join(opt.output,file.replace(root,""))
    os.makedirs(Path(target).parent,exist_ok=True)
    # npArray2asc(get_asc_header_text(file),predict.numpy().reshape(image_size),target)
    npArray2asc(get_asc_header_text(file),predict,target)
print(time.time()-s,"s ellapsed")