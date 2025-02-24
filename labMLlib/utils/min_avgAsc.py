import os
import time
from pathlib import Path
import argparse
import re

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    type=str,
    # default="./datasets/XDS/data_sample/1.asc",
    required=True,
    help="input asc path(or dir)",
)

parser.add_argument(
    "--output",
    type=str,
    default="output",
    help="output directory",
)

opt = parser.parse_args()

if Path(opt.input).is_dir():
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

target = opt.output
s = time.time()



import torch
import torch.nn.functional as F
import numpy as np

def asc2NPArray(p,verbose=False,dtype=np.float32):
    with open(p, "r") as f:
        if verbose:
            print("path: " + p)
        ncols = f.readline().split()[1]
        nrows = f.readline().split()[1]
        xllcorner = f.readline().split()[1]
        try:
            yllcorner = f.readline().split()[1]
        except:
            print("file: ",p)
        # yllcorner = f.readline().split()[1]
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
                # [float(j) if j != NODATA_value else float(6172.4023) for j in i.split()]
                [float(j) for j in i.split()]
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

def get_asc_header_text(asc_file_path: str)->dict:
    with open(asc_file_path, "r") as f:
        lines = f.readlines()
        header = lines[:6]
    return header

def npArray2asc(header, imgData, file_path):
    header_dict = {}
    for i in header:
        header_dict[i.split()[0]] = i.split()[1]
    ncols = header_dict["ncols"]
    nrows = header_dict["nrows"]
    nodata_value = header_dict["NODATA_value"]
    newNODATA_value = -9999
    # print(nodata_value)
    # offset = get_offset((int(nrows),int(ncols)),imgData.shape)
    header = "".join(header)
    header = header.replace(nodata_value, str(newNODATA_value))
    print("header: ",header)
    imgData = np.round(imgData,decimals=4)
    imgData = imgData.astype(str)
    imgData = np.where(imgData == "-9999.0", newNODATA_value, imgData)
    # imgData = np.pad(imgData,offset,mode="constant",constant_values=newNODATA_value)
    # print(imgData[0][0])
    print(imgData[0][0])
    imgData = np.array([[" " + j for j in i] for i in imgData])
    imgData = np.array([" ".join(i) for i in imgData])
    imgData = "\n".join(imgData)
    print("check:",len(imgData),len(imgData[0]))
    with open(file_path, "w") as f:
        f.write(header + imgData)

for i,file in enumerate(files):
    predict_data = (asc2NPArray(file)[1])
    # predict_data = predict_data.numpy()
    print(predict_data[0][0])
    if i==0:
        min_data = predict_data
        avg_data = np.zeros_like(predict_data)
        count = 0
    min_data = np.minimum(min_data,predict_data)
    count += 1
    avg_data = avg_data + (predict_data - avg_data) / count
    print(f"第{i}個file:{file},形狀:{predict_data.shape}")
npArray2asc(get_asc_header_text(file),min_data,os.path.join(target, "min.asc"))
npArray2asc(get_asc_header_text(file),avg_data,os.path.join(target, "avg.asc"))
print(time.time()-s,"s ellapsed")