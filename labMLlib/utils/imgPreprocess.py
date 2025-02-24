import asyncio
import os
import platform
import random
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from pathlib import Path
from .functions import *


def imgPreprocess(
    dataset,
    log_dir,
    image_size=(256, 256),
    geoImgPath="archived/gan_H1100y/hy1correct.png",
    mode="train",
    forgan=False,
    z_dim=100,
):
    """An image preprocessing function that reset labels of images, normalize and concatenate images
    with rain and time information, save a sample answer image and then return the modified dataset.
    """
    RainmeterDataset = []
    labels = [
        5 + i * 5 for i in range(39)
    ]  # include all labels (5, 10, 15, ...) （195->test, 120->missing, 190->validation)
    labels.remove(195)
    labels.remove(120)
    labels.remove(190)
    if mode == "train":
        pass
    elif mode == "validation":
        labels = [190]
    elif mode == "test":
        labels = [195]
    elif mode == "all":
        labels.append(190)
        labels.append(195)
    else:
        raise ValueError('Mode must be one of "train"|"validation"|"test"|"all".')
    # rainTransform = transforms.Normalize(mean=mean(labels), std=stdev(labels))
    # timeTransform = transforms.Normalize(mean=mean([0,1,2,3]),std=stdev([0,1,2,3]))
    rainTransform = transforms.Normalize(mean=0.5, std=0.5)
    timeTransform = transforms.Normalize(mean=0.5, std=0.5)
    geoTransform = transforms.Compose(
        [
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            useGrayscale(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )
    geoImg = Image.open(geoImgPath)
    geoTensor = geoTransform(geoImg)  # geographic information
    for i, data in enumerate(dataset):
        img, label = data
        if 5 + label * 5 in labels:
            label = 5 + label * 5  # change label to rainmeter
            rainTensor = torch.zeros(img.shape).fill_(label)
            timeTensor = torch.zeros(img.shape).fill_(
                i % 4
            )  # fill the time consecutively as 0,1,2,3,0,1,2,3,...
            rainTensor = rainTransform(rainTensor)
            timeTensor = timeTransform(timeTensor)
            if forgan:
                RainmeterDataset.append(
                    (
                        torch.concat(
                            [
                                geoTensor,
                                rainTensor,
                                timeTensor,
                                Variable(
                                    torch.randn(z_dim - 3, image_size[0], image_size[1])
                                ),
                            ],
                            dim=0,
                        ),
                        img,
                    )  # (data, answer)
                )
            else:
                RainmeterDataset.append(
                    (
                        torch.concat([geoTensor, rainTensor, timeTensor], dim=0),
                        img,
                    )  # (data, answer)
                )
    torchvision.utils.save_image(
        [invTransform(RainmeterDataset[i][1]) for i in range(len(RainmeterDataset))],
        log_dir + f"/answer_{mode}.jpg",
        nrow=10,
    )
    return RainmeterDataset  # [(train_data1, answer1),...]


def imgPreprocessv2(
    image_size=(256, 256),
    log_dir="log",
    imgDir="archived",
    geoImgPath="archived/gan_H1100y/hy1correct.png",
    pattern=".*?0.png",
    mode="train",
    extension=".png",
    remove_zero=True,
    forgan=False,
    z_dim=100,
    normalize=True,
    rain_trans=False,
    time_trans=False,
):
    """An image preprocessing function that reset labels of images, normalize and concatenate images
    with rain and time information, save a sample answer image and then return the modified dataset.
    Output format:
    [(data, answer),...]
    data-->(geo,rain,time)
    forgan: [(z_dim_data, answer),...]"""
    assert mode == "train" or mode == "test" or mode == "validation" or mode == "all"
    RainmeterDataset = []
    data_path = search_dataset(extension, mode, imgDir)
    dataset = []
    geoTransform = transforms.Compose(
        [
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            useGrayscale(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )
    imgTransform = transforms.Compose(
        [
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            useGrayscale(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )
    geoImg = Image.open(geoImgPath)
    geoTensor = geoTransform(geoImg)
    for p in data_path:
        plist = [*Path(p).parts]
        while "." in plist:
            plist.remove(".")
        # print(p)
        if re.findall(pattern, plist[-1]) == [] or not remove_zero:
            with open(p, "rb") as f:
                img = Image.open(f)
                if normalize:
                    img = imgTransform(img)
                # img = transforms.CenterCrop(image_size)(img)
                # img = transforms.ToTensor()(img)
                label = plist[-2][:-2]
                t = plist[-1][-5]
                # print("t:",t)
                # print("label",label)
                dataset.append((img, int(label), int(t)))
    # print("dataset:",dataset)
    rainlist = []
    timelist = []
    for data in dataset:
        img, label, t = data
        rainlist.append(label)
        timelist.append(t)
    rainlist = np.array(rainlist)
    timelist = np.array(timelist)
    rainTransform = transforms.Normalize(mean=np.mean(rainlist), std=np.std(rainlist))
    timeTransform = transforms.Normalize(mean=np.mean(timelist), std=np.mean(timelist))
    for i, data in enumerate(dataset):
        img, label, t = data
        rainTensor = torch.zeros(img.shape).fill_(label)
        timeTensor = torch.zeros(img.shape).fill_(
            t
        )  # fill the time consecutively as 0,1,2,3,0,1,2,3,...
        if normalize:
            if rain_trans:
                rainTensor = rainTransform(rainTensor)
            if time_trans:
                timeTensor = timeTransform(timeTensor)
        if forgan:
            RainmeterDataset.append(
                (
                    torch.concat(
                        [
                            geoTensor,
                            rainTensor,
                            timeTensor,
                            Variable(
                                torch.randn(z_dim - 3, image_size[0], image_size[1])
                            ),
                        ],
                        dim=0,
                    ),
                    img,
                )  # (data, answer)
            )
        else:
            RainmeterDataset.append(
                (
                    torch.concat([geoTensor, rainTensor, timeTensor], dim=0),
                    img,
                )  # (data, answer)
            )
    torchvision.utils.save_image(
        [invTransform(RainmeterDataset[i][1]) for i in range(len(RainmeterDataset))],
        log_dir + f"/answer_{mode}.jpg",
        nrow=10,
    )
    if normalize:
        torchvision.utils.save_image(
            [RainmeterDataset[i][1] for i in range(len(RainmeterDataset))],
            log_dir + f"/answer_{mode}_normalize.jpg",
            nrow=10,
        )
    return RainmeterDataset


def imgPreprocessv3(
    image_size=(256, 256),
    log_dir="log",
    imgDir="archived",
    geoImgPath="archived/gan_H1100y/hy1correct.png",
    pattern=".*?0.png",
    mode="train",
    extension=".png",
    remove_zero=True,
    forgan=False,
    z_dim=100,
    normalize=True,
):
    """An image preprocessing function that reset labels of images, normalize and concatenate images
    with rain and time information, save a sample answer image and then return the modified dataset.
    Output format:
    [(data, answer, rain_label, time_label),...]
    forgan: [(z_dim_data, answer, rain_label, time_label),...]"""
    assert mode == "train" or mode == "test" or mode == "validation" or mode == "all"
    RainmeterDataset = []
    data_path = search_dataset(extension, mode, imgDir)
    dataset = []
    rainTransform = transforms.Normalize(mean=0.5, std=0.5)
    timeTransform = transforms.Normalize(mean=0.5, std=0.5)
    geoTransform = transforms.Compose(
        [
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            useGrayscale(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )
    imgTransform = transforms.Compose(
        [
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            useGrayscale(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )
    geoImg = Image.open(geoImgPath)
    geoTensor = geoTransform(geoImg)
    for p in data_path:
        plist = [*Path(p).parts]
        while "." in plist:
            plist.remove(".")
        # print(p)
        if re.findall(pattern, plist[-1]) == [] or not remove_zero:
            with open(p, "rb") as f:
                img = Image.open(f)
                if normalize:
                    img = imgTransform(img)
                # img = transforms.CenterCrop(image_size)(img)
                # img = transforms.ToTensor()(img)
                label = plist[-2][:-2]
                t = plist[-1][-5]
                # print("t:",t)
                # print("label",label)
                dataset.append((img, int(label), int(t)))
    # print("dataset:",dataset)
    return dataset


# TODO: need to test it.
# PS: need to add mulltipthreading to accelerate the speed of processing data.
def imgPreprocessCollapse(
    image_size=(718, 998),
    log_dir="log",
    imgDir="archived",
    geoImgPath="archived/gan_H1100y/hy1correct.png",
    mode="train",
    extension=".png",
    num_workers=16,
    use_async=True,
):
    """An image preprocessing function that creates collapse dataset
    with dimension (n,image_width,image_height) as n is the number of images
    in the dataset.
    Output format:
    [{"Sec_output":[xxx],"Rain":[xxx]},...]
    """
    assert (
        mode == "train" or mode == "test" or mode == "validation" or mode == "all"
    ), "Mode must be one of train|test|validation|all for imgPreprocessCollapse function."
    if mode == "all":
        return (
            imgPreprocessCollapse(
                image_size, log_dir, imgDir, geoImgPath, "train", extension, num_workers
            )
            + imgPreprocessCollapse(
                image_size, log_dir, imgDir, geoImgPath, "test", extension, num_workers
            )
            + imgPreprocessCollapse(
                image_size,
                log_dir,
                imgDir,
                geoImgPath,
                "validation",
                extension,
                num_workers,
            )
        )
    if platform.system() == "Windows":
        match_folder_pattern = "result\\\\.*?\\\\(.*?)%s" % extension
        get_foldername_pattern = "result\\\\(.*?)\\\\.*?"
    else:
        match_folder_pattern = "result/.*?/(.*?)%s" % extension
        get_foldername_pattern = "result/(.*?)/.*?"
    data_path = search_dataset(extension, mode, imgDir, pattern=match_folder_pattern)
    # data_path = search_dataset(extension,mode,imgDir)
    # print(data_path)
    data_path = sort_by_sub_dirname(
        data_path, subdirname_pattern="WHIRAM_c384_.*_00_(.*?)_V3.3"
    )
    dataset = []
    img_transform = transforms.Compose(
        [
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            useGrayscale(),
        ]
    )

    # seperate by case
    async def worker(
        data_path_val,
        get_foldername_pattern,
    ):
        data_path_val = sort_by_sub_dirname(
            data_path_val, subdirname_pattern=get_foldername_pattern
        )
        case_data = {}
        for key, seperated_p in data_path_val.items():
            # Z:\XDS\WHIRAM_c384_amip_hist_00_197901_V3.3\result\Sec_output
            # plist = p.split(os.sep)
            # while '.' in plist: plist.remove('.')
            # print(p)
            # print("seperated_p",seperated_p)
            if (
                key == "Basic"
            ):  # skipping folder Basic because all data inside it are same.
                continue
            seperated_p = sorted(
                seperated_p,
                key=lambda x: int(re.findall("(\d+)", [*Path(x).parts][-1])[0]),
            )
            for p in seperated_p:
                with open(p, "rb") as f:
                    img = Image.open(f)
                    img = img_transform(img)
                    # torchvision.utils.save_image([img],"test.png",now=10)
                    if key not in case_data.keys():
                        case_data[key] = []
                    case_data[key].append(img)
        return case_data

    async def workerPool():
        queue = asyncio.Queue(maxsize=num_workers)
        tasks = []
        for data_path_val in data_path.values():
            # for each case, seperate by subdirectories
            # print(data_path_val)
            await queue.put(worker(data_path_val, get_foldername_pattern))
            while queue.empty() is False:
                tasks.append(await queue.get())
        result = await asyncio.gather(*tasks, return_exceptions=True)
        return result

    if use_async:
        dataset = asyncio.run(workerPool())
    else:
        print("Didn't use async in function imgPreprocessCollapse!")
        dataset = []
        for data_path_val in data_path.values():
            data_path_val = sort_by_sub_dirname(
                data_path_val, subdirname_pattern=get_foldername_pattern
            )
            case_data = {}
            for key, seperated_p in data_path_val.items():
                # Z:\XDS\WHIRAM_c384_amip_hist_00_197901_V3.3\result\Sec_output
                # plist = p.split(os.sep)
                # while '.' in plist: plist.remove('.')
                # print(p)
                # print("seperated_p",seperated_p)
                if (
                    key == "Basic"
                ):  # skipping folder Basic because all data inside it are same.
                    continue
                seperated_p = sorted(
                    seperated_p,
                    key=lambda x: int(re.findall("(\d+)", [*Path(x).parts][-1])[0]),
                )
                for p in seperated_p:
                    with open(p, "rb") as f:
                        img = Image.open(f)
                        img = img_transform(img)
                        # torchvision.utils.save_image([img],"test.png",now=10)
                        if key not in case_data.keys():
                            case_data[key] = []
                        case_data[key].append(img)
            dataset.append(case_data)
    return dataset

def imgPreprocessCollapse_npz(
    image_size=(256, 256),
    root_folder="archived_Annchung_npz",
    csv_fn="raintype.csv",
    delimiter=",",
    geoData_fn="hy1correct.asc",
    mode="train",
    dtype=torch.float32,
):
    """An image preprocessing function that reset labels of images, normalize and concatenate images
    with rain and time information, save a sample answer image and then return the modified dataset.
    """
    assert mode == "train" or mode == "test" or mode == "validation" or mode == "all"
    csv_path = os.path.join(root_folder, csv_fn)
    npz_list = os.path.join(root_folder, mode + ".npz")
    dataset = []
    with np.load(npz_list) as data:
        # print(data.files)
        geoDataCheck = []
        for i in data.files:
            geoDataCheck += [*Path(i).parts]
        if geoData_fn not in geoDataCheck:
            raise KeyError("geoData_fn not found in npz file")
        elif geoDataCheck.count(geoData_fn) > 1:
            raise KeyError(
                "geoData_fn found multiple times in npz file and will lead dataset error, aborting..."
            )
        else:
            for i in data.files:
                if geoData_fn in [*Path(i).parts]:
                    geoData_fn = i
                    # print(geoData_fn)
                    break
        geoData = data[geoData_fn]
        geoData = torch.from_numpy(geoData)
        geoData = transforms.CenterCrop(image_size)(geoData)
        geoData.unsqueeze_(0)
        df = pd.read_csv(csv_path, delimiter=delimiter)
        for key, value in data.items():
            if key == geoData_fn:
                continue
            plist = [*Path(key).parts]
            label = plist[-2][:-1]
            type_label = plist[-2][-1]
            t = plist[-1][-5]
            value = torch.from_numpy(value)
            value = transforms.CenterCrop(image_size)(value)
            value.unsqueeze_(0)
            # print(label, type_label, t, value.shape)
            assert int(t) > 0, "time should not less than zero!"
            labelTensor = torch.zeros(value.shape).fill_(int(label))
            typeTensor = torch.zeros(value.shape).fill_(df[type_label][int(t) - 1])
            tTensor = torch.zeros(value.shape).fill_(int(t))
            data = torch.concat([geoData, labelTensor, typeTensor, tTensor], dim=0)
            data, value = data.type(dtype), value.type(dtype)
            # print(data.shape)
            # print(torch.mean(data[0]),torch.mean(data[1]),torch.mean(data[2]),torch.mean(data[3]))
            dataset.append((data, value))
    return dataset

def imgPreprocessAnnchung_npz(
    image_size=(256, 256),
    root_folder="archived_Annchung_npz",
    csv_fn="raintype.csv",
    delimiter=",",
    geoData_fn="hy1correct.asc",
    mode="train",
    dtype=torch.float32,
):
    """An image preprocessing function that reset labels of images, normalize and concatenate images
    with rain and time information, save a sample answer image and then return the modified dataset.
    """
    assert mode == "train" or mode == "test" or mode == "validation" or mode == "all"
    csv_path = os.path.join(root_folder, csv_fn)
    npz_list = os.path.join(root_folder, mode + ".npz")
    dataset = []
    with np.load(npz_list) as data:
        # print(data.files)
        geoDataCheck = []
        for i in data.files:
            geoDataCheck += [*Path(i).parts]
        if geoData_fn not in geoDataCheck:
            raise KeyError("geoData_fn not found in npz file")
        elif geoDataCheck.count(geoData_fn) > 1:
            raise KeyError(
                "geoData_fn found multiple times in npz file and will lead dataset error, aborting..."
            )
        else:
            for i in data.files:
                if geoData_fn in [*Path(i).parts]:
                    geoData_fn = i
                    # print(geoData_fn)
                    break
        geoData = data[geoData_fn]
        geoData = torch.from_numpy(geoData)
        geoData = transforms.CenterCrop(image_size)(geoData)
        geoData.unsqueeze_(0)
        df = pd.read_csv(csv_path, delimiter=delimiter)
        for key, value in data.items():
            if key == geoData_fn:
                continue
            plist = [*Path(key).parts]
            label = plist[-2][:-1]
            type_label = plist[-2][-1]
            t = plist[-1][-5]
            value = torch.from_numpy(value)
            value = transforms.CenterCrop(image_size)(value)
            value.unsqueeze_(0)
            # print(label, type_label, t, value.shape)
            assert int(t) > 0, "time should not less than zero!"
            labelTensor = torch.zeros(value.shape).fill_(int(label))
            typeTensor = torch.zeros(value.shape).fill_(df[type_label][int(t) - 1])
            tTensor = torch.zeros(value.shape).fill_(int(t))
            data = torch.concat([geoData, labelTensor, typeTensor, tTensor], dim=0)
            data, value = data.type(dtype), value.type(dtype)
            # print(data.shape)
            # print(torch.mean(data[0]),torch.mean(data[1]),torch.mean(data[2]),torch.mean(data[3]))
            dataset.append((data, value))
    return dataset


def imgPreprocessAnnchung_npz_with_filepath(
    image_size=(256, 256),
    root_folder="archived_Annchung_npz",
    csv_fn="raintype.csv",
    delimiter=",",
    geoData_fn="hy1correct.asc",
    mode="train",
    dtype=torch.float32,
):
    """An image preprocessing function that reset labels of images, normalize and concatenate images
    with rain and time information, save a sample answer image and then return the modified dataset.
    """
    assert mode == "train" or mode == "test" or mode == "validation" or mode == "all"
    csv_path = os.path.join(root_folder, csv_fn)
    npz_list = os.path.join(root_folder, mode + ".npz")
    dataset = []
    datapath = []
    with np.load(npz_list) as data:
        # print(data.files)
        geoDataCheck = []
        for i in data.files:
            geoDataCheck += [*Path(i).parts]
        if geoData_fn not in geoDataCheck:
            raise KeyError("geoData_fn not found in npz file")
        elif geoDataCheck.count(geoData_fn) > 1:
            raise KeyError(
                "geoData_fn found multiple times in npz file and will lead dataset error, aborting..."
            )
        else:
            for i in data.files:
                if geoData_fn in [*Path(i).parts]:
                    geoData_fn = i
                    # print(geoData_fn)
                    break
        geoData = data[geoData_fn]
        geoData = torch.from_numpy(geoData)
        geoData = transforms.CenterCrop(image_size)(geoData)
        geoData.unsqueeze_(0)
        df = pd.read_csv(csv_path, delimiter=delimiter)
        for key, value in data.items():
            if key == geoData_fn:
                continue
            plist = [*Path(key).parts]
            label = plist[-2][:-1]
            type_label = plist[-2][-1]
            t = plist[-1][-5]
            value = torch.from_numpy(value)
            value = transforms.CenterCrop(image_size)(value)
            value.unsqueeze_(0)
            # print(label, type_label, t, value.shape)
            assert int(t) > 0, "time should not less than zero!"
            labelTensor = torch.zeros(value.shape).fill_(int(label))
            typeTensor = torch.zeros(value.shape).fill_(df[type_label][int(t) - 1])
            tTensor = torch.zeros(value.shape).fill_(int(t))
            data = torch.concat([geoData, labelTensor, typeTensor, tTensor], dim=0)
            data, value = data.type(dtype), value.type(dtype)
            # print(data.shape)
            # print(torch.mean(data[0]),torch.mean(data[1]),torch.mean(data[2]),torch.mean(data[3]))
            dataset.append((data, value))
            datapath.append(Path(key))
    return dataset,datapath