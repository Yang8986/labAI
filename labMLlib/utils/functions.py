import torch
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms

import time
import os
import platform
import asyncio
from PIL import Image
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from pathlib import Path
import numpy as np
import platform
import re
import random
import pandas as pd
from .transform import NoneTransform


def setTorchNumpySeed(myseed=12345):
    # seed for all random number generator so the random is under control.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)


def useGrayscale():
    if platform.system() == "Windows":
        return transforms.Grayscale()
    elif platform.system() == "Linux":
        return NoneTransform()


def invTransform(tensor, tensorMean=0.5, tensorStd=0.5):
    """performs inverse transformation for images after normalization (denormalized)."""
    acceptType = (list, int, float, tuple)
    invMean = []
    invStd = []
    dim = 0
    if isinstance(tensorMean, acceptType) and isinstance(tensorStd, acceptType):
        if isinstance(tensorMean, (list, tuple)):
            dim = len(tensorMean)
            invMean = [-i for i in tensorMean]
            invStd = [1 / i for i in tensorStd]
        else:
            dim = 1
            invMean = -tensorMean
            invStd = 1 / tensorStd
    else:
        raise TypeError("Type of mean or std inappropriate!")
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0 for _ in range(dim)], std=invStd),
            transforms.Normalize(mean=invMean, std=[1.0 for _ in range(dim)]),
        ]
    )
    inv_tensor = invTrans(tensor)
    return inv_tensor


def search_dataset(extension, mode, imgDir="archived", pattern=None):
    mode_list = ["train", "test", "validation", "basic"]
    assert (
        mode in mode_list
    ), f"Mode must be one of {'|'.join(mode_list+'|')} for search_dataset function."
    data_path = []
    default_path = os.getcwd()
    print("default_path:", default_path)
    imgDir = os.path.join(imgDir, mode)
    print("imgDir:", imgDir)
    # os.chdir(imgDir)
    for dirpath, dirnames, filenames in os.walk(imgDir):
        for filename in [f for f in filenames if f.endswith(extension)]:
            p = os.path.join(dirpath, filename)
            if pattern != None:
                if re.findall(pattern, p) == []:
                    continue
            p = [*Path(dirpath).parts]# here will remove the \\ in windows latter path like Z:
            while "." in p:
                p.remove(".")
            if re.findall("[A-Za-z]:", p[0]) != []:
                if "\\" not in p[0]:
                    p[0] += "\\"
            if platform.system() == "Linux":
                p.insert(0, "/")
            p = os.path.join(*p, filename)
            if imgDir == ".":
                p = os.path.join(default_path, p)
                if not os.path.exists(p):
                    raise ValueError("path %s not exist!" % p)
            data_path.append(p)
    # os.chdir(os.path.dirname(__file__))
    # print("data_path:\n",data_path)
    return data_path


def validation(val_set, model, criterion, device="cpu"):
    """A function to validate the model and return the validation loss."""
    model.eval()  # set model to evaluation mode
    total_loss = 0
    for x, y in val_set:  # iterate through the dataloader
        x, y = x.to(device), y.to(device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            pred = model(x)  # forward pass (compute output)
            pred = pred.view(y.shape)
            loss = criterion(pred, y)
        total_loss += loss.detach().cpu().item()  # accumulate loss
    total_loss /= len(val_set)  # compute averaged loss
    return total_loss

def validationCollapse(val_set, model, criterion, device="cpu"):
    """A function to validate the model and return the validation loss."""
    model.eval()  # set model to evaluation mode
    total_loss = 0
    for _,x, y in val_set:  # iterate through the dataloader
        x, y = x.to(device), y.to(device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            pred = pred.view(y.shape)
            loss = criterion(pred, y)
        total_loss += loss.detach().cpu().item()  # accumulate loss
    return float(total_loss/len(val_set))

def test(
    test_set, model, criterion, log_dir, epoch, generateImage=True, device="cpu"
):
    """A function to test the trained model with given dataset and save images with both generated images and answers."""
    if generateImage:
        os.makedirs(log_dir, exist_ok=True)
    model.eval()  # set model to evaluation mode
    # create a list to store the generated images
    generatedImgs = []
    total_loss = 0
    for test_data, answer in test_set:
        with torch.no_grad():  # disable gradient calculation
            test_data, answer = test_data.to(device), answer.to(device)
            generatedImg = model(test_data.unsqueeze(0)).data.squeeze(
                0
            )  # get image from model with input as test_data
            # answer = invTransform(answer,0.5,0.5)                               # perform invTransfrom to change back the picture after normalization
            # generatedImg = invTransform(generatedImg,0.5,0.5)
            # torch.argmax(generatedImg, 1)                                       # set the maximum argument as 1 （same concept as scaler)
            generatedImg = generatedImg.view(answer.shape)
            generatedImgs.append(generatedImg)
            generatedImgs.append(answer)
            loss = criterion(generatedImg, answer)
            total_loss += loss.detach().cpu().item()
    total_loss /= len(test_set)
    filename = os.path.join(log_dir, f"Epoch_{epoch:03d}(loss={total_loss:.6f}).jpg")
    if generateImage:
        # save image with 10 per row (form as generatedImg then answer)
        torchvision.utils.save_image(generatedImgs, filename, nrow=10)
    model.train() # set back to train
    return total_loss

def testCollapse(
    test_set, model, criterion, log_dir, epoch, generateImage=True, device="cpu"
):
    """A function to test the trained model with given dataset and save images with both generated images and answers."""
    if generateImage:
        os.makedirs(log_dir, exist_ok=True)
    model.eval()  # set model to evaluation mode
    # create a list to store the generated images
    total_loss = 0
    count=0
    for key,test_data, answer in test_set:
        with torch.no_grad():  # disable gradient calculation
            test_data, answer = test_data.to(device), answer.to(device)
            generatedImg = model(test_data)
            generatedImg = generatedImg.view(answer.shape)
            loss = criterion(generatedImg, answer)
            if generateImage:
                torchvision.utils.save_image([generatedImg,answer], os.path.join(log_dir,f"{count}_{key}_{loss}.jpg"), nrow=2)
                count+=1
            total_loss += loss.detach().cpu().item()
    total_loss /= len(test_set)
    model.train() # set back to train
    return total_loss

def plot_learning_curve(
    loss_record, filename, ylimit=(0.0, 1.0), loss="MSE Loss", title="", xlabel="Epochs"
):
    """A function to plot the learning curve with given title name and file name."""
    lower, upper = ylimit
    total_steps = len(loss_record["train"])
    x_1 = range(total_steps)
    x_2 = x_1[:: len(loss_record["train"]) // len(loss_record["validation"])]
    x_3 = x_1[:: len(loss_record["train"]) // len(loss_record["test"])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record["train"], c="tab:red", label="train")
    plt.plot(x_2, loss_record["validation"], c="tab:cyan", label="validation")
    plt.plot(x_3, loss_record["test"], c="tab:orange", label="test")
    plt.plot()
    plt.ylim(lower, upper)
    plt.xlabel(xlabel)
    plt.ylabel(loss)
    plt.title("Learning curve of {}".format(title))
    plt.legend()
    plt.savefig(filename)
    plt.close()


def dim_expand(dimension):
    if isinstance(dimension, int):
        return dimension * dimension
    elif isinstance(dimension, tuple):
        expanded_dim = 1
        for i in dimension:
            expanded_dim *= i
        return expanded_dim
    else:
        raise TypeError("Only int or tuple can be accepted.")


def sort_by_subdir_level(data_path, subdir_level):
    assert isinstance(data_path, list), "data_path must be a list."
    path_dict = {}
    for i in data_path:
        assert isinstance(i, str), "data_path must be a list of string."
        assert not os.path.isdir(i), "data_path must not be directory."
        assert subdir_level > 0, "subdir_level must be greater than 0."
        sort_dirname = [*Path(i).parts][-(subdir_level + 1)]
        if sort_dirname not in path_dict.keys():
            path_dict[sort_dirname] = []
        path_dict[sort_dirname].append(i)
    return path_dict


def sort_by_sub_dirname(data_path, subdirname_pattern):
    """
    subdirname_pattern: use regex as the pattern like "(.*?)0.png" or "\d+"
    """
    assert isinstance(data_path, list), "data_path must be a list."
    path_dict = {}
    for i in data_path:
        assert isinstance(i, str), "data_path must be a list of string."
        assert not os.path.isdir(i), "data_path must not be directory."
        assert isinstance(
            subdirname_pattern, str
        ), "subdirname_pattern must be a string."
        sort_dirname_list = re.findall(subdirname_pattern, i)
        if len(sort_dirname_list) > 0 and len(sort_dirname_list) == 1:
            sort_dirname = sort_dirname_list[0]
        elif len(sort_dirname_list) == 0:
            raise ValueError(
                "subdirname_pattern must match at least one result. Cannot match from %s"
                % i
            )
        else:
            raise ValueError(
                "subdirname_pattern must only match one pattern. Got result %s"
                % sort_dirname_list
            )
        if sort_dirname not in path_dict.keys():
            path_dict[sort_dirname] = []
        path_dict[sort_dirname].append(i)
    return path_dict

def predictDataTypeConvert(config,ALLOW_INPUT_TYPES):
        if isinstance(config,dict):
            for key,val in config.items():
                if isinstance(val,dict):
                    val = predictDataTypeConvert(val,ALLOW_INPUT_TYPES)
                elif isinstance(val,str):
                    if val!="":
                        try:
                            val_type = eval(f"type({val}).__name__")
                            if val_type in ALLOW_INPUT_TYPES:
                                config[key]=eval(val)
                        except:
                            pass
        elif isinstance(config,str):
            if config!="":
                try:
                    val_type = eval(f"type({config}).__name__")
                    if val_type in ALLOW_INPUT_TYPES:
                        config=eval(config)
                except:
                    pass
        return config