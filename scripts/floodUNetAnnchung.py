import sys
import configparser
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

# models from https://github.com/milesial/Pytorch-UNet
from labMLlib.MLModels.unet_models import UNet
from labMLlib.utils.functions import *
from labMLlib.utils.imgPreprocess import imgPreprocessAnnchung_npz
from labMLlib.utils.loss import RMSELoss, absMSELoss

import time
import os
import platform
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
import re
import csv
import configparser
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="./settings/floodUNetAnnchung/floodUNetAnnchung.json",
        help="config file (.json) path",
    )
    opt = parser.parse_args()

    config_file_path = opt.cfg
    config = json.load(open(config_file_path,"r",encoding="utf-8"))
    ############## configuration ##################
    ## settings ##
    dataroot = config["settings"]["dataroot"]  # train, test, validation 存放的地方
    geoData_fn = config["settings"]["geodata_fn"]
    loss_record_fn = config["settings"]["loss_record_fn"]
    csv_fn = config["settings"]["csv_fn"]
    device = config["settings"]["device"]
    log_dir_root = config["settings"]["log_dir_root"]
    ckpt_dir = config["settings"]["ckpt_dir"]
    model_filename = config["settings"]["model_filename"]
    latest_model_filename = config["settings"]["latest_model_filename"]
    load_model_path = config["settings"]["load_model_path"]
    learning_curve_title = config["settings"]["learning_curve_title"]

    ## hyperparameters ##
    load_models = config["hyperparameters"]["load_models"]
    bilinear = config["hyperparameters"]["bilinear"]
    use_layer = config["hyperparameters"]["use_layer"]
    batch_size = int(config["hyperparameters"]["batch_size"])
    criterionAlgorithm = config["hyperparameters"]["criterionalgorithm"]
    optimizerAlgorithm = config["hyperparameters"]["optimizeralgorithm"]
    image_width = int(config["hyperparameters"]["image_width"])
    image_height = int(config["hyperparameters"]["image_height"])
    image_size = (image_width, image_height)  # image default size: (262,302)
    n_epoch = int(config["hyperparameters"]["n_epoch"])
    warmup_epoch = int(config["hyperparameters"]["warmup_epoch"])
    clip_value = float(config["hyperparameters"]["clip_value"])
    ## load dataset ##
    train_set = imgPreprocessAnnchung_npz(
        root_folder=dataroot, geoData_fn=geoData_fn, csv_fn=csv_fn, mode="train"
    )
    val_set = imgPreprocessAnnchung_npz(
        root_folder=dataroot,
        geoData_fn=geoData_fn,
        csv_fn=csv_fn,
        mode="validation",
    )
    test_set = imgPreprocessAnnchung_npz(
        root_folder=dataroot, geoData_fn=geoData_fn, csv_fn=csv_fn, mode="test"
    )
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    ## setup model, optimizer, criterion, dataset, gradient scaler, dataloader and some variables that will be used in training ##
    train_data_sample, answer_sample = train_set[0]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    unet = UNet(
        train_data_sample.shape[0], 1, bilinear=bilinear, use_layer=use_layer
    ).to(device)
    # if platform.system() == 'Linux': unet = torch.compile(unet)
    criterion_dict = {
        "MSELoss": nn.MSELoss(),
        "L1Loss": nn.L1Loss(),
        "HuberLoss": nn.HuberLoss(),
        "RMSELoss": RMSELoss(),
        "absMSELoss": absMSELoss(),
    }
    optimizer_dict = {
        "Adam": optim.Adam(
            unet.parameters(),
            lr=float(config["Adam"]["lr"]),
            betas=(float(config["Adam"]["beta1"]), float(config["Adam"]["beta2"])),
        ),
        "RMSprop": optim.RMSprop(
            unet.parameters(),
            lr=float(config["RMSprop"]["lr"]),
            weight_decay=float(config["RMSprop"]["weight_decay"]),
            momentum=float(config["RMSprop"]["momentum"]),
            foreach=float(True if config["RMSprop"]["foreach"].lower()=="true"else False),
        ),
    }
    ## calculate runtime and make directories ##
    start = time.strftime("%Y-%m-%d %H-%M")
    log_dir = os.path.join(log_dir_root, str(start))
    ckpt_dir += "+" + criterionAlgorithm + "+" + optimizerAlgorithm + f"({start})"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    ## some preparation ##
    model_structure = torch.save(unet, os.path.join(ckpt_dir, "structure.pt"))
    criterion = criterion_dict[criterionAlgorithm]
    optimizer = optimizer_dict[optimizerAlgorithm]
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    loss_record = {"train": [], "validation": [], "test": []}
    checkpoint = {}
    min_loss = 1000.0
    y_limit = (0.0, 1.0)
    epoch = 0
    total_epochs = 0
    start = time.time()
    if load_models:
        checkpoint = torch.load(os.path.join(ckpt_dir, load_model_path))
        unet = checkpoint["structure"]
        unet.load_state_dict(checkpoint["model_state_dict"])
        try:
            optimizer.load_state_dict(checkpoint[optimizerAlgorithm])
        except KeyError:
            print("optimzer state dict not found, skipping...")
        try:
            total_epochs = checkpoint["epoch"]
        except KeyError:
            print("epoch not found, skipping...")
        try:
            min_loss = checkpoint["loss"]
        except KeyError:
            print("loss not found, skipping...")
        unet.train()
    print("min_loss:", min_loss, "\naccumulated trained epoches:", total_epochs)
    current_label = 0
    images_with_same_label = []
    file = open(os.path.join(log_dir, loss_record_fn), "w")
    csv_file = csv.writer(file)
    csv_file.writerow(["epoch", "train loss", "validation loss", "test loss"])
    ###############################################

    ################ training #####################
    while epoch < n_epoch:
        unet.train()
        print(f"current epoch:({epoch+1}/{n_epoch})...")
        train_loss = 0
        for idx, data in enumerate(train_dl):
            train_data, answer = data
            train_data, answer = train_data.to(device), answer.to(device)
            pred = unet(train_data)
            loss = criterion(pred, answer)
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), clip_value)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_loss += loss.detach().cpu().item() / len(train_dl)
        loss_record["train"].append(train_loss)

        epoch += 1
        val_loss = validation(val_set, unet, criterion, device=device)
        loss_record["validation"].append(val_loss)
        if abs(val_loss) < abs(min_loss):
            if epoch > warmup_epoch:
                min_loss = val_loss
                print(
                    "Saving model (epoch = {:4d}, loss = {:4f})".format(epoch, min_loss)
                )
                checkpoint["epoch"] = total_epochs + epoch
                checkpoint["model_state_dict"] = unet.state_dict()
                checkpoint[optimizerAlgorithm] = optimizer.state_dict()
                checkpoint["loss"] = loss
                torch.save(
                    {
                        "structure": model_structure,
                        "epoch": total_epochs + epoch,
                        "model_state_dict": unet.state_dict(),
                        optimizerAlgorithm: optimizer.state_dict(),
                        "loss": loss.detach().cpu().item(),
                    },
                    os.path.join(ckpt_dir, model_filename),
                )
            test_loss = test(
                test_set,
                unet,
                criterion,
                log_dir,
                epoch,
                generateImage=True,
                device=device,
            )
        else:
            print(f"validation loss in epoch {epoch}:", val_loss)
            torch.save(
                {
                    "structure": model_structure,
                    "epoch": total_epochs + epoch,
                    "model_state_dict": unet.state_dict(),
                    optimizerAlgorithm: optimizer.state_dict(),
                    "loss": loss.detach().cpu().item(),
                },
                os.path.join(ckpt_dir, latest_model_filename),
            )
            test_loss = test(
                test_set,
                unet,
                criterion,
                log_dir,
                epoch,
                generateImage=False,
                device=device,
            )
        loss_record["test"].append(test_loss)
        csv_file.writerow([epoch, train_loss, val_loss, test_loss])
        plot_learning_curve(
            loss_record,
            os.path.join(log_dir, "learning_curve.png"),
            ylimit=(0, 1),
            loss=criterionAlgorithm,
            title=learning_curve_title,
        )

    ###############################################
    file.close()
    with open(os.path.join(log_dir,"time_used.txt"),"w") as f:
        f.writelines(f"time used: {time.time()-start}s")
    ## print some results and save learning curve ##
    print(f"time used: {time.time()-start}s")

    ################################################

    ############ some test results ################
    # gpu spec: 3060ti
    # time used to complete 20 epochs and their avg loss(run on windows)
    # loss type| RMSprop                 | Adam                     |
    # ----------------------------------------------------------------
    # HuberLoss| 367.4s (avg loss ~ 0.01)| 375.46s (loss = 0.0824)  |
    # MSELoss  | 370.45s (avg loss ~ 0.1)| 383.27s (loss = 0.092286)|
    # L1Loss   |367.74s (avg loss ~ 0.45)| 394.34s (loss = 0.371574)|
    # on linux it runs 3x faster! gpu spec: laptop 3050ti
    ############ some test results ################
    # HuberLoss+Adam
    # 10000 epoches: 90663.9436583519s ~~ 25.18hrs
    # RMSELoss+Adam
    # 3000 epoches: 27016.827490329742s ~~ 7.5hrs
    # mse loss+Adam
    # 5000 epoches: 27:51-16:49=11:02 ~~ 11.03hrs
    ################################################
