import sys
import os
import time
import configparser
import json
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
from vit_pytorch.efficient import ViT
from nystrom_attention import Nystromformer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import csv
import configparser
import argparse
from datetime import datetime
from pathlib import Path

from labMLlib.utils.loss import *
from labMLlib.utils.functions import (
    test,
    dim_expand,
    validation,
    plot_learning_curve,
)
from labMLlib.utils.imgPreprocess import imgPreprocessAnnchung_npz
from labMLlib.datasets.collapseVITDataset import collapseVITDataset
from labMLlib.Optim.optimizerDict import optimDict
from labMLlib.Loss.lossDict import lossDict
from torch.cuda.amp import GradScaler, autocast

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="./settings/VITNystromAnnchung.ini",
        help="config file (.ini) path",
    )
    opt = parser.parse_args()

    config_file_path = opt.cfg
    config = json.load(open(config_file_path,"r",encoding="utf-8"))
    ############## configuration ##################
    ## settings ##
    # [settings]
    dataroot = Path(config["settings"]["dataroot"])  # train, test, validation 存放的地方
    geoData_fn = config["settings"]["geodata_fn"]
    loss_record_fn = config["settings"]["loss_record_fn"]
    csv_fn = config["settings"]["csv_fn"]
    device = config["settings"]["device"]
    log_dir_root = Path(config["settings"]["log_dir_root"])
    ckpt_dir = config["settings"]["ckpt_dir"]
    model_filename = config["settings"]["model_filename"]
    latest_model_filename = config["settings"]["latest_model_filename"]
    load_model_path = Path(config["settings"]["load_model_path"])
    learning_curve_title = config["settings"]["learning_curve_title"]

    ## hyperparameters ##
    # [hyperparameters]
    # 载入模型存档点
    load_models = config["hyperparameters"]["load_models"]
    batch_size = config["hyperparameters"]["batch_size"]
    criterionAlgorithm = config["hyperparameters"]["criterionalgorithm"]
    optimizerAlgorithm = config["hyperparameters"]["optimizeralgorithm"]
    image_width = config["hyperparameters"]["image_width"]
    image_height = config["hyperparameters"]["image_height"]
    image_size = (image_width, image_height)  # image default size: (262,302)
    patch_size = config["hyperparameters"]["patch_size"]
    n_epoch = config["hyperparameters"]["n_epoch"]
    warmup_epoch = config["hyperparameters"]["warmup_epoch"]
    sample_interval = config["hyperparameters"]["sample_interval"]
    use_gradient_accumulations = config["hyperparameters"][
        "use_gradient_accumulations"
    ]
    gradient_accumulations = config["hyperparameters"]["gradient_accumulations"]
    clip_value = config["hyperparameters"]["clip_value"]

    # [padding]
    p_up = config["padding"]["p_up"]
    p_down = config["padding"]["p_down"]
    p_left = config["padding"]["p_left"]
    p_right = config["padding"]["p_right"]
    ##################################################
    start = datetime.now()
    time_const = start.strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join(log_dir_root, time_const)
    log_dir+=f"({criterionAlgorithm}+{optimizerAlgorithm})"
    ckpt_dir += "+" + criterionAlgorithm + "+" + optimizerAlgorithm + f"({str(start).replace(':','-')})"
    ckpt_dir = Path(ckpt_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    padding = (p_up, p_down, p_left, p_right)
    padding_size = (p_up, p_left)
    # reversed_image_size = (718,998)

    print("initializing train dataset...")
    # train_dataset = collapseVITDataset(mode="train",imgDir=os.path.join(os.getcwd(),"archived"),padding=padding,num_workers=16)
    train_dataset = imgPreprocessAnnchung_npz(
        image_size=image_size,
        root_folder=dataroot,
        csv_fn=csv_fn,
        geoData_fn=geoData_fn,
        mode="train",
    )
    print("initializing test dataset...")
    test_dataset = imgPreprocessAnnchung_npz(
        image_size=image_size,
        root_folder=dataroot,
        csv_fn=csv_fn,
        geoData_fn=geoData_fn,
        mode="test",
    )
    print("initializing validation dataset...")
    validation_dataset = imgPreprocessAnnchung_npz(
        image_size=image_size,
        root_folder=dataroot,
        csv_fn=csv_fn,
        geoData_fn=geoData_fn,
        mode="validation",
    )
    train_data_sample, answer_sample = train_dataset[0]
    channels = train_data_sample.shape[0]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=16
    )
    assert (
        len(image_size) == 2
    ), "image_size must be strictly just width and height, the three dimension output of img function haven't developed!"
    padded_image_size = (
        image_size[0] + padding_size[0] * 2,
        image_size[1] + padding_size[1] * 2,
    )

    efficient_transformer = Nystromformer(dim=1024, depth=6, heads=8, num_landmarks=256)

    v = ViT(
        image_size=padded_image_size,
        patch_size=patch_size,
        num_classes=dim_expand(image_size),
        transformer=efficient_transformer,
        dim=1024,
        channels=channels,
    ).to(device)

    criterion = lossDict(**config['criterion'][criterionAlgorithm])[criterionAlgorithm]
    optimizer = optimDict(v.parameters(),**config['optimizer'][optimizerAlgorithm])[optimizerAlgorithm]
    model_structure = torch.save(v, os.path.join(ckpt_dir, "structure.pt"))
    scaler = GradScaler()
    min_loss = 1000.0
    min_image_loss = 1000.0
    min_validation_loss = 1000.0
    total_epochs = 0
    with open(os.path.join(log_dir, loss_record_fn), "w") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["epoch", "train loss", "validation loss", "test loss"])
    torch.save(v, os.path.join(ckpt_dir, "structure.pt"))
    loss_record = {"train": [], "validation": [], "test": []}
    checkpoint = {}
    if load_models:
        checkpoint = torch.load(os.path.join(ckpt_dir, load_model_path))
        v = checkpoint["structure"]
        v.load_state_dict(checkpoint["model_state_dict"])
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
        v.train()
    print("min_loss:", min_loss, "\naccumulated trained epoches:", total_epochs)
    for epoch in range(1, n_epoch + 1):
        for i, data in enumerate(dataloader):
            batches_done = (epoch - 1) * len(dataloader) + i + 1
            optimizer.zero_grad()
            img, answer = data
            img, answer = img.to(device), answer.to(device)
            # print(img.shape," and ",answer.shape)
            preds = v(img)  # (1,img_size*img_size)
            # v.zero_grad()
            preds = preds.view(-1, 1, *image_size)
            # print(preds.shape)
            loss = criterion(preds, answer)
            torch.nn.utils.clip_grad_norm_(v.parameters(), clip_value)
            if device == "cuda" and use_gradient_accumulations:
                scaler.scale(loss / gradient_accumulations).backward()
                if (i + 1) % gradient_accumulations == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    # v.zero_grad()
            else:
                loss.backward()
                optimizer.step()
            loss = loss.item()
            loss_record["train"].append(loss)
            validation_loss = validation(
                val_set=validation_dataset, model=v, criterion=criterion, device=device
            )
            loss_record["validation"].append(validation_loss)
            test_loss = test(
                test_dataset,
                v,
                criterion,
                os.path.join(
                    log_dir,
                    "[Epoch %d] [Batch %d] [train loss %f] [validation loss %f]"
                    % (
                        epoch,
                        i + 1,
                        loss,
                        validation_loss,
                    ),
                ),
                epoch=epoch,
                generateImage=False,
                device=device,
            )
            loss_record["test"].append(test_loss)
            print(
                "[Epoch %d/%d] [Batch %d/%d] [train loss: %f] [validation loss: %f]"
                % (
                    epoch,
                    n_epoch,
                    i + 1,
                    len(dataloader),
                    loss,
                    validation_loss,
                )
            )
            with open(os.path.join(log_dir, loss_record_fn), "a") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(
                    [f"{epoch}", f"{loss}", f"{validation_loss}", f"{test_loss}"]
                )
            saved = False
            if loss < min_loss and epoch > warmup_epoch:
                if not saved:
                    print(
                        "Saving model (epoch = {:4d}, loss = {:4f})".format(
                            epoch, min_loss
                        )
                    )
                    checkpoint["epoch"] = total_epochs + epoch
                    checkpoint["model_state_dict"] = v.state_dict()
                    checkpoint[optimizerAlgorithm] = optimizer.state_dict()
                    checkpoint["loss"] = loss
                    torch.save(
                        {
                            "structure": model_structure,
                            "epoch": total_epochs + epoch,
                            "model_state_dict": v.state_dict(),
                            optimizerAlgorithm: optimizer.state_dict(),
                            "loss": loss,
                        },
                        os.path.join(ckpt_dir, model_filename),
                    )
                    saved = True
                min_loss = loss
            if validation_loss < min_validation_loss and epoch > warmup_epoch:
                if not saved:
                    print(
                        "Saving model (epoch = {:4d}, validation loss = {:4f})".format(
                            epoch, validation_loss
                        )
                    )
                    checkpoint["epoch"] = total_epochs + epoch
                    checkpoint["model_state_dict"] = v.state_dict()
                    checkpoint[optimizerAlgorithm] = optimizer.state_dict()
                    checkpoint["loss"] = loss
                    torch.save(
                        {
                            "structure": model_structure,
                            "epoch": total_epochs + epoch,
                            "model_state_dict": v.state_dict(),
                            optimizerAlgorithm: optimizer.state_dict(),
                            "loss": loss,
                        },
                        os.path.join(ckpt_dir, model_filename),
                    )
                    test(
                        test_dataset,
                        v,
                        criterion,
                        os.path.join(
                            log_dir,
                            "[Epoch %d] [Batch %d] [train loss %f] [validation loss %f]"
                            % (
                                epoch,
                                i + 1,
                                loss,
                                validation_loss,
                            ),
                        ),
                        epoch=epoch,
                        generateImage=True,
                        device=device,
                    )
                    saved = True
                min_validation_loss = validation_loss
            if batches_done % len(dataloader) == 0 or batches_done == 1:
                # sample_image(n_row=8, batches_done=batches_done)
                v.to(device)
                plot_learning_curve(
                    loss_record,
                    os.path.join(log_dir, "learning_curve.png"),
                    ylimit=(0, 1),
                    loss=criterionAlgorithm,
                    title=learning_curve_title,
                )

    with open(os.path.join(log_dir,"time_used.txt"),"w") as f:
        f.writelines(f"time used: {datetime.now() - start}")
    print("ellapsed time:", datetime.now() - start)
