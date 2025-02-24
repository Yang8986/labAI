import sys
import os
import time
import configparser
import json
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from labMLlib.datasets.collapseVITDataset import collapseVITDataset
from labMLlib.utils.loss import *
from labMLlib.Optim.optimizerDict import optimDict
from labMLlib.Loss.lossDict import lossDict
from labMLlib.utils.functions import dim_expand
from labMLlib.utils.imgPreprocess import imgPreprocessAnnchung_npz

# dataset = collapseVITDataset(mode="train",dataDir="datasets/XDS")
# print(dataset[0])
# key,d, ans = dataset[0]
# print(d.shape)

from labMLlib.train.trainer import *

import torch
from vit_pytorch.efficient import ViT
from nystrom_attention import Nystromformer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cfg",
    type=str,
    default="./settings/VITNystromCollapse/VITNystromCollapse(adam+huber).ini",
    help="config file (.ini) path",
)
opt = parser.parse_args()

config_file_path = opt.cfg
config = json.load(open(config_file_path,"r",encoding="utf-8"))
############## configuration ##################
## settings ##
# [settings]
dataroot = Path(config["settings"]["dataroot"])  # train, test, validation 存放的地方
loss_record_fn = config["settings"]["loss_record_fn"]
geoData_fn = config["settings"]["geodata_fn"]
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
image_size = (image_width, image_height)  # image default size: (718,998)
patch_size = config["hyperparameters"]["patch_size"]
n_epoch = config["hyperparameters"]["n_epoch"]
warmup_epoch = config["hyperparameters"]["warmup_epoch"]
sample_interval = config["hyperparameters"]["sample_interval"]
use_gradient_accumulations = config["hyperparameters"]["use_gradient_accumulations"]
gradient_accumulations = config["hyperparameters"]["gradient_accumulations"]
clip_value = config["hyperparameters"]["clip_value"]

# [padding]
p_up = config["padding"]["p_up"]
p_down = config["padding"]["p_down"]
p_left = config["padding"]["p_left"]
p_right = config["padding"]["p_right"]
##################################################

if __name__ == "__main__":

    start = datetime.now()
    time_const = start.strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join(log_dir_root, time_const)
    log_dir += f"({criterionAlgorithm}+{optimizerAlgorithm})"
    ckpt_dir += (
        "+"
        + criterionAlgorithm
        + "+"
        + optimizerAlgorithm
        + f"({str(start).replace(':','-')})"
    )
    ckpt_dir = Path(ckpt_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    padding = (p_up, p_down, p_left, p_right)
    padding_size = (p_up, p_left)
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
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=16
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True, num_workers=16
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=16
    )
    len_train_dataloader = len(train_dataloader)
    train_data_sample, _ = train_dataset[0]
    channels = train_data_sample.shape[0]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    padded_image_size = (
        image_size[0] + padding_size[0] * 2,
        image_size[1] + padding_size[1] * 2,
    )

    efficient_transformer = Nystromformer(dim=1024, depth=3, heads=4, num_landmarks=256)

    model = ViT(
        image_size=padded_image_size,
        patch_size=patch_size,
        num_classes=dim_expand(image_size),
        transformer=efficient_transformer,
        dim=1024,
        channels=channels,
    ).to(device)
    print(dim_expand(image_size))
    criterion = lossDict(**config['criterion'][criterionAlgorithm])[criterionAlgorithm]
    optimizer = optimDict(model.parameters(),**config['optimizer'][optimizerAlgorithm])[optimizerAlgorithm]
    print(optimizer)
    model_structure = torch.save(model, os.path.join(ckpt_dir, "structure.pt"))
    scaler = GradScaler()if use_gradient_accumulations else None

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        trainMethod=Train(
            dataloader=train_dataloader,
            clip_value=clip_value,
            gradient_accumulations=gradient_accumulations,
            scaler=scaler,
            device=device,
        ),
        validationMethod=Validation(
            dataloader=validation_dataloader,
            device=device,
        ),
        testMethod=Test(dataloader=test_dataloader, logDir=log_dir, device=device),
        logDir=log_dir,
        ckptDir=ckpt_dir,
        n_epoch=n_epoch,
        warmup_epoch=warmup_epoch,
        model_fn=model_filename,
        loss_record_fn=loss_record_fn,
        learning_curve_title=learning_curve_title,
        device=device,
    )
    trainer.train()
