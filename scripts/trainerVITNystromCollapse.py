# python global packages
import sys
import configparser
import json
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from labMLlib.datasets.collapseVITDataset import collapseVITDataset
from labMLlib.utils.loss import *
from labMLlib.utils.functions import dim_expand
from labMLlib.Optim.optimizerDict import optimDict
from labMLlib.Loss.lossDict import lossDict

from labMLlib.train.trainer import *

import torch
from vit_pytorch.efficient import ViT
from nystrom_attention import Nystromformer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
device = config["settings"]["device"]
log_dir_root = Path(config["settings"]["log_dir_root"])
ckpt_dir = config["settings"]["ckpt_dir"]
model_filename = config["settings"]["model_filename"]
latest_model_filename = config["settings"]["latest_model_filename"]
load_model_path = Path(config["settings"]["load_model_path"])
learning_curve_title = config["settings"]["learning_curve_title"]
is_basic = True if "is_basic" not in config["settings"].keys() else config["settings"]["is_basic"]

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
vit_dim = 128 if "vit_dim" not in config["hyperparameters"].keys() else config["hyperparameters"]["vit_dim"]
nystromformer_depth=3 if "nystromformer_depth" not in config["hyperparameters"].keys() else config["hyperparameters"]["nystromformer_depth"]
nystromformer_heads=4 if "nystromformer_heads" not in config["hyperparameters"].keys() else config["hyperparameters"]["nystromformer_heads"]
nystromformer_num_landmarks=256 if "nystromformer_num_landmarks" not in config["hyperparameters"].keys() else config["hyperparameters"]["nystromformer_num_landmarks"]
n_epoch = config["hyperparameters"]["n_epoch"]
warmup_epoch = config["hyperparameters"]["warmup_epoch"]
sample_interval = config["hyperparameters"]["sample_interval"]
use_gradient_accumulations = config["hyperparameters"][
    "use_gradient_accumulations"
]
try:
    use_lr_scheduler = config["hyperparameters"]["use_lr_scheduler"]
except KeyError:
    print("warning: use_lr_scheduler not found in config, set to False")
    use_lr_scheduler = False
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
train_dataset = collapseVITDataset(
    mode="train",
    dataDir=dataroot,
    image_size=image_size,
    padding=padding,
    is_basic=is_basic,
)
print("Sample shape: ", train_dataset[0][1].shape)
test_dataset = collapseVITDataset(
    mode="test",
    dataDir=dataroot,
    image_size=image_size,
    padding=padding,
    is_basic=is_basic,
)
print("Sample shape: ", test_dataset[0][1].shape)
validation_dataset = collapseVITDataset(
    mode="validation",
    dataDir=dataroot,
    image_size=image_size,
    padding=padding,
    is_basic=is_basic,
)
print("Sample shape: ", validation_dataset[0][1].shape)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
)
validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
)
len_train_dataloader = len(train_dataloader)
_, train_data_sample, _ = train_dataset[0]
channels = train_data_sample.shape[0]
if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"

padded_image_size = (
    image_size[0] + padding[2] + padding[3],
    image_size[1] + padding[0] + padding[1],
)
print("padded_image_size: ", padded_image_size)
efficient_transformer = Nystromformer(dim=vit_dim, depth=nystromformer_depth, heads=nystromformer_heads, num_landmarks=nystromformer_num_landmarks)

model = ViT(
    image_size=padded_image_size,
    patch_size=patch_size,
    num_classes=dim_expand(image_size),
    transformer=efficient_transformer,
    dim=vit_dim,
    channels=channels,
).to(device)

model_structure = torch.save(model, os.path.join(ckpt_dir, "structure.pt"))
criterion = lossDict(**config['criterion'][criterionAlgorithm])[criterionAlgorithm]
optimizer = optimDict(model.parameters(),**config['optimizer'][optimizerAlgorithm])[optimizerAlgorithm]
scheduler = ReduceLROnPlateau(optimizer, 'min')if use_lr_scheduler else None
# optimizer = optim.AdamW(model.parameters(),lr=lr)
# criterion = nn.MSELoss()
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
        scheduler=scheduler,
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
