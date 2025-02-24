import sys
import configparser
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from labMLlib.utils.ascPreprocess import (
    search_files_by_extension,
    get_asc_header,
    get_asc_mask,
    dict2asc_header,
)
from labMLlib.utils.imgPreprocess import (
    imgPreprocessAnnchung_npz_with_filepath,
    plot_learning_curve,
)
from PIL import Image
from pathlib import Path
import torch
import numpy as np
import os
import time
import csv


parser = argparse.ArgumentParser()
parser.add_argument(
    "--cfg",
    type=str,
    default="./settings/predictAnnchung(VITNystrom+huber+adam).ini",
    help="config file (.ini) path",
)
opt = parser.parse_args()
asc_list = search_files_by_extension(
    ".asc", chkpath="originalDatasets/Annchung0808_split"
)
config_file_path = opt.cfg
config = configparser.ConfigParser()
config.read(config_file_path)

model_weight_path = config["path"]["model_path"]
model_structure_path = config["path"]["model_structure_path"]
log_dir = config["path"]["log_dir_root"]
lossRecordPath = Path(config["path"]["lossRecordPath"])

dataroot = config["settings"]["dataroot"]  # train, test, validation 存放的地方
geoData_fn = config["settings"]["geoData_fn"]
csv_fn = config["settings"]["csv_fn"]
device = config["settings"]["device"]

xllcorner = config["header"].getfloat("xllcorner")
yllcorner = config["header"].getfloat("yllcorner")
nrows = config["header"].getint("nrows")
ncols = config["header"].getint("ncols")
example_index = config["example_index"].getint("example_index")

loss = config["plot"]["loss"]
title = config["plot"]["title"]
trimmed_at = config["plot"].getint("trimmed_at")
y_min = config["plot"].getfloat("y_min")
y_max = config["plot"].getfloat("y_max")
xlabel = config["plot"]["xlabel"]

os.makedirs(log_dir, exist_ok=True)
model = torch.load(model_structure_path, map_location=torch.device(device))
checkpoint = torch.load(model_weight_path, map_location=torch.device(device))
print("epoch:", checkpoint["epoch"])
# print("Adam:",checkpoint["Adam"])
print("loss:", checkpoint["loss"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.to(device)
test_set, test_set_path = imgPreprocessAnnchung_npz_with_filepath(
    root_folder=dataroot, geoData_fn=geoData_fn, csv_fn=csv_fn, mode="test"
)
data_sample, answer_sample = test_set[example_index]
data_sample_path = test_set_path[example_index]
prediction_filename = "prediction-" + "-".join(Path(data_sample_path).parts[-2:])
answer_sample_filename = "answer-" + "-".join(Path(data_sample_path).parts[-2:])
print(prediction_filename)
data_sample, answer_sample = data_sample.to(device), answer_sample.to(device)
print("answer_sample.shape", answer_sample.shape)
start = time.time()
prediction = (
    model(data_sample.unsqueeze(0)).detach().view(*answer_sample.shape).squeeze()
)
# print(answer_sample.shape)
# set value smaller than 0 to 0 since rain height should never be lower than 0
for i in range(nrows):
    for j in range(ncols):
        if prediction[i, j] < 0:
            prediction[i, j] = 0

with torch.no_grad():
    with open(os.path.join(log_dir, "record.txt"), "w") as f:
        # prediction = layer(prediction)
        print("answer min:", torch.min(answer_sample))
        print("predict min:", torch.min(prediction))
        print("answer max:", torch.argmax(answer_sample))
        print("predict max:", torch.argmax(prediction))
        print("answer max:", torch.max(answer_sample))
        print("predict max:", torch.max(prediction))
        print(
            "difference between max:",
            (torch.max(answer_sample) - torch.max(prediction)),
        )
        print("answer mean:", torch.mean(answer_sample))
        print("predict mean:", torch.mean(prediction))
        print("answer standard diviation:", torch.std(answer_sample))
        print("predict standard diviation:", torch.std(prediction))
        f.writelines("answer min: %f\n" % torch.min(answer_sample).item())
        f.writelines("predict min: %f\n" % torch.min(prediction).item())
        f.writelines("answer max location: %d\n" % torch.argmax(answer_sample).item())
        f.writelines("predict max location: %d\n" % torch.argmax(prediction).item())
        f.writelines("answer max: %f\n" % torch.max(answer_sample).item())
        f.writelines("predict max: %f\n" % torch.max(prediction).item())
        f.writelines(
            "difference between max(answer_sample-prediction): %f\n"
            % (torch.max(answer_sample).item() - torch.max(prediction).item())
        )
        f.writelines("answer mean: %f\n" % torch.mean(answer_sample).item())
        f.writelines("predict mean: %f\n" % torch.mean(prediction).item())
        f.writelines(
            "answer standard diviation: %f\n" % torch.std(answer_sample).item()
        )
        f.writelines("predict standard diviation: %f\n" % torch.std(prediction).item())
        f.writelines(f"time used: {time.time()-start}s")

prediction = prediction.numpy()
header_list = {}
for p in asc_list:
    if "hy1correct.asc" not in p:
        header_list[p] = get_asc_header(p, verbose=False)

for key, val in header_list.items():
    header_sample = val
    header_sample_path = key
    break
break_flag = False
for key, val in header_list.items():
    if val != header_sample:
        print("header not equal")
        print(key)
        print(val)
        print(header_sample_path)
        print(header_sample)
        break_flag = True
        break

if not break_flag:
    print("header equal")
    print("generating mask...")
    mask = get_asc_mask(header_sample_path, verbose=False)
    mask_nrows, mask_ncols = mask.shape
    # if nrows > mask_nrows or ncols > mask_ncols:
    #     padding_ltrb = [
    #         (nrows - mask_nrows) // 2 if nrows > mask_nrows else 0,
    #         (ncols - mask_ncols) // 2 if ncols > mask_ncols else 0,
    #         (nrows - mask_nrows + 1) // 2 if nrows > mask_nrows else 0,
    #         (ncols - mask_ncols + 1) // 2 if ncols > mask_ncols else 0,
    #     ]
    #     img = pad(img, padding_ltrb, fill=0)  # PIL uses fill value 0
    #     _, mask_ncols, mask_nrows = get_dimensions(img)
    #     if nrows == mask_nrows and ncols == mask_ncols:
    #         return img

    # crop_top = int(round((mask_ncols - ncols) / 2.0))
    # crop_left = int(round((mask_nrows - nrows) / 2.0))
    # return crop(img, crop_top, crop_left, ncols, nrows)
    assert nrows <= mask_nrows and ncols <= mask_ncols
    row_pad = int(round((mask_nrows - nrows) / 2.0))
    col_pad = int(round((mask_ncols - ncols) / 2.0))
    mask = mask[row_pad : nrows + row_pad, col_pad : ncols + col_pad]
    print(mask.shape)
    # torchvision.utils.save_image(torch.from_numpy(mask),"mask_torchvision.png")
    mask_img = Image.fromarray(mask, "L")
    mask_img.save("mask.png")
    with open(header_sample_path, "r") as f:
        lines = f.readlines()
        header, array = lines[:6], lines[6:]
        NODATA_value = header[-1].split()[1]
        header_dict = {}
        for i in header:
            header_dict[i.split()[0]] = i.split()[1]
        imgData = []
        for i in array:
            imgData.append([1 if j != NODATA_value else 0 for j in i.split()])
        imgData = np.array(imgData, dtype=np.float16)
        # print("     ".join(str(i) for i in list(imgData[1, :])))
        # print("prediction.shape: ",prediction.shape)
        # print("imgData.shape: ",imgData.shape)
        # print(header_dict)
        header_dict["xllcorner"] = 167200.5
        header_dict["yllcorner"] = 2549282.5
        header_dict["ncols"] = 256
        header_dict["nrows"] = 256
        nrows = int(header_dict["nrows"])
        ncols = int(header_dict["ncols"])
        # if nrows >= prediction.shape[0]:
        #     prediction = np.pad(
        #         prediction,
        #         ((0, nrows - prediction.shape[0]), (0, 0)),
        #         "constant",
        #         constant_values=0,
        #     )
        # if ncols >= prediction.shape[1]:
        #     prediction = np.pad(
        #         prediction,
        #         ((0, 0), (0, ncols - prediction.shape[1])),
        #         "constant",
        #         constant_values=0,
        #     )
        # print(prediction.shape)
        # xllcorner->left
        # yllcorner->bottom
        header = dict2asc_header(header_dict)
        with open(os.path.join(log_dir, prediction_filename), "w") as asc_file:
            asc_file.writelines(header)
            # print(header)
            prediction = prediction * mask
            for i in range(nrows):
                for j in range(ncols):
                    if prediction[i, j] == np.nan:
                        prediction[i, j] = NODATA_value
            for j in range(nrows):
                asc_file.writelines(
                    "     ".join(str(i) for i in list(prediction[j, :])) + "\n"
                )

        with open(os.path.join(log_dir, answer_sample_filename), "w") as asc_file:
            asc_file.writelines(header)
            # print(answer_sample.shape)
            answer_sample.squeeze_()
            for j in range(nrows):
                asc_file.writelines(
                    "     ".join(str(float(i)) for i in list(answer_sample[j, :]))
                    + "\n"
                )

with open(lossRecordPath, "r") as f:
    reader = list(csv.DictReader(f))
    record = {"train": [], "test": [], "validation": []}
    for i in reader:
        # record["epoch"] = i["epoch"]
        record["train"].append(float(i["train loss"]))
        record["validation"].append(float(i["validation loss"]))
        record["test"].append(float(i["test loss"]))
    record_trimmed = {"train": [], "test": [], "validation": []}
    record_trimmed["test"] = record["test"][:trimmed_at]
    record_trimmed["train"] = record["train"][:trimmed_at]
    record_trimmed["validation"] = record["validation"][:trimmed_at]
    plot_learning_curve(
        loss_record=record,
        filename=os.path.join(log_dir, "loss_record.png"),
        ylimit=(y_min, y_max),
        loss=loss,
        title=title,
        xlabel=xlabel,
    )
    plot_learning_curve(
        loss_record=record_trimmed,
        filename=os.path.join(log_dir, f"loss_record_{trimmed_at}.png"),
        ylimit=(y_min, y_max),
        loss=loss,
        title=title,
        xlabel=xlabel,
    )
