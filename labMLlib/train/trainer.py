from typing import Any
import torch
import torchvision
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchmetrics.regression import R2Score
from PIL import Image
import gc

import os
import csv
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from labMLlib.utils.functions import plot_learning_curve


class Train:
    def __init__(
        self,
        dataloader:DataLoader,
        clip_value=None,
        gradient_accumulations=None,
        scaler:GradScaler=None,
        scheduler:torch.optim.lr_scheduler._LRScheduler=None,
        device="cuda",
        # tqdm params
        use_tqdm=True,
        description_format="train loss:%.6f,r-squared:%.6f",
        position=1,
    ) -> None:
        self.dataloader = dataloader
        self.clip_value = clip_value
        self.gradient_accumulations = gradient_accumulations
        self.scaler = scaler
        self.scheduler = scheduler
        self.device = device
        self.use_tqdm = use_tqdm
        self.description_format = description_format
        self.position = position
        self.r2score = R2Score()
        self.set_pbar()

    def __call__(
        self,
        i,
        data,
        model,
        optimizer,
        criterion,
    ) -> Any:
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        if len(data)==3:
            key, img, answer = data
        elif len(data)==2:
            img,answer = data
        else:
            raise NotImplementedError("data length mismatch!")
        img, answer = img.to(self.device), answer.to(self.device)
        preds = model(img)
        preds = preds.view(*answer.shape)
        loss = criterion(preds, answer)
        # print("before backward: ",loss)
        if self.clip_value:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
        if self.gradient_accumulations and self.scaler:
            self.scaler.scale(loss / self.gradient_accumulations).backward()
            if (i + 1) % self.gradient_accumulations == 0:
                self.scaler.step(optimizer)
                self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
        # print("after backward: ",loss)
        loss = loss.detach().cpu().item()
        if self.scheduler:
            self.scheduler.step(loss)
        if self.use_tqdm:
            r2score = self.r2score(preds.cpu().detach().flatten(),answer.cpu().detach().flatten()).item()
            self.pbar.set_description(self.description_format % (loss,r2score))
        return loss,model,optimizer,criterion

    def loop(
        self,
        model,
        optimizer,
        criterion,
    ) -> float:
        total_loss = 0
        self.set_pbar()
        model.train()
        for i, data in self.pbar:
            loss,model,optimizer,criterion = self.__call__(i, data, model, optimizer, criterion)
            total_loss += loss
        return float(total_loss / len(self.dataloader))

    def set_pbar(self):
        self.pbar = (
            tqdm(
                enumerate(self.dataloader),
                total=len(self.dataloader),
                position=self.position,
                leave=False,
            )
            if self.use_tqdm
            else enumerate(self.dataloader)
        )
        if self.use_tqdm:
            self.pbar.set_description(self.__class__.__name__)


class Validation:
    def __init__(
        self,
        dataloader:DataLoader,
        device="cuda",
        # tqdm params
        use_tqdm=True,
        description_format="validation loss:%.6f,r-squared:%.6f",
        position=2,
    ) -> None:
        self.dataloader = dataloader
        self.device = device
        self.use_tqdm = use_tqdm
        self.description_format = description_format
        self.position = position
        self.r2score = R2Score()
        self.set_pbar()

    def __call__(
        self,
        data,
        model,
        criterion,
    ) -> Any:
        model.eval()  # set model to evaluation mode
        if len(data)==3:
            _, x, y = data
        elif len(data)==2:
            x,y = data
        else:
            raise NotImplementedError("data length mismatch!")
        x, y = x.to(self.device), y.to(self.device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            pred = pred.view(y.shape)
            loss = criterion(pred, y).detach().cpu().item()
        if self.use_tqdm:
            r2score = self.r2score(pred.cpu().detach().flatten(),y.cpu().detach().flatten()).item()
            self.pbar.set_description(self.description_format % (loss,r2score))
        return loss  # accumulate loss

    def loop(
        self,
        model,
        criterion,
    ) -> float:
        model.eval()  # set model to evaluation mode
        total_loss = 0
        self.set_pbar()
        for data in self.pbar:  # iterate through the dataloader
            loss = self.__call__(data, model, criterion)
            total_loss += loss  # accumulate loss
        model.train()
        return float(total_loss / len(self.pbar))

    def set_pbar(self):
        self.pbar = (
            tqdm(
                self.dataloader,
                total=len(self.dataloader),
                position=self.position,
                leave=False,
            )
            if self.use_tqdm
            else self.dataloader
        )
        if self.use_tqdm:
            self.pbar.set_description(self.__class__.__name__)


class Test:
    def __init__(
        self,
        dataloader:DataLoader,
        logDir,
        device="cuda",
        # tqdm params
        use_tqdm=True,
        description_format="test loss:%.6f,r-squared:%.6f",
        position=3,
    ) -> None:
        self.dataloader = dataloader
        self.logDir = logDir
        self.device = device
        self.use_tqdm = use_tqdm
        self.description_format = description_format
        self.position = position
        self.r2score = R2Score()
        self.set_pbar()

    def __call__(
        self,
        epoch,
        data,
        model,
        criterion,
        saveImage=False,
        saveImageFolder=None,
    ) -> Any:
        model.eval()  # set model to evaluation mode
        if len(data)==3:
            key, x, y = data
        elif len(data)==2:
            x,y = data
            key = None
        else:
            raise NotImplementedError("data length mismatch!")
        x, y = x.to(self.device), y.to(self.device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            pred = pred.view(y.shape)
            if len(pred.shape)==4:
                bs, channel, w, h = pred.shape
            elif len(pred.shape)==3:
                channel, w, h = pred.shape
                bs = 1
                pred.unsqueeze_(0)
                y.unsqueeze_(0)
            elif len(pred.shape)==5:
                bs, channel, w, h, d = pred.shape
                pred = pred.squeeze(0)
                y = y.squeeze(0)
            else:raise NotImplementedError("channel of pred and answer must be 3 or 4!")
            loss = criterion(pred, y).detach().cpu().item()
            if saveImage:
                pred_img, y_img = pred.clone().detach().to("cpu"), y.clone().detach().to("cpu")
                for batch,i,j in zip(range(1,bs+1),pred_img,y_img):
                    if not saveImageFolder:os.makedirs(os.path.join(self.logDir,f"epoch_{epoch}"),exist_ok=True)
                    torchvision.utils.save_image(
                        [i, j],
                        os.path.join(saveImageFolder if saveImageFolder else os.path.join(self.logDir,f"epoch_{epoch}"), f"{key[batch-1]}_{loss}_batch{batch}.jpg" if key else f"None_{loss}_batch{batch}.jpg"),
                        nrow=2,
                    )
        if self.use_tqdm:
            r2score = self.r2score(pred.cpu().detach().flatten(),y.cpu().detach().flatten()).item()
            self.pbar.set_description(self.description_format % (loss,r2score))
        return loss  # accumulate loss

    def loop(
        self,
        model,
        criterion,
        saveImage=False,
        saveImageFolder=None,
    ) -> float:
        model.eval()  # set model to evaluation mode
        total_loss = 0
        self.set_pbar()
        for epoch, data in self.pbar:  # iterate through the dataloader
            loss = self.__call__(epoch, data, model, criterion, saveImage,saveImageFolder)
            total_loss += loss  # accumulate loss
        model.train()
        return float(total_loss / len(self.pbar))

    def set_pbar(self):
        self.pbar = (
            tqdm(
                enumerate(self.dataloader),
                total=len(self.dataloader),
                position=self.position,
                leave=False,
            )
            if self.use_tqdm
            else self.dataloader
        )
        if self.use_tqdm:
            self.pbar.set_description(self.__class__.__name__)

    def save_image(self,imglist,fp,nrow=2):
        assert len(imglist)==2,"imglist should be just pred and ans!"
        pred, ans = imglist
        pred, ans = self.minMaxScale(pred)*255,self.minMaxScale(ans)*255
        pred,ans = pred.permute(1,2,0),ans.permute(1,2,0)
        pred,ans = pred.numpy().astype("uint8"), ans.numpy().astype("uint8")
        img = Image.fromarray(pred.squeeze(),"L")
        img.save(fp)

    def minMaxScale(self,tensor):
        with torch.no_grad():
            return (tensor-torch.min(tensor))/(torch.max(tensor)-torch.min(tensor))

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        trainMethod: Train,
        validationMethod: Validation,
        testMethod: Test,
        logDir=None,
        ckptDir=None,
        n_epoch=1000,
        warmup_epoch=0,
        model_fn=None,
        loss_record_fn="lossRecord.csv",
        learning_curve_title=None,
        device="auto",
        saveStructure=True,
        use_tqdm=True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainMethod = trainMethod
        self.validationMethod = validationMethod
        self.testMethod = testMethod
        self.logDir = (
            logDir
            if logDir
            else Path(
                os.path.join(
                    self.__class__.__name__,
                    f"{self.model.__class__.__name__}_{datetime.now()}",
                )
            )
        )
        self.ckptDir = ckptDir if ckptDir else Path(os.path.join(self.logDir, "models"))
        os.makedirs(self.logDir, exist_ok=True)
        os.makedirs(self.ckptDir, exist_ok=True)
        self.n_epoch = n_epoch
        self.warmup_epoch = warmup_epoch
        self.model_fn = model_fn if model_fn else self.model.__class__.__name__ + ".pth"
        self.loss_record_fn = loss_record_fn
        self.learning_curve_title = (
            learning_curve_title
            if learning_curve_title
            else self.model.__class__.__name__
        )
        self.device = device
        self.loss_record = {"train": [], "validation": [], "test": []}
        self.min_train_loss = 1000.0
        self.min_test_loss = 1000.0
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if saveStructure:
            torch.save(self.model, os.path.join(self.ckptDir, "structure.pt"))
        self.use_tqdm = use_tqdm
        self.set_pbar()

    def train(self):
        start = datetime.now()
        try:
            torch.multiprocessing.freeze_support()
            with open(os.path.join(self.logDir, self.loss_record_fn), "w") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(["epoch", "train loss", "validation loss", "test loss"])
            for epoch in self.pbar:
                self.pbar.set_description(f"epoch {epoch}")
                self.model.train()
                train_loss = self.trainMethod.loop(
                    self.model, self.optimizer, self.criterion
                )
                validation_loss = self.validationMethod.loop(self.model, self.criterion)
                test_loss = self.testMethod.loop(
                    self.model, self.criterion, saveImage=False
                )
                self.loss_record["train"].append(train_loss)
                self.loss_record["validation"].append(validation_loss)
                self.loss_record["test"].append(test_loss)
                self.pbar.set_description(
                    "[Epoch %d/%d] [loss: [train: %f] [val: %f] [test: %f]]"
                    % (
                        epoch,
                        self.n_epoch,
                        train_loss,
                        validation_loss,
                        test_loss,
                    )
                )
                with open(os.path.join(self.logDir, self.loss_record_fn), "a") as f:
                    w = csv.writer(f, lineterminator="\n")
                    w.writerow(
                        [
                            f"{epoch}",
                            f"{train_loss}",
                            f"{validation_loss}",
                            f"{test_loss}",
                        ]
                    )
                saved = False
                if train_loss < self.min_train_loss and epoch > self.warmup_epoch:
                    self.min_train_loss = train_loss
                    if not saved:
                        self.save(epoch, self.min_train_loss)
                        saved = True
                if test_loss < self.min_test_loss and epoch > self.warmup_epoch:
                    os.makedirs(os.path.join(self.logDir,f"epoch_{epoch}"),exist_ok=True)
                    self.testMethod.loop(self.model, self.criterion, saveImage=True,saveImageFolder=os.path.join(self.logDir,f"epoch_{epoch}"))
                    self.min_test_loss = validation_loss
                    if not saved:
                        self.save(epoch, self.min_test_loss)
                        saved = True
                plot_learning_curve(
                    self.loss_record,
                    os.path.join(self.logDir, "learning_curve.png"),
                    ylimit=(0, 1),
                    loss=self.criterion.__class__.__name__,
                    title=self.learning_curve_title,
                )
                torch.cuda.empty_cache()
                gc.collect()
        finally:
            with open(os.path.join(self.logDir, "time_used.txt"), "w") as f:
                time_used = datetime.now() - start
                f.writelines(f"time used: {time_used}")
            print("time used: ", time_used)

    def set_pbar(self):
        self.pbar = (
            tqdm(range(1, self.n_epoch + 1), position=0)
            if self.use_tqdm
            else range(1, self.n_epoch + 1)
        )
        if self.use_tqdm:
            self.pbar.set_description(self.__class__.__name__)

    def save(self, epoch, loss):
        if self.use_tqdm:
            self.pbar.set_description(
                "Saving model (epoch = {:4d}, loss = {:4f})".format(epoch, loss)
            )
        self.model.to("cpu")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                self.optimizer.__class__.__name__: self.optimizer.state_dict(),
                "loss": loss,
            },
            os.path.join(self.ckptDir, self.model_fn),
        )
        self.model.to(self.device)
