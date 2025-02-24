import numpy as np
from PIL import Image
from pathlib import Path
import os
import re
import asyncio
import traceback
from tqdm import tqdm


def np2dRangeScalar(np_obj, NewMin, NewMax):
    OldMax = np.max(np_obj)
    OldMin = np.min(np_obj)
    OldRange = OldMax - OldMin
    shape = np_obj.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if OldRange == 0:
                NewValue = NewMin
            else:
                NewRange = NewMax - NewMin
                NewValue = (((np_obj[i][j] - OldMin) * NewRange) / OldRange) + NewMin
            np_obj[i][j] = NewValue
    return np_obj


def asc2png(asc_path: str | list, target_folder=".", verbose=None,dryrun=False):
    if verbose is None:
        verbose = False
    if type(asc_path) != list:
        asc_path = [asc_path]
    pbar = tqdm(asc_path,desc="Progress")
    for p in pbar:
        with open(p, "r") as f:
            pbar.set_description(f"processing {os.path.join(*p.split(os.sep)[-4:])}")
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
            imgData = np.array(imgData, dtype=np.float16)
            imgData = np2dRangeScalar(imgData, 0, 256)
            # imgData = np.uint8(imgData)
            if verbose:
                print("shape:", imgData.shape)
                print("dtype:", imgData.dtype)
                print("min:", imgData.min(), "max", imgData.max())
            # imgData = imgData.astype(np.uint8)
            # print(imgData.min(), imgData.max(), imgData.mean(), imgData[0][0])
            img = Image.fromarray(np.uint8(imgData), "L")  # ,'L'
            p_filename = os.path.splitext(os.path.basename(p))[0] + ".png"
            relativepath = [*Path(os.path.splitext(p)[0]).parts][
                len([*Path(os.path.dirname(__file__)).parts]) : -1
            ]
            target_folder_path = os.path.join(
                target_folder, *relativepath
            )
            if verbose:
                print("target_folder_path", target_folder_path)
            if not dryrun:
                if not os.path.exists(target_folder_path):
                    os.makedirs(target_folder_path)
                with open(os.path.join(target_folder_path, p_filename), "wb") as fp:
                    img.save(fp)
                    fp.close()
            f.close()


def search_files_by_extension(extension, chkpath=".", verbose=None):
    default_path = os.getcwd()
    os.chdir(chkpath)
    if verbose is None:
        verbose = False
    png_list = []
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(extension)]:
            p = os.path.join(dirpath, filename)
            p = [*Path(dirpath).parts]
            while "." in p:
                p.remove(".")
            p = os.path.join(*p, filename)
            if verbose:
                print("p:", p)
            if chkpath != ".":
                p = os.path.join(default_path, chkpath, p)
            else:
                p = os.path.join(default_path, p)
            png_list.append(p)
    if verbose:
        print(png_list)
    os.chdir(default_path)
    return png_list


def ascGenMask(
    asc_list,
    target_folder,
    fn="mask.npy",
    pattern=".*?0.asc",
    verbose=None,
    dtype=np.float16,
):
    if verbose is None:
        verbose = False
    if type(asc_list) != list:
        asc_list = [asc_list]
    pbar = tqdm(asc_list,desc="Progress")

    for p in pbar:
        arr = get_asc_mask(p)
        arr[np.isnan(arr)] = 0
        arr = 1 - arr
        try:
            mask*=arr
        except NameError:
            mask=arr
    mask = 1 - mask

    np.save(os.path.join(target_folder,fn),mask)

def asc2npz(
    asc_list,
    target_folder,
    fn="dataset.npz",
    remove_pattern=".*?0.asc",
    verbose=None,
    dtype=np.float32,
    num_workers=16,
):
    if verbose is None:
        verbose = False
    if type(asc_list) != list:
        asc_list = [asc_list]
    async def worker(p,pbar:tqdm,verbose):
        try:
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
                pbar.set_description(os.path.join(*p.split(os.sep)[-4:]))
                pbar.update(1)
                # compressed_data[p] = imgData
                f.close()
            return p,imgData
        except:
            print(traceback.format_exc())

    async def workerPool():
        queue = asyncio.Queue(maxsize=num_workers)
        tasks = []
        compressed_data = {}
        pbar = tqdm(range(len(asc_list)),desc="Progress")
        for p in asc_list:
            if re.findall(remove_pattern, p) == []:
                # print(p)
                await queue.put(worker(p,pbar, verbose))
                while queue.empty() is False:
                    tasks.append(await queue.get())
        result = await asyncio.gather(*tasks, return_exceptions=True)
        for p,imgData in result:
            compressed_data[p] = imgData
        return compressed_data
    compressed_data = asyncio.run(workerPool())
    np.savez_compressed(os.path.join(target_folder, fn), **compressed_data)

def get_asc_header(asc_file_path: str, verbose=None)->dict:
    header_dict = {}
    with open(asc_file_path, "r") as f:
        lines = f.readlines()
        header, array = lines[:6],lines[6:]
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

def dict2asc_header(header_dict: dict, verbose=None)->list:
    key_list = ["ncols","nrows","xllcorner","yllcorner","cellsize","NODATA_value"]
    assert all(x in list(header_dict.keys()) for x in key_list), f"the header dict should contain {key_list}"
    header = []
    for key, val in header_dict.items():
        header.append("\t\t".join([key,str(val)])+"\n")
    return header


if __name__ == "__main__":
    # 查找並返還當前目錄下的所有.asc文件
    asc_list_train = search_files_by_extension(
        ".asc", chkpath="Annchung0808_split/train"
    )
    asc_list_test = search_files_by_extension(".asc", chkpath="Annchung0808_split/test")
    asc_list_val = search_files_by_extension(
        ".asc", chkpath="Annchung0808_split/validation"
    )
    # print(asc_list)
    # 把asc文件轉換成png並儲存在指定文件夾archive内，png文件位置會參考.asc文件的相對位置
    # 列如下面的代碼會把此文件目錄下的1202/5mm/dm1d0000.asc生成的png放在此文件目錄下的archive/1202/5mm/dm1d0000.png
    # 需要注意的是相對路徑根據此文件所在位置來定義
    # asc2png(asc_list,target_folder='archived_Annchung',verbose=False)

    # 把asc文件轉換成npz並儲存在指定文件夾archive内，npz文件位置會參考.asc文件的相對位置
    # 列如下面的代碼會把此文件目錄下的1202/5mm/dm1d0000.asc生成的npz放在此文件目錄下的archive/1202/5mm/dm1d0000.npz
    # 需要注意的是相對路徑根據此文件所在位置來定義
    asc2npz(
        asc_list_train,
        target_folder="archived_Annchung_npz",
        fn="train.npz",
        verbose=False,
    )
    asc2npz(
        asc_list_test,
        target_folder="archived_Annchung_npz",
        fn="test.npz",
        verbose=False,
    )
    asc2npz(
        asc_list_val,
        target_folder="archived_Annchung_npz",
        fn="validation.npz",
        verbose=False,
    )
