import imageio
import os


def genGIF(path: str | list, fn="visualize.gif", folder="./visualize_results"):
    os.makedirs(folder, exist_ok=True)
    if isinstance(path, list):
        for count, p in enumerate(path):
            genGIF(p, fn=f"{count}.gif", folder=folder)
    elif isinstance(path, str):
        images = []
        filenames = os.listdir(path)
        # print(filenames)
        for filename in filenames:
            filename = os.path.join(p, filename)
            images.append(imageio.imread(filename))
        imageio.mimsave(fn, images)
