import os

import numpy as np
from PIL import Image
from attacks.help_function.flow_plot import colorplot_light


def quickvis_tensor(t, filename):
    """Saves a tensor with three dimensions as image to a specified file location.

    Args:
        t (tensor):
            3-dimensional tensor, following the dimension order (c,H,W)
        filename (str):
            name for the image to save, including path and file extension
    """
    # check if filename already contains .png extension
    if not filename.endswith(".png"):
        filename += ".png"
    valid = False
    if len(t.size()) == 3:
        img = t.detach().cpu().numpy()
        valid = True

    elif len(t.size()) == 4 and t.size()[0] == 1:
        img = t[0, :, :, :].detach().cpu().numpy()
        valid = True

    else:
        print(
            "Encountered invalid tensor dimensions %s, abort printing." % str(t.size())
        )

    if valid:
        img = np.rollaxis(img, 0, 3)
        data = img.astype(np.uint8)
        data = Image.fromarray(data)
        data.save(filename)


def quickvisualization_tensor(t, filename, min=0.0, max=255.0):
    """Saves a batch (>= 1) of image tensors with three dimensions as images to a specified file location.
    Also rescales the color values according to the specified range of the color scale.

    Args:
        t (tensor):
            batch of 3-dimensional tensor, following the dimension order (b,c,H,W)
        filename (str):
            name for the image to save, including path and file extension. Batches will append a number at the end of the filename.
        min (float, optional):
            minimum value of the color scale used by tensor. Defaults to 0.
        max (float, optional):
            maximum value of the color scale used by tensor Defaults to 255.
    """
    # rescale to [0,255]
    t = (t.detach().clone() - min) / (max - min) * 255.0

    if len(t.size()) == 3 or (len(t.size()) == 4 and t.size()[0] == 1):
        quickvis_tensor(t, filename)

    elif len(t.size()) == 4:
        for i in range(t.size()[0]):
            if i == 0:
                quickvis_tensor(t[i, :, :, :], filename)
            else:
                quickvis_tensor(t[i, :, :, :], filename + "_" + str(i))

    else:
        print(
            "Encountered unprocessable tensor dimensions %s, abort printing."
            % str(t.size())
        )


def quickvis_flow(flow, filename, auto_scale=True, max_scale=-1):
    """Saves a flow field tensor with two dimensions as image to a specified file location.

    Args:
        flow (tensor):
            2-dimensional tensor (c=2), following the dimension order (c,H,W) or (1,c,H,W)
        filename (str):
            name for the image to save, including path and file extension.
        auto_scale (bool, optional):
            automatically scale color values. Defaults to True.
        max_scale (int, optional):
            if auto_scale is false, scale flow by this value. Defaults to -1.
    """
    # check if filename already contains .png extension
    if not filename.endswith(".png"):
        filename += ".png"
    valid = False
    if len(flow.size()) == 3:
        flow_img = flow.clone().detach().cpu().numpy()
        valid = True

    elif len(flow.size()) == 4 and flow.size()[0] == 1:
        flow_img = flow[0, :, :, :].clone().detach().cpu().numpy()
        valid = True

    else:
        print(
            "Encountered invalid tensor dimensions %s, abort printing."
            % str(flow.size())
        )

    if valid:
        # make directory and ignore if it exists
        if not os.path.dirname(filename) == "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        # write flow
        flow_img = np.rollaxis(flow_img, 0, 3)
        data = colorplot_light(
            flow_img, auto_scale=auto_scale, max_scale=max_scale, return_max=False
        )
        data = data.astype(np.uint8)
        data = Image.fromarray(data)
        data.save(filename)


def quickvisualization_flow(flow, filename, auto_scale=True, max_scale=-1):
    """Saves a batch (>= 1) of 2-dimensional flow field tensors as images to a specified file location.

    Args:
        flow (tensor):
            single or batch of 2-dimensional flow tensors, following the dimension order (c,H,W) or (b,c,H,W)
        filename (str):
            name for the image to save, including path and file extension.
        auto_scale (bool, optional):
            automatically scale color values. Defaults to True.
        max_scale (int, optional):
            if auto_scale is false, scale flow by this value. Defaults to -1.
    """
    if len(flow.size()) == 3 or (len(flow.size()) == 4 and flow.size()[0] == 1):
        quickvis_flow(flow, filename, auto_scale=auto_scale, max_scale=max_scale)

    elif len(flow.size()) == 4:
        for i in range(flow.size()[0]):
            if i == 0:
                quickvis_flow(
                    flow[i, :, :, :],
                    filename,
                    auto_scale=auto_scale,
                    max_scale=max_scale,
                )
            else:
                quickvis_flow(
                    flow[i, :, :, :],
                    filename + "_" + str(i),
                    auto_scale=auto_scale,
                    max_scale=max_scale,
                )

    else:
        print(
            "Encountered unprocessable tensor dimensions %s, abort printing."
            % str(flow.size())
        )


def load_weather(path):
    try:
        weather = np.load(path, allow_pickle=True)
        points3D = weather["points3D"]
        motion3D = weather["motion3D"]
        flakes = weather["flakes"]
        flakescol = weather["flakescol"]
        flakestrans = weather["flakestrans"]
        weather = (points3D, motion3D, flakes, flakescol, flakestrans)
        success = True
    except FileNotFoundError as e:
        print(
            f"Failed to load weather data from {path}, File Not Found. Please generate weather instead."
        )
        weather = None
        success = False
    return weather, success
