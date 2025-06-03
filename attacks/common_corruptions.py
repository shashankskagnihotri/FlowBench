from typing import Dict, List
import torch
import numpy as np
from ptlflow.models.base_model.base_model import BaseModel
from attacks.attack_utils.utils import (
    get_image_tensors,
    replace_images_dic,
)
from imagecorruptions import corrupt


def common_corrupt(
    attack_args: Dict[str, List[object]],
    inputs: Dict[str, torch.Tensor],
    model: BaseModel,
):
    # Define what device we are using
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Convert images to numpy array
    image_1, image_2 = get_image_tensors(inputs)

    # Rearrange the dimensions of the tensor, remove the batch dimension
    image_1_permuted = image_1.squeeze(0).permute(
        1, 2, 0
    )  # To RGB format (height x width x channels)
    image_2_permuted = image_2.squeeze(0).permute(1, 2, 0)

    # Create the image as a numpy array scaled to [0, 255]
    image_1_numpy = (image_1_permuted * 255).clamp(0, 255).byte().cpu().numpy()
    image_2_numpy = (image_2_permuted * 255).clamp(0, 255).byte().cpu().numpy()

    # Create corruption on each input image
    image_1_corrupt = corrupt(
        image_1_numpy,
        corruption_name=attack_args["cc_name"],
        severity=attack_args["cc_severity"],
    )
    image_2_corrupt = corrupt(
        image_2_numpy,
        corruption_name=attack_args["cc_name"],
        severity=attack_args["cc_severity"],
    )

    # Rescale the numpy images to [0, 1] and convert them back to tensors
    image_1 = torch.from_numpy(image_1_corrupt / 255).to(device).float()
    image_2 = torch.from_numpy(image_2_corrupt / 255).to(device).float()

    # Rearrange the dimensions of the tensor back to the original shape
    image_1 = image_1.permute(2, 0, 1).unsqueeze(
        0
    )  # To RGB format (batch_size x channels x height x width)
    image_2 = image_2.permute(2, 0, 1).unsqueeze(0)

    perturbed_inputs = replace_images_dic(inputs, image_1, image_2, clone=True)

    # Make prediction
    preds = model(perturbed_inputs)

    return preds, perturbed_inputs
