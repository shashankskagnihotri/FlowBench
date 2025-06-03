from typing import Dict, List, Optional
import torch
from ptlflow.models.base_model.base_model import BaseModel
from attacks.attack_utils.utils import (
    get_image_tensors,
    get_image_grads,
    replace_images_dic,
)
from attacks.attack_utils.loss_criterion import LossCriterion
from cospgd import functions as attack_functions

batch_size = 1


@torch.enable_grad()
def fgsm(
    attack_args: Dict[str, List[object]],
    inputs: Dict[str, torch.Tensor],
    model: BaseModel,
    targeted_inputs: Optional[Dict[str, torch.Tensor]],
):
    criterion = LossCriterion(attack_args["attack_loss"])
    # import pdb

    # pdb.set_trace()
    orig_image_1 = inputs["images"][0][0].clone().unsqueeze(0)
    orig_image_2 = inputs["images"][0][1].clone().unsqueeze(0)

    if attack_args["attack_targeted"]:
        labels = targeted_inputs["flows"].clone().squeeze(0)
    else:
        labels = inputs["flows"].clone().squeeze(0)

    inputs["images"].requires_grad_(True)

    preds = model(inputs)
    preds_raw = preds["flows"].squeeze(0)

    loss = criterion.loss(preds_raw.float(), labels.float())
    loss = loss.mean()
    loss.backward()
    # pdb.set_trace()

    image_1, image_2 = get_image_tensors(inputs, clone=True)
    image_1_grad, image_2_grad = get_image_grads(inputs)
    # pdb.set_trace()

    image_1_adv = fgsm_attack(attack_args, image_1, image_1_grad, orig_image_1)
    image_2_adv = fgsm_attack(attack_args, image_2, image_2_grad, orig_image_2)
    # pdb.set_trace()
    with torch.no_grad():
        perturbed_inputs = replace_images_dic(inputs, image_1_adv, image_2_adv, clone=True)
        preds = model(perturbed_inputs)

    return preds, perturbed_inputs


def fgsm_attack(
    attack_args: Dict[str, List[object]], perturbed_image, data_grad, orig_image
):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input imag
    if attack_args["attack_targeted"]:
        sign_data_grad *= -1
    # import pdb

    # pdb.set_trace()
    if attack_args["attack_norm"] == "two":
        sign_data_grad = attack_functions.lp_normalize(
            sign_data_grad, p=2, epsilon=1.0, decrease_only=False
        )
    perturbed_image = (
        perturbed_image.detach() + attack_args["attack_epsilon"] * sign_data_grad
    )
    # Adding clipping to maintain [0,1] range
    if attack_args["attack_norm"] == "inf":
        delta = torch.clamp(
            perturbed_image - orig_image,
            min=-1 * attack_args["attack_epsilon"],
            max=attack_args["attack_epsilon"],
        )
    elif attack_args["attack_norm"] == "two":
        # delta = perturbed_image - orig_image
        # delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
        # factor = attack_args["attack_epsilon"] / delta_norms
        # factor = torch.min(factor, torch.ones_like(delta_norms))
        # delta = delta * factor.view(-1, 1, 1, 1)
        # pdb.set_trace()
        delta = attack_functions.lp_normalize(
            noise=perturbed_image - orig_image,
            p=2,
            epsilon=attack_args["attack_epsilon"],
            decrease_only=True,
        )
        # Adding clipping to maintain [0,1] range
        # perturbed_image = torch.clamp(orig_image + delta, clamp_min, clamp_max).detach()
    # pdb.set_trace()
    perturbed_image = torch.clamp(orig_image + delta, 0, 1)
    # Return the perturbed image
    return perturbed_image
