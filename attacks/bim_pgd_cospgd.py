from typing import Dict, List, Optional
import torch
from ptlflow.models.base_model.base_model import BaseModel
from attacks.attack_utils.utils import (
    get_image_tensors,
    get_image_grads,
    replace_images_dic,
)
from cospgd import functions as attack_functions
from attacks.attack_utils.loss_criterion import LossCriterion
import attacks.attack_utils.loss_criterion as losses


@torch.enable_grad()
def bim_pgd_cospgd(
    attack_args: Dict[str, List[object]],
    inputs: Dict[str, torch.Tensor],
    model: BaseModel,
    targeted_inputs: Optional[Dict[str, torch.Tensor]],
    orig_preds: Optional[Dict[str, torch.Tensor]]
):
    """Perform bim, pgd or cospgd adversarial attack on input images.

    Parameters
    ----------
    args : Namespace
        Arguments to configure the model and the validation.
    inputs : Dict[str, torch.Tensor]
        Input images and labels.
    model : BaseModel
        The model for adversarial attack.

    Returns
    -------
    torch.Tensor
        Perturbed images.
    """
    # import pdb

    iteration_metrics = {}

    if attack_args["attack_targeted"]:
        labels = targeted_inputs["flows"].squeeze(0)
    elif attack_args["attack_optim_target"] == "ground_truth":
        labels = inputs["flows"].clone().squeeze(0)
    elif attack_args["attack_optim_target"] == "initial_flow":
        labels = orig_preds["flows"].clone().squeeze(0)

    criterion = LossCriterion(attack_args["attack_loss"])

    image_1, image_2 = get_image_tensors(inputs, clone=True)

    orig_image_1, orig_image_2 = get_image_tensors(inputs)

    if "pgd" in attack_args["attack"]:
        if attack_args["attack_norm"] == "inf":
            image_1 = attack_functions.init_linf(
                image_1, epsilon=attack_args["attack_epsilon"], clamp_min=0, clamp_max=1
            )
            image_2 = attack_functions.init_linf(
                image_2, epsilon=attack_args["attack_epsilon"], clamp_min=0, clamp_max=1
            )
        elif attack_args["attack_norm"] == "two":
            image_1 = attack_functions.init_l2(
                image_1, epsilon=attack_args["attack_epsilon"], clamp_min=0, clamp_max=1
            )
            image_2 = attack_functions.init_l2(
                image_2, epsilon=attack_args["attack_epsilon"], clamp_min=0, clamp_max=1
            )
    else:
        attack_args["attack_alpha"] = attack_args["attack_epsilon"]

    perturbed_inputs = replace_images_dic(inputs, image_1, image_2, clone=True)

    perturbed_inputs["images"].requires_grad_(True)

    preds = model(perturbed_inputs)
    pred_flows = preds["flows"].squeeze(0)

    loss = criterion.loss(pred_flows.float(), labels.float())
    for t in range(attack_args["attack_iterations"]):
        if attack_args["attack"] == "cospgd":
            loss = attack_functions.cospgd_scale(
                predictions=pred_flows,
                labels=labels.float(),
                loss=loss,
                targeted=attack_args["attack_targeted"],
                one_hot=False,
            )
        loss = loss.mean()
        loss.backward()
        # pdb.set_trace()
        image_1_adv, image_2_adv = get_image_tensors(perturbed_inputs)
        image_1_grad, image_2_grad = get_image_grads(perturbed_inputs)
        if attack_args["attack_norm"] == "inf":
            image_1_adv = attack_functions.step_inf(
                perturbed_image=image_1_adv,
                epsilon=attack_args["attack_epsilon"],
                data_grad=image_1_grad,
                orig_image=orig_image_1,
                alpha=attack_args["attack_alpha"],
                targeted=attack_args["attack_targeted"],
                clamp_min=0,
                clamp_max=1,
                grad_scale=None,
            )
            image_2_adv = attack_functions.step_inf(
                perturbed_image=image_2_adv,
                epsilon=attack_args["attack_epsilon"],
                data_grad=image_2_grad,
                orig_image=orig_image_2,
                alpha=attack_args["attack_alpha"],
                targeted=attack_args["attack_targeted"],
                clamp_min=0,
                clamp_max=1,
                grad_scale=None,
            )
        elif attack_args["attack_norm"] == "two":
            image_1_adv = attack_functions.step_l2(
                perturbed_image=image_1_adv,
                epsilon=attack_args["attack_epsilon"],
                data_grad=image_1_grad,
                orig_image=orig_image_1,
                alpha=attack_args["attack_alpha"],
                targeted=attack_args["attack_targeted"],
                clamp_min=0,
                clamp_max=1,
                grad_scale=None,
            )
            image_2_adv = attack_functions.step_l2(
                perturbed_image=image_2_adv,
                epsilon=attack_args["attack_epsilon"],
                data_grad=image_2_grad,
                orig_image=orig_image_2,
                alpha=attack_args["attack_alpha"],
                targeted=attack_args["attack_targeted"],
                clamp_min=0,
                clamp_max=1,
                grad_scale=None,
            )
        # pdb.set_trace()
        perturbed_inputs = replace_images_dic(
            perturbed_inputs, image_1_adv, image_2_adv
        )

        perturbed_inputs["images"].requires_grad_(True)

        preds = model(perturbed_inputs)
        pred_flows = preds["flows"].squeeze(0)
        loss = criterion.loss(pred_flows.float(), labels.float())

        delta1 = image_1_adv - orig_image_1
        delta2 = image_2_adv - orig_image_2
        iteration_metrics = iteration_metrics | losses.calc_delta_metrics(
            delta1, delta2, t + 1
        )
        iteration_metrics = iteration_metrics | losses.calc_epe_metrics(
            model, preds, inputs, t + 1, targeted_inputs
        )

    loss = loss.mean()
    return preds, perturbed_inputs, iteration_metrics
