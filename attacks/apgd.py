from typing import Dict, List, Optional
import torch
from ptlflow.models.base_model.base_model import BaseModel
from attacks.adversarial_attacks_pytorch.torchattacks import APGD


def apgd(
    attack_args: Dict[str, List[object]],
    inputs: Dict[str, torch.Tensor],
    model: BaseModel,
    targeted_inputs: Optional[Dict[str, torch.Tensor]],
):
    norm = attack_args["attack_norm"]
    if norm == "inf":
        norm = "Linf"
    elif norm == "two":
        norm = "L2"

    if attack_args["attack_targeted"]:
        attack = APGD(
            model,
            eps=attack_args["attack_epsilon"],
            loss=attack_args["attack_loss"],
            verbose=False,
            steps=attack_args["apgd_steps"],
            n_restarts=attack_args["apgd_n_restarts"],
            seed=attack_args["apgd_seed"],
            rho=attack_args["apgd_rho"],
            eot_iter=attack_args["apgd_eot_iter"],
        )
        attack.targeted = True
        attack.set_mode_targeted_by_label()
        perturbed_images = attack(inputs["images"], targeted_inputs["flows"])
    else:
        attack = APGD(
            model,
            eps=attack_args["attack_epsilon"],
            loss=attack_args["attack_loss"],
            verbose=False,
            steps=attack_args["apgd_steps"],
            n_restarts=attack_args["apgd_n_restarts"],
            seed=attack_args["apgd_seed"],
            rho=attack_args["apgd_rho"],
            eot_iter=attack_args["apgd_eot_iter"],
        )
        perturbed_images = attack(inputs["images"], inputs["flows"])
    perturbed_inputs = inputs
    perturbed_inputs["images"] = perturbed_images
    preds = model(perturbed_inputs)
    images = None
    labels = None

    return preds, perturbed_inputs  # images, labels, preds, None
