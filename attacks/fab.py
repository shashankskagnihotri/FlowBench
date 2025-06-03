from typing import Dict, List, Optional
import torch
from ptlflow.models.base_model.base_model import BaseModel
from attacks.adversarial_attacks_pytorch.torchattacks import FAB


def fab(
    attack_args: Dict[str, List[object]],
    inputs: Dict[str, torch.Tensor],
    model: BaseModel,
    targeted_inputs: Optional[Dict[str, torch.Tensor]],
):
    attack = FAB(model, attack_args["attack_epsilon"])
    if attack_args["attack_targeted"]:
        attack.targeted = True
        attack.set_mode_targeted_by_label()
        perturbed_inputs = attack(inputs["images"], targeted_inputs["flows"])
    else:
        perturbed_inputs = attack(inputs["images"], inputs["flows"])
    preds = model(perturbed_inputs)
    images = None
    labels = None

    return preds, perturbed_inputs  # images, labels, preds, None
