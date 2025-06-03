from argparse import Namespace
from typing import Any, Dict, List, Optional
import torch
from ptlflow.models.base_model.base_model import BaseModel
from attacks.attack_utils.utils import (
    get_image_tensors,
    get_image_grads,
    replace_images_dic,
    get_flow_tensors,
)
import torch.nn as nn
import attacks.attack_utils.loss_criterion as losses
import torch.optim as optim


def pcfa(
    attack_args: Dict[str, List[object]],
    inputs: Dict[str, torch.Tensor],
    model: BaseModel,
    targeted_inputs: Dict[str, torch.Tensor],
):
    """
    Performs an PCFA attack on a given model and for all images of a specified dataset.
    """

    optim_mu = 2500.0 / attack_args["attack_epsilon"]

    if attack_args["attack_epsilon"] == 0.01:
        optim_mu = 100000
    elif attack_args["attack_epsilon"] == 0.001:
        optim_mu = 1000000

    if attack_args["attack_target"] not in ["zero"]:
        optim_mu = 1.5 * optim_mu

    eps_box = attack_args["attack_alpha"]

    # Define what device we are using
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Make sure the model is not trained:
    for param in model.parameters():
        param.requires_grad = False

    preds, delta1, delta2, iteration_metrics = pcfa_attack(
        model, targeted_inputs, inputs, eps_box, device, optim_mu, attack_args
    )
    image1, image2 = get_image_tensors(targeted_inputs, clone=True)
    perturbed_image1 = image1 + delta1
    perturbed_image2 = image2 + delta2
    perturbed_inputs = replace_images_dic(
        targeted_inputs, perturbed_image1, perturbed_image2, clone=True
    )

    return preds, perturbed_inputs, iteration_metrics


def pcfa_attack(model, targeted_inputs, inputs, eps_box, device, optim_mu, attack_args):
    """Subroutine to optimize a PCFA perturbation on a given image pair. For a specified number of steps."""

    # For logging of different norms for the deltas and EPE for each iteration
    iteration_metrics = {}

    model = InputModel(
        model,
        eps_box,
        variable_change=attack_args["pcfa_boxconstraint"] in ["change_of_variables"],
    )

    image1, image2 = get_image_tensors(targeted_inputs)
    image1, image2 = image1.to(device), image2.to(device)

    # Set requires_grad attribute of tensor. Important for Attack
    image1.requires_grad = False
    image2.requires_grad = False

    # initialize perturbation and auxiliary variables:
    delta1 = torch.zeros_like(image1)
    delta2 = torch.zeros_like(image2)
    delta1 = delta1.to(device)
    delta2 = delta2.to(device)

    nw_input1 = None
    nw_input2 = None

    # Set up the optimizer and variables if individual perturbations delta1 and delta2 for images 1 and 2 should be trained
    delta1.requires_grad = False
    delta2.requires_grad = False

    if attack_args["pcfa_boxconstraint"] in ["change_of_variables"]:
        nw_input1 = torch.atanh(
            2.0 * (1.0 - eps_box) * (image1 + delta1) - (1 - eps_box)
        )
        nw_input2 = torch.atanh(
            2.0 * (1.0 - eps_box) * (image2 + delta2) - (1 - eps_box)
        )
    else:
        nw_input1 = image1 + delta1
        nw_input2 = image2 + delta2

    nw_input1.requires_grad = True
    nw_input2.requires_grad = True

    optimizer = optim.LBFGS([nw_input1, nw_input2], max_iter=10)

    # Predict the flow
    preds = model(inputs, nw_input1, nw_input2)
    flow_pred = preds["flows"].squeeze(0)
    flow_pred = flow_pred.to(device)

    # define target (potentially based on first flow prediction)
    target = get_flow_tensors(targeted_inputs)
    target = target.to(device)
    target.requires_grad = False

    # save nw_inputs, images, flow_pred, target
    #import pdb
    #pdb.set_trace()

    # Zero all existing gradients
    model.zero_grad()
    optimizer.zero_grad()

    for steps in range(attack_args["attack_iterations"]):
        # Calculate the deltas from the quantities that go into the network
        delta1, delta2 = extract_deltas(
            nw_input1,
            nw_input2,
            image1,
            image2,
            attack_args["pcfa_boxconstraint"],
            eps_box=eps_box,
        )
        # Calculate the loss
        loss = losses.loss_delta_constraint(
            flow_pred,
            target,
            delta1,
            delta2,
            device,
            delta_bound=attack_args["attack_epsilon"],
            mu=optim_mu,
            f_type=attack_args["attack_loss"],
        )
        # Update the optimization parameters
        loss.backward()
        # save deltas and loss
        #pdb.set_trace()
        def closure():
            optimizer.zero_grad()

            # Predict the flow
            flow_closure = model(inputs, nw_input1, nw_input2)["flows"].squeeze(0)
            flow_closure = flow_closure.to(device)

            delta1_closure, delta2_closure = extract_deltas(
                nw_input1,
                nw_input2,
                image1,
                image2,
                attack_args["pcfa_boxconstraint"],
                eps_box=eps_box,
            )
            loss_closure = losses.loss_delta_constraint(
                flow_closure,
                target,
                delta1_closure,
                delta2_closure,
                device,
                delta_bound=attack_args["attack_epsilon"],
                mu=optim_mu,
                f_type=attack_args["attack_loss"],
            )
            loss_closure.backward()

            return loss_closure

        # Update the optimization parameters
        optimizer.step(closure)

        # calculate the magnitude of the updated distortion
        delta1, delta2 = extract_deltas(
            nw_input1,
            nw_input2,
            image1,
            image2,
            attack_args["pcfa_boxconstraint"],
            eps_box=eps_box,
        )
        # The nw_inputs remain unchanged in this case, and can be directly fed into the network again for further perturbation training

        # Re-predict flow with the perturbed image, and update the flow prediction for the next iteration
        preds = model(inputs, nw_input1, nw_input2)
        flow_pred = preds["flows"].squeeze(0)
        flow_pred = flow_pred.to(device)

        #pdb.set_trace()
        # save nw_inputs and deltas and flow_pred

        iteration_metrics = iteration_metrics | losses.calc_delta_metrics(
            delta1, delta2, steps + 1
        )
        iteration_metrics = iteration_metrics | losses.calc_epe_metrics(
            model.model_loaded, preds, inputs, steps + 1, targeted_inputs
        )

    return preds, delta1, delta2, iteration_metrics


# For PCFA
def extract_deltas(nw_input1, nw_input2, image1, image2, boxconstraint, eps_box=0.0):

    if boxconstraint in ["change_of_variables"]:
        delta1 = (1.0 / 2.0) * 1.0 / (1.0 - eps_box) * (
            torch.tanh(nw_input1) + (1.0 - eps_box)
        ) - image1
        delta2 = (1.0 / 2.0) * 1.0 / (1.0 - eps_box) * (
            torch.tanh(nw_input2) + (1.0 - eps_box)
        ) - image2
    else:  # case clipping (ScaledInputModel also treats everything that is not change_of_variables by clipping it into range before feeding it to network.)
        delta1 = torch.clamp(nw_input1, 0.0, 1.0) - image1
        delta2 = torch.clamp(nw_input2, 0.0, 1.0) - image2

    return delta1, delta2


def torchfloat_to_float64(torch_float):
    """helper function to convert a torch.float to numpy float

    Args:
        torch_float (torch.float):
            scalar floating point number in torch

    Returns:
        numpy.float: floating point number in numpy
    """
    float_val = float(torch_float.detach().cpu().numpy())
    return float_val


class InputModel(nn.Module):
    def __init__(self, model, eps_box, variable_change=False, **kwargs):
        super(InputModel, self).__init__()

        self.var_change = variable_change

        self.eps_box = eps_box

        self.model_loaded = model

    def forward(self, inputs, image1, image2):
        # Perform the Carlini&Wagner Change of Variables, if the ScaledInputModel was configured to do so.
        if self.var_change:
            image1 = (
                (1.0 / 2.0)
                * 1.0
                / (1.0 - self.eps_box)
                * (torch.tanh(image1) + (1 - self.eps_box))
            )
            image2 = (
                (1.0 / 2.0)
                * 1.0
                / (1.0 - self.eps_box)
                * (torch.tanh(image2) + (1 - self.eps_box))
            )

        # Clipping case, which will only clip something if change of variables was not defined. otherwise, the change of variables has already brought the iamges into the range [0,1]
        image1 = torch.clamp(image1, 0.0, 1.0)
        image2 = torch.clamp(image2, 0.0, 1.0)

        dic = replace_images_dic(inputs, image1, image2, clone=True)

        return self.model_loaded(dic)
