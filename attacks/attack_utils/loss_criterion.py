import torch
import torch.nn as nn
import numpy as np


class LossCriterion:
    def __init__(self, name="epe"):
        self.name = name
        self.nn_criterion = None
        if name == "mse":
            self.nn_criterion = nn.MSELoss(reduction="none")

    def loss(self, flow_1, flow_2):
        if self.name == "epe":
            return epe(flow_1, flow_2)
        else:
            return self.nn_criterion(flow_1, flow_2)


def epe(flow1, flow2):
    epe = torch.norm(flow1 - flow2, p=2, dim=1)
    return epe


def avg_epe(flow1, flow2):
    epe = torch.norm(flow1 - flow2, p=2, dim=1)
    return torch.mean(epe)


def calc_epe_metrics(model, preds, inputs, iteration, targeted_inputs=None):
    epe_gt = model.val_metrics(preds, inputs)["val/epe"]
    if targeted_inputs is None:
        return {f"val/epe_gt_i{iteration}": epe_gt}
    else:
        epe_tgt = model.val_metrics(preds, targeted_inputs)["val/epe"]
        return {
            f"val/epe_ground_truth_i{iteration}": epe_gt,
            f"val/epe_target_i{iteration}": epe_tgt,
        }


def calc_delta_metrics(delta1, delta2, iteration=None):
    l2_delta1 = two_norm_avg(delta1)
    l2_delta2 = two_norm_avg(delta2)
    l2_delta12 = two_norm_avg_delta(delta1, delta2)

    l0_delta1 = l0_norm(delta1)
    l0_delta2 = l0_norm(delta2)
    l0_delta12 = l0_norm_delta(delta1, delta2)

    l_inf_delta1 = l_infinity_norm(delta1)
    l_inf_delta2 = l_infinity_norm(delta2)
    l_inf_delta12 = l_infinity_norm_delta(delta1, delta2)

    if iteration is None:
        return {
            "val/l2_delta1": l2_delta1,
            "val/l2_delta2": l2_delta2,
            "val/l2_delta12": l2_delta12,
            "val/l0_delta1": l0_delta1,
            "val/l0_delta2": l0_delta2,
            "val/l0_delta12": l0_delta12,
            "val/l_inf_delta1": l_inf_delta1,
            "val/l_inf_delta2": l_inf_delta2,
            "val/l_inf_delta12": l_inf_delta12,
        }
    else:
        return {
            f"val/l2_delta1_i{iteration}": l2_delta1,
            f"val/l2_delta2_i{iteration}": l2_delta2,
            f"val/l2_delta12_i{iteration}": l2_delta12,
            f"val/l0_delta1_i{iteration}": l0_delta1,
            f"val/l0_delta2_i{iteration}": l0_delta2,
            f"val/l0_delta12_i{iteration}": l0_delta12,
            f"val/l_inf_delta1_i{iteration}": l_inf_delta1,
            f"val/l_inf_delta2_i{iteration}": l_inf_delta2,
            f"val/l_inf_delta12_i{iteration}": l_inf_delta12,
        }


def avg_mse(flow1, flow2):
    """Computes mean squared error between two flow fields.

    Args:
        flow1 (tensor):
            flow field, which must have the same dimension as flow2
        flow2 (tensor):
            flow field, which must have the same dimension as flow1

    Returns:
        float: scalar average squared end-point-error
    """
    return torch.mean((flow1 - flow2) ** 2)


def f_epe(pred, target):
    """Wrapper function to compute the average endpoint error between prediction and target

    Args:
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)

    Returns:
        float: scalar average endpoint error
    """
    return avg_epe(pred, target)


def f_mse(pred, target):
    """Wrapper function to compute the mean squared error between prediction and target

    Args:
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)

    Returns:
        float: scalar average squared end-point-error
    """
    return avg_mse(pred, target)


def f_cosim(pred, target):
    """Compute the mean cosine similarity between the two flow fields prediction and target

    Args:
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)

    Returns:
        float: scalar mean cosine similarity
    """
    return -1 * torch.mean(torch.nn.functional.cosine_similarity(pred, target))


def two_norm_avg_delta(delta1, delta2):
    """Computes the mean of the L2-norm of two perturbations used during PCFA.

    Args:
        delta1 (tensor):
            perturbation applied to the first image
        delta2 (tensor):
            perturbation applied to the second image

    Returns:
        float: scalar average L2-norm of two perturbations
    """
    numels_delta1 = torch.numel(delta1)
    numels_delta2 = torch.numel(delta2)
    sqrt_numels = (numels_delta1 + numels_delta2) ** (0.5)
    two_norm = torch.sqrt(
        torch.sum(torch.pow(torch.flatten(delta1), 2))
        + torch.sum(torch.pow(torch.flatten(delta2), 2))
    )
    return two_norm / sqrt_numels


def two_norm_avg_delta_squared(delta1, delta2):
    """Computes the mean of the squared L2-norm of two perturbations used during PCFA.

    Args:
        delta1 (tensor):
            perturbation applied to the first image
        delta2 (tensor):
            perturbation applied to the second image

    Returns:
        float: scalar average squared L2-norm of two perturbations
    """
    numels_delta1 = torch.numel(delta1)
    numels_delta2 = torch.numel(delta2)
    numels = numels_delta1 + numels_delta2
    two_norm = torch.sum(torch.pow(torch.flatten(delta1), 2)) + torch.sum(
        torch.pow(torch.flatten(delta2), 2)
    )
    return two_norm / numels


def two_norm_avg(x):
    """Computes the L2-norm of the input normalized by the root of the number of elements.

    Args:
        x (tensor):
            input tensor with variable dimensions

    Returns:
        float: normalized L2-norm
    """
    numels_x = torch.numel(x)
    sqrt_numels = numels_x**0.5
    two_norm = torch.sqrt(torch.sum(torch.pow(torch.flatten(x), 2)))
    return two_norm / sqrt_numels


def l0_norm(x):
    """Computes the L0-norm of the input, which is the count of non-zero elements.

    Args:
        x (tensor): input tensor with variable dimensions

    Returns:
        float: L0-norm (count of non-zero elements)
    """
    return torch.sum(x != 0).float()


def l_infinity_norm(x):
    """Computes the L-infinity norm of the input, which is the maximum absolute value.

    Args:
        x (tensor): input tensor with variable dimensions

    Returns:
        float: L-infinity norm (maximum absolute value)
    """
    return torch.max(torch.abs(x))


def l0_norm_delta(delta1, delta2):
    """Computes the sum of L0-norms of two perturbations.

    Args:
        delta1 (tensor): perturbation applied to the first image
        delta2 (tensor): perturbation applied to the second image

    Returns:
        float: sum of L0-norms of two perturbations
    """
    return l0_norm(delta1) + l0_norm(delta2)


def l_infinity_norm_delta(delta1, delta2):
    """Computes the maximum of the L-infinity norms of two perturbations.

    Args:
        delta1 (tensor): perturbation applied to the first image
        delta2 (tensor): perturbation applied to the second image

    Returns:
        float: maximum of the L-infinity norms of two perturbations
    """
    return max(l_infinity_norm(delta1), l_infinity_norm(delta2))


def get_loss(f_type, pred, target):
    """Wrapper to return a specified loss metric.

    Args:
        f_type (str):
            specifies the returned metric. Options: [epe | mse | cosim]
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)

    Raises:
        NotImplementedError: Unknown metric.

    Returns:
        float: scalar representing the loss measured with the specified norm
    """

    similarity_term = None

    if f_type == "epe":
        similarity_term = f_epe(pred, target)
    elif f_type == "cosim":
        similarity_term = f_cosim(pred, target)
    elif f_type == "mse":
        similarity_term = f_mse(pred, target)
    else:
        raise (
            NotImplementedError,
            "The requested loss type %s does not exist. Please choose one of 'epe', 'mse' or 'cosim'"
            % (f_type),
        )

    return similarity_term


def relu_penalty(delta1, delta2, device, delta_bound=0.001):
    """Implementation of the penalty term.
    The penalty function linearly penalizes deviations from a constraint and is otherwise zero.
    This is implemented using the ReLU function.

    Args:
        delta1 (tensor):
            perturbation for image1
        delta2 (tensor):
            perturbation for image2
        device (torch.device):
            changes the selected device
        delta_bound (float, optional):
            L2-constraint for the perturbation. Defaults to 0.001.

    Returns:
        float: scalar penalty value
    """
    zero_tensor = torch.tensor(0.0).to(device)
    delta_minus_bound = two_norm_avg_delta_squared(delta1, delta2) - torch.tensor(
        delta_bound**2
    ).to(device)
    return torch.max(
        zero_tensor, delta_minus_bound
    )  # This is relu( ||delta||**2-delta_bond**2).


def loss_delta_constraint(
    pred, target, delta1, delta2, device, delta_bound=0.001, mu=100.0, f_type="epe"
):
    """Penalty method to optimize the perturbations.
    An exact penalty function is used to transform the inequality constrained problem into an
    unconstrained optimization problem.

    Args:
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)
        delta1 (tensor):
            perturbation for image1
        delta2 (tensor):
            perturbation for image2
        device (torch.device):
            changes the selected device
        delta_bound (float, optional):
            L2-constraint for the perturbation. Defaults to 0.001.
        mu (_type_, optional):
            penalty parameter which enforces the unconstrained the specified constraint. Defaults to 100..
        f_type (str, optional):
            specifies the metric used for comparing prediction and target. Options: [epe | mse | cosim]. Defaults to "epe".

    Returns:
        _type_: _description_
    """

    similarity_term = get_loss(f_type, pred, target)
    penalty_term = relu_penalty(
        delta1, delta2, device, delta_bound
    )  # This is relu( ||delta||**2-delta_bond**2).

    return similarity_term + mu * penalty_term


def loss_weighted(pred, target, delta1, delta2, c=1.0, f_type="epe"):

    similarity_term = get_loss(f_type, pred, target)

    return two_norm_avg_delta(delta1, delta2) + c * similarity_term


def avg_sq_dist_weighted_depth(dist, depth):
    eps = 1e-7
    return torch.mean(torch.sum(dist**2, dim=-1) / (depth**2 + eps))


def mse_transparency_change(transparency, transparency_init):
    tr = 1.0 / 2.0 * (torch.tanh(transparency) + 1)
    tr_i = 1.0 / 2.0 * (torch.tanh(transparency_init) + 1)
    return torch.mean((tr - tr_i) ** 2)


def loss_weather(
    pred,
    target,
    f_type="aee",
    init_pos=None,
    offsets=None,
    alph_offsets=50,
    motion_offsets=None,
    alph_motion=100,
    flakes_transp=None,
    flakes_transp_init=None,
    alph_transp=10,
):

    similarity_term = get_loss(f_type, pred, target)

    offset_term = 0.0
    motion_offset_term = 0.0
    transp_term = 0.0

    if init_pos is not None:

        if alph_motion != 0 or alph_offsets != 0:
            depths = init_pos[..., 2]
            if alph_offsets != 0:
                offset_term = alph_offsets * avg_sq_dist_weighted_depth(
                    offsets[..., :3], depths
                )
            if alph_motion != 0:
                motion_offset_term = alph_motion * avg_sq_dist_weighted_depth(
                    motion_offsets[..., :3], depths
                )
        if alph_transp != 0:
            transp_term = alph_transp * mse_transparency_change(
                flakes_transp, flakes_transp_init
            )

    return similarity_term + offset_term + motion_offset_term + transp_term