from os import path
from attacks.help_function import ownutilities


def get_path_and_name(
    batch, name, extension, output_folder, offset=0, number_first=True, suffix=""
):
    number = f"{batch+offset:04d}"

    # check if name already includes lower dash - then remove, because it will be re-added.
    if name[-1] == "_":
        name = name[:-1]
    if suffix != "":
        suffix = "_" + suffix

    if number_first:
        filename = f"{number}_{name}{suffix}.{extension}"
    else:
        filename = f"{name}_{number}{suffix}.{extension}"
    filepath = path.join(output_folder, filename)

    return filename, filepath


def save_image(
    image_data,
    batch,
    output_folder,
    image_name="image",
    unit_input=True,
    normalize_max=None,
    unregistered_artifacts=True,
    offset=0,
    number_first=True,
    suffix="",
    force_name=False,
):
    """
    Saves a distortion tensor as .npy object to a specified output folder.
    In case the perturbation for image 1 and 2 are the same, setting common_perturbation=True only saves one instead of both distortions to save memory.

    Args:
        delta1 (tensor): the distortion tensor for image 1
        delta2 (tensor): the distortion tensor for image 2
        batch (int): a sample counter
        output_folder (str): the folder to which the distortion files should be saved
        common_perturbation (bool, optional): If true, only delta1 is saved because both delta1 and delta2 are assumed to be the same.
    """
    if not force_name:
        filename, filepath = get_path_and_name(
            batch,
            image_name,
            "png",
            output_folder,
            offset=offset,
            number_first=number_first,
            suffix=suffix,
        )
    else:
        filepath = path.join(output_folder, image_name)

    image_data = image_data.clone().detach()

    if normalize_max is not None:
        image_data = image_data / normalize_max / 2.0 + 0.5
        unit_input = True
    if unit_input:
        image_data = image_data * 255.0

    ownutilities.quickvisualization_tensor(image_data, filepath)


def save_flow(
    flow,
    batch,
    output_folder,
    flow_name="flowgt",
    auto_scale=True,
    max_scale=-1,
    unregistered_artifacts=True,
    offset=0,
    number_first=True,
):
    """
    Saves a distortion tensor as .npy object to a specified output folder.
    In case the perturbation for image 1 and 2 are the same, setting common_perturbation=True only saves one instead of both distortions to save memory.

    Args:
        delta1 (tensor): the distortion tensor for image 1
        delta2 (tensor): the distortion tensor for image 2
        batch (int): a sample counter
        output_folder (str): the folder to which the distortion files should be saved
    """
    filename, filepath = get_path_and_name(
        batch, flow_name, "png", output_folder, offset=offset, number_first=number_first
    )

    flow_data = flow.clone()
    ownutilities.quickvisualization_flow(
        flow_data, filepath, auto_scale=auto_scale, max_scale=max_scale
    )
