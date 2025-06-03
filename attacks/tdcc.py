import logging
from torch.utils.data import Dataset
from ptlflow.utils.utils import config_logging
from ptlflow.utils.utils import (
    config_logging,
    make_divisible,
    bgr_val_as_tensor,
)
from torch.utils.data import DataLoader, Dataset

config_logging()
from ptlflow.data import flow_transforms as ft
from ptlflow.data.datasets_3DCC import (
    KittiDataset3DCC,
    SintelDataset3DCC,
)
import attacks.tdcc as tdcc

# call_dictionary = {"_get_kitti3DCC_dataset": _get_kitti3DCC_dataset}


def _get_kitti3DCC_dataset(
    model, is_train: bool, tdcc_corruption, tdcc_intensity, *args: str
) -> Dataset:
    device = "cuda" if model.args.train_transform_cuda else "cpu"
    md = make_divisible

    if is_train:
        if model.args.train_crop_size is None:
            cy, cx = (md(288, model.output_stride), md(960, model.output_stride))
            model.args.train_crop_size = (cy, cx)
            logging.warning(
                "--train_crop_size is not set. It will be set as (%d, %d).", cy, cx
            )
        else:
            cy, cx = (
                md(model.args.train_crop_size[0], model.output_stride),
                md(model.args.train_crop_size[1], model.output_stride),
            )

        # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
        transform = ft.Compose(
            [
                ft.ToTensor(device=device, fp16=model.args.train_transform_fp16),
                ft.RandomScaleAndCrop((cy, cx), (-0.2, 0.4), (-0.2, 0.2), sparse=True),
                ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                ft.GaussianNoise(0.02),
                ft.RandomPatchEraser(
                    0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                ),
            ]
        )
    else:
        transform = ft.ToTensor()

    versions = ["2012", "2015"]
    split = "trainval"
    for v in args:
        if v in ["2012", "2015"]:
            versions = [v]
        elif v in ["train", "val", "trainval", "test"]:
            split = v

    dataset = KittiDataset3DCC(
        model.args.kitti_2012_root_dir,
        model.args.kitti_2015_root_dir,
        model.args.kitti_2012_3DCC_root_dir,
        model.args.kitti_2015_3DCC_root_dir,
        tdcc_corruption=tdcc_corruption,
        tdcc_intensity=tdcc_intensity,
        versions=versions,
        split=split,
        transform=transform,
    )
    return dataset


def _get_sintel3DCC_dataset(
    model, is_train: bool, tdcc_corruption, tdcc_intensity, *args: str
) -> Dataset:
    device = "cuda" if model.args.train_transform_cuda else "cpu"
    md = make_divisible

    if is_train:
        if model.args.train_crop_size is None:
            cy, cx = (md(368, model.output_stride), md(768, model.output_stride))
            model.args.train_crop_size = (cy, cx)
            logging.warning(
                "--train_crop_size is not set. It will be set as (%d, %d).", cy, cx
            )
        else:
            cy, cx = (
                md(model.args.train_crop_size[0], model.output_stride),
                md(model.args.train_crop_size[1], model.output_stride),
            )

        # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
        transform = ft.Compose(
            [
                ft.ToTensor(device=device, fp16=model.args.train_transform_fp16),
                ft.RandomScaleAndCrop((cy, cx), (-0.2, 0.6), (-0.2, 0.2)),
                ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                ft.GaussianNoise(0.02),
                ft.RandomPatchEraser(
                    0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                ),
                ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
            ]
        )
    else:
        transform = ft.ToTensor()

    pass_names = ["clean", "final"]
    split = "trainval"
    get_occlusion_mask = False
    sequence_length = 2
    sequence_position = "first"
    for v in args:
        if v in ["clean", "final"]:
            pass_names = [v]
        elif v in ["train", "val", "trainval", "test"]:
            split = v
        elif v == "occ":
            get_occlusion_mask = True
        elif v.startswith("seqlen"):
            sequence_length = int(v.split("_")[1])
        elif v.startswith("seqpos"):
            sequence_position = v.split("_")[1]

    dataset = SintelDataset3DCC(
        model.args.mpi_sintel_root_dir,
        model.args.mpi_sintel_3DCC_root_dir,
        tdcc_corruption=tdcc_corruption,
        tdcc_intensity=tdcc_intensity,
        split=split,
        pass_names=pass_names,
        transform=transform,
        get_occlusion_mask=get_occlusion_mask,
        sequence_length=sequence_length,
        sequence_position=sequence_position,
    )
    return dataset


def get_dataset_3DCC(model, dataset_name, tdcc_corruption, tdcc_intensity):

    parsed_datasets = model.parse_dataset_selection(dataset_name)
    dataset_name = parsed_datasets[0][1]
    dataset = getattr(tdcc, f"_get_{dataset_name}3DCC_dataset")(
        model, False, tdcc_corruption, tdcc_intensity, *parsed_datasets[0][2:]
    )
    dataloader = DataLoader(
        dataset,
        1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader
