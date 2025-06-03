"""Handle common datasets used in optical flow estimation."""

# =============================================================================
# Copyright 2021 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from ptlflow.data.datasets import BaseFlowDataset
import torch
from ptlflow.utils.utils import config_logging
from ptlflow.utils.utils import config_logging, make_divisible, bgr_val_as_tensor
from torch.utils.data import DataLoader, Dataset

config_logging()

THIS_DIR = Path(__file__).resolve().parent


class FlyingChairsDataset3DCC(BaseFlowDataset):
    # TODO: currently unchanged
    """Handle the FlyingChairs dataset."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_meta: bool = True,
    ) -> None:
        """Initialize FlyingChairsDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the FlyingChairs dataset.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval'}.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_meta : bool, default True
            Whether to get metadata.
        """
        super().__init__(
            dataset_name="FlyingChairs",
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=False,
            get_motion_boundary_mask=False,
            get_backward=False,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.split_file = THIS_DIR / "FlyingChairs_val.txt"

        # Read data from disk
        img1_paths = sorted((Path(self.root_dir) / "data").glob("*img1.ppm"))
        img2_paths = sorted((Path(self.root_dir) / "data").glob("*img2.ppm"))
        flow_paths = sorted((Path(self.root_dir) / "data").glob("*flow.flo"))

        # Sanity check
        assert len(img1_paths) == len(
            img2_paths
        ), f"{len(img1_paths)} vs {len(img2_paths)}"
        assert len(img1_paths) == len(
            flow_paths
        ), f"{len(img1_paths)} vs {len(flow_paths)}"

        with open(self.split_file, "r") as f:
            val_names = f.read().strip().splitlines()

        if split == "trainval":
            remove_names = []
        elif split == "train":
            remove_names = val_names
        elif split == "val":
            remove_names = [
                p.stem.split("_")[0]
                for p in img1_paths
                if p.stem.split("_")[0] not in val_names
            ]

        # Keep only data from the correct split
        self.img_paths = [
            [img1_paths[i], img2_paths[i]]
            for i in range(len(img1_paths))
            if img1_paths[i].stem.split("_")[0] not in remove_names
        ]
        self.flow_paths = [
            [flow_paths[i]]
            for i in range(len(flow_paths))
            if flow_paths[i].stem.split("_")[0] not in remove_names
        ]
        assert len(self.img_paths) == len(
            self.flow_paths
        ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"

        self.metadata = [
            {
                "image_paths": [str(p) for p in paths],
                "is_val": paths[0].stem in val_names,
                "misc": "",
                "is_seq_start": True,
            }
            for paths in self.img_paths
        ]

        self._log_status()


class FlyingChairs2Dataset3DCC(BaseFlowDataset):
    # TODO: Currently unchanged
    """Handle the FlyingChairs 2 dataset."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        add_reverse: bool = True,
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 1000.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = True,
        get_motion_boundary_mask: bool = True,
        get_backward: bool = True,
        get_meta: bool = True,
    ) -> None:
        """Initialize FlyingChairs2Dataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the FlyingChairs2 dataset.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval'}.
        add_reverse : bool, default True
            If True, double the number of samples by appending the backward samples as additional samples.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_occlusion_mask : bool, default True
            Whether to get occlusion masks.
        get_motion_boundary_mask : bool, default True
            Whether to get motion boundary masks.
        get_backward : bool, default True
            Whether to get the occluded version of the inputs.
        get_meta : bool, default True
            Whether to get metadata.
        """
        super().__init__(
            dataset_name="FlyingChairs2",
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=get_occlusion_mask,
            get_motion_boundary_mask=get_motion_boundary_mask,
            get_backward=get_backward,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.add_reverse = add_reverse

        if split == "train":
            dir_names = ["train"]
        elif split == "val":
            dir_names = ["val"]
        else:
            dir_names = ["train", "val"]

        for dname in dir_names:
            # Read data from disk
            img1_paths = sorted((Path(self.root_dir) / dname).glob("*img_0.png"))
            img2_paths = sorted((Path(self.root_dir) / dname).glob("*img_1.png"))
            self.img_paths.extend(
                [[img1_paths[i], img2_paths[i]] for i in range(len(img1_paths))]
            )
            self.flow_paths.extend(
                [
                    [x]
                    for x in sorted((Path(self.root_dir) / dname).glob("*flow_01.flo"))
                ]
            )
            self.occ_paths.extend(
                [[x] for x in sorted((Path(self.root_dir) / dname).glob("*occ_01.png"))]
            )
            self.mb_paths.extend(
                [[x] for x in sorted((Path(self.root_dir) / dname).glob("*mb_01.png"))]
            )
            if self.get_backward:
                self.flow_b_paths.extend(
                    [
                        [x]
                        for x in sorted(
                            (Path(self.root_dir) / dname).glob("*flow_10.flo")
                        )
                    ]
                )
                self.occ_b_paths.extend(
                    [
                        [x]
                        for x in sorted(
                            (Path(self.root_dir) / dname).glob("*occ_10.png")
                        )
                    ]
                )
                self.mb_b_paths.extend(
                    [
                        [x]
                        for x in sorted(
                            (Path(self.root_dir) / dname).glob("*mb_10.png")
                        )
                    ]
                )
            if self.add_reverse:
                self.img_paths.extend(
                    [[img2_paths[i], img1_paths[i]] for i in range(len(img1_paths))]
                )
                self.flow_paths.extend(
                    [
                        [x]
                        for x in sorted(
                            (Path(self.root_dir) / dname).glob("*flow_10.flo")
                        )
                    ]
                )
                self.occ_paths.extend(
                    [
                        [x]
                        for x in sorted(
                            (Path(self.root_dir) / dname).glob("*occ_10.png")
                        )
                    ]
                )
                self.mb_paths.extend(
                    [
                        [x]
                        for x in sorted(
                            (Path(self.root_dir) / dname).glob("*mb_10.png")
                        )
                    ]
                )
                if self.get_backward:
                    self.flow_b_paths.extend(
                        [
                            [x]
                            for x in sorted(
                                (Path(self.root_dir) / dname).glob("*flow_01.flo")
                            )
                        ]
                    )
                    self.occ_b_paths.extend(
                        [
                            [x]
                            for x in sorted(
                                (Path(self.root_dir) / dname).glob("*occ_01.png")
                            )
                        ]
                    )
                    self.mb_b_paths.extend(
                        [
                            [x]
                            for x in sorted(
                                (Path(self.root_dir) / dname).glob("*mb_01.png")
                            )
                        ]
                    )

        self.metadata = [
            {
                "image_paths": [str(p) for p in paths],
                "is_val": False,
                "misc": "",
                "is_seq_start": True,
            }
            for paths in self.img_paths
        ]

        # Sanity check
        assert len(img1_paths) == len(
            img2_paths
        ), f"{len(img1_paths)} vs {len(img2_paths)}"
        assert len(self.img_paths) == len(
            self.flow_paths
        ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"
        assert len(self.img_paths) == len(
            self.occ_paths
        ), f"{len(self.img_paths)} vs {len(self.occ_paths)}"
        assert len(self.img_paths) == len(
            self.mb_paths
        ), f"{len(self.img_paths)} vs {len(self.mb_paths)}"
        if self.get_backward:
            assert len(self.img_paths) == len(
                self.flow_b_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_b_paths)}"
            assert len(self.img_paths) == len(
                self.occ_b_paths
            ), f"{len(self.img_paths)} vs {len(self.occ_b_paths)}"
            assert len(self.img_paths) == len(
                self.mb_b_paths
            ), f"{len(self.img_paths)} vs {len(self.mb_b_paths)}"

        self._log_status()


class FlyingThings3DDataset3DCC(BaseFlowDataset):
    # TODO: currently unchanged
    """Handle the FlyingThings3D dataset.

    Note that this only works for the complete FlyingThings3D dataset. For the subset version, use FlyingThings3DSubsetDataset.
    """

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        split: str = "train",
        pass_names: Union[str, List[str]] = "clean",
        side_names: Union[str, List[str]] = "left",
        add_reverse: bool = True,
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 1000.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = True,
        get_motion_boundary_mask: bool = True,
        get_backward: bool = True,
        get_meta: bool = True,
        sequence_length: int = 2,
        sequence_position: str = "first",
    ) -> None:
        """Initialize FlyingThings3DDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the FlyingThings3D dataset.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval'}.
        pass_names : Union[str, List[str]], default 'clean'
            Which passes should be loaded. It can be one of {'clean', 'final', ['clean', 'final']}.
        side_names : Union[str, List[str]], default 'left'
             Samples from which side view should be loaded. It can be one of {'left', 'right', ['left', 'right']}.
        add_reverse : bool, default True
            If True, double the number of samples by appending the backward samples as additional samples.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_occlusion_mask : bool, default True
            Whether to get occlusion masks.
        get_motion_boundary_mask : bool, default True
            Whether to get motion boundary masks.
        get_backward : bool, default True
            Whether to get the backward version of the inputs.
        get_meta : bool, default True
            Whether to get metadata.
        sequence_length : int, default 2
            How many consecutive images are loaded per sample. More than two images can be used for model which exploit more
            temporal information.
        sequence_position : str, default "first"
            Only used when sequence_length > 2.
            Determines the position where the main image frame will be in the sequence. It can one of three values:
            - "first": the main frame will be the first one of the sequence,
            - "middle": the main frame will be in the middle of the sequence (at position sequence_length // 2),
            - "last": the main frame will be the penultimate in the sequence.
        """
        super().__init__(
            dataset_name="FlyingThings3D",
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=get_occlusion_mask,
            get_motion_boundary_mask=get_motion_boundary_mask,
            get_backward=get_backward,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.add_reverse = add_reverse
        self.pass_names = pass_names
        self.sequence_length = sequence_length
        self.sequence_position = sequence_position
        if isinstance(self.pass_names, str):
            self.pass_names = [self.pass_names]
        self.side_names = side_names
        if isinstance(self.side_names, str):
            self.side_names = [self.side_names]

        if split == "val":
            split_dir_names = ["TEST"]
        elif split == "train":
            split_dir_names = ["TRAIN"]
        else:
            split_dir_names = ["TRAIN", "TEST"]

        pass_dirs = [f"frames_{p}pass" for p in self.pass_names]

        directions = [("into_future", "into_past")]
        reverts = [False]
        if self.add_reverse:
            directions.append(("into_past", "into_future"))
            reverts.append(True)

        # Read paths from disk
        for passd in pass_dirs:
            for split in split_dir_names:
                split_path = Path(self.root_dir) / passd / split
                for letter_path in split_path.glob("*"):
                    for seq_path in letter_path.glob("*"):
                        for direcs, rev in zip(directions, reverts):
                            for side in self.side_names:
                                image_paths = sorted(
                                    (seq_path / side).glob("*.png"), reverse=rev
                                )
                                image_paths = self._extend_paths_list(
                                    image_paths, sequence_length, sequence_position
                                )
                                flow_paths = sorted(
                                    (
                                        Path(
                                            str(seq_path).replace(passd, "optical_flow")
                                        )
                                        / direcs[0]
                                        / side
                                    ).glob("*.pfm"),
                                    reverse=rev,
                                )
                                flow_paths = self._extend_paths_list(
                                    flow_paths, sequence_length, sequence_position
                                )

                                occ_paths = []
                                if (Path(self.root_dir) / "occlusions").exists():
                                    occ_paths = sorted(
                                        (
                                            Path(
                                                str(seq_path).replace(
                                                    passd, "occlusions"
                                                )
                                            )
                                            / direcs[0]
                                            / side
                                        ).glob("*.png"),
                                        reverse=rev,
                                    )
                                    occ_paths = self._extend_paths_list(
                                        occ_paths, sequence_length, sequence_position
                                    )
                                mb_paths = []
                                if (Path(self.root_dir) / "motion_boundaries").exists():
                                    mb_paths = sorted(
                                        (
                                            Path(
                                                str(seq_path).replace(
                                                    passd, "motion_boundaries"
                                                )
                                            )
                                            / direcs[0]
                                            / side
                                        ).glob("*.png"),
                                        reverse=rev,
                                    )
                                    mb_paths = self._extend_paths_list(
                                        mb_paths, sequence_length, sequence_position
                                    )

                                flow_b_paths = []
                                occ_b_paths = []
                                mb_b_paths = []
                                if self.get_backward:
                                    flow_b_paths = sorted(
                                        (
                                            Path(
                                                str(seq_path).replace(
                                                    passd, "optical_flow"
                                                )
                                            )
                                            / direcs[1]
                                            / side
                                        ).glob("*.pfm"),
                                        reverse=rev,
                                    )
                                    flow_b_paths = self._extend_paths_list(
                                        flow_b_paths, sequence_length, sequence_position
                                    )
                                    if (Path(self.root_dir) / "occlusions").exists():
                                        occ_b_paths = sorted(
                                            (
                                                Path(
                                                    str(seq_path).replace(
                                                        passd, "occlusions"
                                                    )
                                                )
                                                / direcs[1]
                                                / side
                                            ).glob("*.png"),
                                            reverse=rev,
                                        )
                                        occ_b_paths = self._extend_paths_list(
                                            occ_b_paths,
                                            sequence_length,
                                            sequence_position,
                                        )
                                    if (
                                        Path(self.root_dir) / "motion_boundaries"
                                    ).exists():
                                        mb_b_paths = sorted(
                                            (
                                                Path(
                                                    str(seq_path).replace(
                                                        passd, "motion_boundaries"
                                                    )
                                                )
                                                / direcs[1]
                                                / side
                                            ).glob("*.png"),
                                            reverse=rev,
                                        )
                                        mb_b_paths = self._extend_paths_list(
                                            mb_b_paths,
                                            sequence_length,
                                            sequence_position,
                                        )

                                for i in range(
                                    len(image_paths) - self.sequence_length + 1
                                ):
                                    self.img_paths.append(
                                        image_paths[i : i + self.sequence_length]
                                    )
                                    if len(flow_paths) > 0:
                                        self.flow_paths.append(
                                            flow_paths[i : i + self.sequence_length - 1]
                                        )
                                    if len(occ_paths) > 0:
                                        self.occ_paths.append(
                                            occ_paths[i : i + self.sequence_length - 1]
                                        )
                                    if len(mb_paths) > 0:
                                        self.mb_paths.append(
                                            mb_paths[i : i + self.sequence_length - 1]
                                        )
                                    self.metadata.append(
                                        {
                                            "image_paths": [
                                                str(p)
                                                for p in image_paths[
                                                    i : i + self.sequence_length
                                                ]
                                            ],
                                            "is_val": False,
                                            "misc": "",
                                            "is_seq_start": i == 0,
                                        }
                                    )
                                    if self.get_backward:
                                        if len(flow_b_paths) > 0:
                                            self.flow_b_paths.append(
                                                flow_b_paths[
                                                    i + 1 : i + self.sequence_length
                                                ]
                                            )
                                        if len(occ_b_paths) > 0:
                                            self.occ_b_paths.append(
                                                occ_b_paths[
                                                    i + 1 : i + self.sequence_length
                                                ]
                                            )
                                        if len(mb_b_paths) > 0:
                                            self.mb_b_paths.append(
                                                mb_b_paths[
                                                    i + 1 : i + self.sequence_length
                                                ]
                                            )

        assert len(self.img_paths) == len(
            self.flow_paths
        ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"
        assert len(self.occ_paths) == 0 or len(self.img_paths) == len(
            self.occ_paths
        ), f"{len(self.img_paths)} vs {len(self.occ_paths)}"
        assert len(self.mb_paths) == 0 or len(self.img_paths) == len(
            self.mb_paths
        ), f"{len(self.img_paths)} vs {len(self.mb_paths)}"
        if self.get_backward:
            assert len(self.img_paths) == len(
                self.flow_b_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_b_paths)}"
            assert len(self.occ_b_paths) == 0 or len(self.img_paths) == len(
                self.occ_b_paths
            ), f"{len(self.img_paths)} vs {len(self.occ_b_paths)}"
            assert len(self.mb_b_paths) == 0 or len(self.img_paths) == len(
                self.mb_b_paths
            ), f"{len(self.img_paths)} vs {len(self.mb_b_paths)}"

        self._log_status()


class FlyingThings3DSubsetDataset3DCC(BaseFlowDataset):
    # TODO: currently unchanged
    """Handle the FlyingThings3D subset dataset.

    Note that this only works for the FlyingThings3D subset dataset. For the complete version, use FlyingThings3DDataset.
    """

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        split: str = "train",
        pass_names: Union[str, List[str]] = "clean",
        side_names: Union[str, List[str]] = "left",
        add_reverse: bool = True,
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 1000.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = True,
        get_motion_boundary_mask: bool = True,
        get_backward: bool = True,
        get_meta: bool = True,
        sequence_length: int = 2,
        sequence_position: str = "first",
    ) -> None:
        """Initialize FlyingThings3DSubsetDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the FlyingThings3D dataset.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval'}.
        pass_names : Union[str, List[str]], default 'clean'
            Which passes should be loaded. It can be one of {'clean', 'final', ['clean', 'final']}.
        side_names : Union[str, List[str]], default 'left'
             Samples from which side view should be loaded. It can be one of {'left', 'right', ['left', 'right']}.
        add_reverse : bool, default True
            If True, double the number of samples by appending the backward samples as additional samples.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_occlusion_mask : bool, default True
            Whether to get occlusion masks.
        get_motion_boundary_mask : bool, default True
            Whether to get motion boundary masks.
        get_backward : bool, default True
            Whether to get the occluded version of the inputs.
        get_meta : bool, default True
            Whether to get metadata.
        sequence_length : int, default 2
            How many consecutive images are loaded per sample. More than two images can be used for model which exploit more
            temporal information.
        sequence_position : str, default "first"
            Only used when sequence_length > 2.
            Determines the position where the main image frame will be in the sequence. It can one of three values:
            - "first": the main frame will be the first one of the sequence,
            - "middle": the main frame will be in the middle of the sequence (at position sequence_length // 2),
            - "last": the main frame will be the penultimate in the sequence.
        """
        super().__init__(
            dataset_name="FlyingThings3DSubset",
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=get_occlusion_mask,
            get_motion_boundary_mask=get_motion_boundary_mask,
            get_backward=get_backward,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.add_reverse = add_reverse
        self.pass_names = pass_names
        self.sequence_length = sequence_length
        self.sequence_position = sequence_position
        if isinstance(self.pass_names, str):
            self.pass_names = [self.pass_names]
        self.side_names = side_names
        if isinstance(self.side_names, str):
            self.side_names = [self.side_names]

        if split == "train" or split == "val":
            split_dir_names = [split]
        else:
            split_dir_names = ["train", "val"]

        directions = [("into_future", "into_past")]
        reverts = [False]
        if self.add_reverse:
            directions.append(("into_past", "into_future"))
            reverts.append(True)

        # Read paths from disk
        for split in split_dir_names:
            for pass_name in self.pass_names:
                for side in self.side_names:
                    for direcs, rev in zip(directions, reverts):
                        flow_dir = (
                            Path(self.root_dir) / split / "flow" / side / direcs[0]
                        )
                        flow_paths = sorted(flow_dir.glob("*.flo"), reverse=rev)

                        # Create groups to separate different sequences
                        flow_groups_paths = [[flow_paths[0]]]
                        prev_idx = int(flow_paths[0].stem)
                        for path in flow_paths[1:]:
                            idx = int(path.stem)
                            if (idx - 1) == prev_idx:
                                flow_groups_paths[-1].append(path)
                            else:
                                flow_groups_paths.append([path])
                            prev_idx = idx

                        for flow_group in flow_groups_paths:
                            flow_group = self._extend_paths_list(
                                flow_group, sequence_length, sequence_position
                            )
                            for i in range(len(flow_group) - self.sequence_length + 2):
                                flow_paths = flow_group[
                                    i : i + self.sequence_length - 1
                                ]
                                self.flow_paths.append(flow_paths)

                                img_dir = (
                                    Path(self.root_dir)
                                    / split
                                    / f"image_{pass_name}"
                                    / side
                                )
                                img_paths = [
                                    img_dir / (fp.stem + ".png") for fp in flow_paths
                                ]
                                if rev:
                                    idx = int(img_paths[0].stem) - 1
                                else:
                                    idx = int(img_paths[-1].stem) + 1
                                img_paths.append(img_dir / f"{idx:07d}.png")
                                self.img_paths.append(img_paths)

                                if (
                                    Path(self.root_dir) / split / "flow_occlusions"
                                ).exists():
                                    occ_paths = [
                                        Path(
                                            str(fp)
                                            .replace("flow", "flow_occlusions")
                                            .replace(".flo", ".png")
                                        )
                                        for fp in flow_paths
                                    ]
                                    self.occ_paths.append(occ_paths)
                                if (
                                    Path(self.root_dir) / split / "motion_boundaries"
                                ).exists():
                                    mb_paths = [
                                        Path(
                                            str(fp)
                                            .replace("flow", "motion_boundaries")
                                            .replace(".flo", ".png")
                                        )
                                        for fp in flow_paths
                                    ]
                                    self.mb_paths.append(mb_paths)

                                self.metadata.append(
                                    {
                                        "image_paths": [str(p) for p in img_paths],
                                        "is_val": False,
                                        "misc": "",
                                        "is_seq_start": i == 0,
                                    }
                                )

                        if self.get_backward:
                            flow_dir = (
                                Path(self.root_dir) / split / "flow" / side / direcs[1]
                            )
                            flow_paths = sorted(flow_dir.glob("*.flo"), reverse=rev)

                            # Create groups to separate different sequences
                            flow_groups_paths = [[flow_paths[0]]]
                            prev_idx = int(flow_paths[0].stem)
                            for path in flow_paths[1:]:
                                idx = int(path.stem)
                                if (idx - 1) == prev_idx:
                                    flow_groups_paths[-1].append(path)
                                else:
                                    flow_groups_paths.append([path])
                                prev_idx = idx

                            for flow_group in flow_groups_paths:
                                flow_group = self._extend_paths_list(
                                    flow_group, sequence_length, sequence_position
                                )
                                for i in range(
                                    len(flow_group) - self.sequence_length + 2
                                ):
                                    flow_paths = flow_group[
                                        i : i + self.sequence_length - 1
                                    ]
                                    self.flow_b_paths.append(flow_paths)

                                    if (
                                        Path(self.root_dir) / split / "flow_occlusions"
                                    ).exists():
                                        occ_paths = [
                                            Path(
                                                str(fp)
                                                .replace("flow", "flow_occlusions")
                                                .replace(".flo", ".png")
                                            )
                                            for fp in flow_paths
                                        ]
                                        self.occ_b_paths.append(occ_paths)
                                    if (
                                        Path(self.root_dir)
                                        / split
                                        / "motion_boundaries"
                                    ).exists():
                                        mb_paths = [
                                            Path(
                                                str(fp)
                                                .replace("flow", "motion_boundaries")
                                                .replace(".flo", ".png")
                                            )
                                            for fp in flow_paths
                                        ]
                                        self.mb_b_paths.append(mb_paths)

        assert len(self.img_paths) == len(
            self.flow_paths
        ), f"{len(self.img_paths)} vs  {len(self.flow_paths)}"
        assert len(self.occ_paths) == 0 or len(self.img_paths) == len(
            self.occ_paths
        ), f"{len(self.img_paths)} vs {len(self.occ_paths)}"
        assert len(self.mb_paths) == 0 or len(self.img_paths) == len(
            self.mb_paths
        ), f"{len(self.img_paths)} vs {len(self.mb_paths)}"
        if self.get_backward:
            assert len(self.img_paths) == len(
                self.flow_b_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_b_paths)}"
            assert len(self.occ_b_paths) == 0 or len(self.img_paths) == len(
                self.occ_b_paths
            ), f"{len(self.img_paths)} vs {len(self.occ_b_paths)}"
            assert len(self.mb_b_paths) == 0 or len(self.img_paths) == len(
                self.mb_b_paths
            ), f"{len(self.img_paths)} vs {len(self.mb_b_paths)}"

        self._log_status()


class Hd1kDataset3DCC(BaseFlowDataset):
    # TODO: unchanged
    """Handle the HD1K dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        split: str = "train",
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 512.0,
        get_valid_mask: bool = True,
        get_meta: bool = True,
        sequence_length: int = 2,
        sequence_position: str = "first",
    ) -> None:
        """Initialize Hd1kDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the HD1K dataset.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval', 'test'}.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 512.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_meta : bool, default True
            Whether to get metadata.
        sequence_length : int, default 2
            How many consecutive images are loaded per sample. More than two images can be used for model which exploit more
            temporal information.
        sequence_position : str, default "first"
            Only used when sequence_length > 2.
            Determines the position where the main image frame will be in the sequence. It can one of three values:
            - "first": the main frame will be the first one of the sequence,
            - "middle": the main frame will be in the middle of the sequence (at position sequence_length // 2),
            - "last": the main frame will be the penultimate in the sequence.
        """
        super().__init__(
            dataset_name="HD1K",
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=False,
            get_motion_boundary_mask=False,
            get_backward=False,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.split = split
        self.sequence_length = sequence_length
        self.sequence_position = sequence_position

        if split == "test":
            split_dir = "hd1k_challenge"
        else:
            split_dir = "hd1k_input"

        img_paths = sorted((Path(root_dir) / split_dir / "image_2").glob("*.png"))
        img_names = [p.stem for p in img_paths]

        # Group paths by sequence
        img_names_grouped = {}
        for n in img_names:
            seq_name = n.split("_")[0]
            if img_names_grouped.get(seq_name) is None:
                img_names_grouped[seq_name] = []
            img_names_grouped[seq_name].append(n)

        val_names = []
        split_file = THIS_DIR / "Hd1k_val.txt"
        with open(split_file, "r") as f:
            val_names = f.read().strip().splitlines()

        # Remove names that do not belong to the chosen split
        for seq_name, seq_img_names in img_names_grouped.items():
            if split == "train":
                img_names_grouped[seq_name] = [
                    n for n in seq_img_names if n not in val_names
                ]
            elif split == "val":
                img_names_grouped[seq_name] = [
                    n for n in seq_img_names if n in val_names
                ]

        for seq_img_names in img_names_grouped.values():
            seq_img_names = self._extend_paths_list(
                seq_img_names, sequence_length, sequence_position
            )
            for i in range(len(seq_img_names) - self.sequence_length + 1):
                self.img_paths.append(
                    [
                        Path(root_dir) / split_dir / "image_2" / (n + ".png")
                        for n in seq_img_names[i : i + self.sequence_length]
                    ]
                )
                if split != "test":
                    self.flow_paths.append(
                        [
                            Path(root_dir) / "hd1k_flow_gt" / "flow_occ" / (n + ".png")
                            for n in seq_img_names[i : i + self.sequence_length - 1]
                        ]
                    )

                self.metadata.append(
                    {
                        "image_paths": [str(p) for p in self.img_paths[-1]],
                        "is_val": (seq_img_names[i] in val_names),
                        "misc": "",
                        "is_seq_start": True,
                    }
                )

        if split != "test":
            assert len(self.img_paths) == len(
                self.flow_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"

        self._log_status()


class KittiDataset3DCC(BaseFlowDataset):
    # TODO: currently unchanged
    """Handle the KITTI dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir_2012: Optional[str] = None,
        root_dir_2015: Optional[str] = None,
        root_dir_2012_3DCC: Optional[str] = None,
        root_dir_2015_3DCC: Optional[str] = None,
        tdcc_intensity: int = 3,
        tdcc_corruption: str = "far_focus",
        split: str = "train",
        versions: Union[str, List[str]] = "2015",
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 512.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = False,
        get_meta: bool = True,
    ) -> None:
        """Initialize KittiDataset.

        Parameters
        ----------
        root_dir_2012 : str, optional.
            Path to the root directory of the KITTI 2012 dataset, if available.
        root_dir_2015 : str, optional.
            Path to the root directory of the KITTI 2015 dataset, if available.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval', 'test'}.
        versions : Union[str, List[str]], default '2015'
            Which version should be loaded. It can be one of {'2012', '2015', ['2012', '2015']}.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 512.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_occlusion_mask : bool, default True
            Whether to get occlusion masks.
        get_meta : bool, default True
            Whether to get metadata.
        """
        if isinstance(versions, str):
            versions = [versions]
        super().__init__(
            dataset_name=f'KITTI_{"_".join(versions)}',
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=get_occlusion_mask,
            get_motion_boundary_mask=False,
            get_backward=False,
            get_meta=get_meta,
        )
        self.root_dir = {"2012": root_dir_2012, "2015": root_dir_2015}
        self.root_dir_corrupted = {
            "2012": root_dir_2012_3DCC,
            "2015": root_dir_2015_3DCC,
        }
        self.versions = versions
        self.split = split

        if split == "test":
            split_dir = "testing"
        else:
            split_dir = "training"

        for ver in versions:
            if self.root_dir[ver] is None:
                continue

            if ver == "2012":
                image_dir = "colored_0"
            else:
                image_dir = "image_2"

            img1_paths = sorted(
                (
                    Path(self.root_dir_corrupted[ver])
                    / split_dir
                    / image_dir
                    / tdcc_corruption
                    / str(tdcc_intensity)
                ).glob("*_10.png")
            )
            img2_paths = sorted(
                (
                    Path(self.root_dir_corrupted[ver])
                    / split_dir
                    / image_dir
                    / tdcc_corruption
                    / str(tdcc_intensity)
                ).glob("*_11.png")
            )
            assert len(img1_paths) == len(
                img2_paths
            ), f"{len(img1_paths)} vs {len(img2_paths)}"
            flow_paths = []
            if split != "test":
                flow_paths = sorted(
                    (Path(self.root_dir[ver]) / split_dir / "flow_occ").glob("*_10.png")
                )
                assert len(img1_paths) == len(
                    flow_paths
                ), f"{len(img1_paths)} vs {len(flow_paths)}"

            split_file = THIS_DIR / f"Kitti{ver}_val.txt"
            with open(split_file, "r") as f:
                val_names = f.read().strip().splitlines()

            if split == "trainval" or split == "test":
                remove_names = []
            elif split == "train":
                remove_names = val_names
            elif split == "val":
                remove_names = [p.stem for p in img1_paths if p.stem not in val_names]

            self.img_paths.extend(
                [
                    [img1_paths[i], img2_paths[i]]
                    for i in range(len(img1_paths))
                    if img1_paths[i].stem not in remove_names
                ]
            )
            if split != "test":
                self.flow_paths.extend(
                    [
                        [flow_paths[i]]
                        for i in range(len(flow_paths))
                        if flow_paths[i].stem not in remove_names
                    ]
                )
            self.metadata.extend(
                [
                    {
                        "image_paths": [str(img1_paths[i]), str(img2_paths[i])],
                        "is_val": img1_paths[i].stem in val_names,
                        "misc": ver,
                        "is_seq_start": True,
                    }
                    for i in range(len(img1_paths))
                    if img1_paths[i].stem not in remove_names
                ]
            )

        if split != "test":
            assert len(self.img_paths) == len(
                self.flow_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"

        self._log_status()


class SintelDataset3DCC(BaseFlowDataset):
    # TODO: currently unchanged
    """Handle the MPI Sintel dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        root_dir_3DCC: str,
        tdcc_intensity: int = 3,
        tdcc_corruption: str = "far_focus",
        split: str = "train",
        pass_names: Union[str, List[str]] = "clean",
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = True,
        get_meta: bool = True,
        sequence_length: int = 2,
        sequence_position: str = "first",
    ) -> None:
        """Initialize SintelDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the MPI Sintel dataset.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval', 'test'}.
        pass_names : Union[str, List[str]], default 'clean'
            Which passes should be loaded. It can be one of {'clean', 'final', ['clean', 'final']}.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_occlusion_mask : bool, default True
            Whether to get occlusion masks.
        get_meta : bool, default True
            Whether to get metadata.
        sequence_length : int, default 2
            How many consecutive images are loaded per sample. More than two images can be used for model which exploit more
            temporal information.
        sequence_position : str, default "first"
            Only used when sequence_length > 2.
            Determines the position where the main image frame will be in the sequence. It can one of three values:
            - "first": the main frame will be the first one of the sequence,
            - "middle": the main frame will be in the middle of the sequence (at position sequence_length // 2),
            - "last": the main frame will be the penultimate in the sequence.
        """
        if isinstance(pass_names, str):
            pass_names = [pass_names]
        super().__init__(
            dataset_name=f'Sintel_{"_".join(pass_names)}',
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=get_occlusion_mask,
            get_motion_boundary_mask=False,
            get_backward=False,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.root_dir_3DCC = root_dir_3DCC
        self.split = split
        self.pass_names = pass_names
        self.sequence_length = sequence_length
        self.sequence_position = sequence_position

        # Get sequence names for the given split
        if split == "test":
            split_dir = "test"
        else:
            split_dir = "training"

        split_file = THIS_DIR / "Sintel_val.txt"
        with open(split_file, "r") as f:
            val_seqs = f.read().strip().splitlines()

        sequence_names = sorted(
            [p.stem for p in (Path(root_dir) / split_dir / "clean").glob("*")]
        )
        if split == "train" or split == "val":
            if split == "train":
                sequence_names = [s for s in sequence_names if s not in val_seqs]
            else:
                sequence_names = val_seqs

        # Read paths from disk
        for passd in pass_names:
            for seq_name in sequence_names:
                image_paths = sorted(
                    (
                        Path(self.root_dir_3DCC)
                        / split_dir
                        / passd
                        / seq_name
                        / tdcc_corruption
                        / str(tdcc_intensity)
                    ).glob("*.png")
                )
                image_paths = self._extend_paths_list(
                    image_paths, sequence_length, sequence_position
                )
                flow_paths = []
                occ_paths = []
                if split != "test":
                    flow_paths = sorted(
                        (Path(self.root_dir) / split_dir / "flow" / seq_name).glob(
                            "*.flo"
                        )
                    )
                    flow_paths = self._extend_paths_list(
                        flow_paths, sequence_length, sequence_position
                    )

                    assert len(image_paths) - 1 == len(
                        flow_paths
                    ), f"{passd}, {seq_name}: {len(image_paths)-1} vs {len(flow_paths)}"
                    if (Path(self.root_dir) / split_dir / "occlusions").exists():
                        occ_paths = sorted(
                            (
                                Path(self.root_dir)
                                / split_dir
                                / "occlusions"
                                / seq_name
                            ).glob("*.png")
                        )
                        occ_paths = self._extend_paths_list(
                            occ_paths, sequence_length, sequence_position
                        )
                        assert len(occ_paths) == len(flow_paths)
                for i in range(len(image_paths) - self.sequence_length + 1):
                    self.img_paths.append(image_paths[i : i + self.sequence_length])
                    if len(flow_paths) > 0:
                        self.flow_paths.append(
                            flow_paths[i : i + self.sequence_length - 1]
                        )
                    if len(occ_paths) > 0:
                        self.occ_paths.append(
                            occ_paths[i : i + self.sequence_length - 1]
                        )
                    self.metadata.append(
                        {
                            "image_paths": [
                                str(p)
                                for p in image_paths[i : i + self.sequence_length]
                            ],
                            "is_val": seq_name in val_seqs,
                            "misc": seq_name,
                            "is_seq_start": i == 0,
                        }
                    )

        # Sanity check
        if split != "test":
            assert len(self.img_paths) == len(
                self.flow_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"
        if len(self.occ_paths) > 0:
            assert len(self.img_paths) == len(
                self.occ_paths
            ), f"{len(self.img_paths)} vs {len(self.occ_paths)}"

        self._log_status()


class SpringDataset3DCC(BaseFlowDataset):
    # TODO: currently unchanged
    """Handle the Spring dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        split: str = "train",
        side_names: Union[str, List[str]] = "left",
        add_reverse: bool = True,
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_backward: bool = False,
        get_meta: bool = True,
        sequence_length: int = 2,
        sequence_position: str = "first",
        reverse_only: bool = False,
    ) -> None:
        """Initialize SintelDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the MPI Sintel dataset.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval', 'test'}.
        side_names : Union[str, List[str]], default 'left'
             Samples from which side view should be loaded. It can be one of {'left', 'right', ['left', 'right']}.
        add_reverse : bool, default True
            If True, double the number of samples by appending the backward samples as additional samples.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_backward : bool, default True
            Whether to get the backward version of the inputs.
        get_meta : bool, default True
            Whether to get metadata.
        sequence_length : int, default 2
            How many consecutive images are loaded per sample. More than two images can be used for model which exploit more
            temporal information.
        sequence_position : str, default "first"
            Only used when sequence_length > 2.
            Determines the position where the main image frame will be in the sequence. It can one of three values:
            - "first": the main frame will be the first one of the sequence,
            - "middle": the main frame will be in the middle of the sequence (at position sequence_length // 2),
            - "last": the main frame will be the penultimate in the sequence.
        reverse_only : bool, default False
            If True, only uses the backward samples, discarding the forward ones.
        """
        if isinstance(side_names, str):
            side_names = [side_names]
        super().__init__(
            dataset_name="Spring",
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=False,
            get_motion_boundary_mask=False,
            get_backward=get_backward,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.split = split
        self.side_names = side_names
        self.sequence_length = sequence_length
        self.sequence_position = sequence_position

        # Get sequence names for the given split
        if split == "test":
            split_dir = "test"
        else:
            split_dir = "train"

        sequence_names = sorted(
            [p.stem for p in (Path(root_dir) / split_dir).glob("*")]
        )

        if reverse_only:
            directions = [("BW", "FW")]
        else:
            directions = [("FW", "BW")]
            if add_reverse:
                directions.append(("BW", "FW"))

        # Read paths from disk
        for seq_name in sequence_names:
            for side in side_names:
                for direcs in directions:
                    rev = direcs[0] == "BW"
                    image_paths = sorted(
                        (
                            Path(self.root_dir) / split_dir / seq_name / f"frame_{side}"
                        ).glob("*.png"),
                        reverse=rev,
                    )
                    image_paths = self._extend_paths_list(
                        image_paths, sequence_length, sequence_position
                    )
                    flow_paths = []
                    flow_b_paths = []
                    if split != "test":
                        flow_paths = sorted(
                            (
                                Path(self.root_dir)
                                / split_dir
                                / seq_name
                                / f"flow_{direcs[0]}_{side}"
                            ).glob("*.flo5"),
                            reverse=rev,
                        )
                        flow_paths = self._extend_paths_list(
                            flow_paths, sequence_length, sequence_position
                        )
                        assert len(image_paths) - 1 == len(
                            flow_paths
                        ), f"{seq_name}, {side}: {len(image_paths)-1} vs {len(flow_paths)}"
                        if self.get_backward:
                            flow_b_paths = sorted(
                                (
                                    Path(self.root_dir)
                                    / split_dir
                                    / seq_name
                                    / f"flow_{direcs[1]}_{side}"
                                ).glob("*.flo5"),
                                reverse=rev,
                            )
                            flow_b_paths = self._extend_paths_list(
                                flow_b_paths, sequence_length, sequence_position
                            )
                            assert len(image_paths) - 1 == len(
                                flow_paths
                            ), f"{seq_name}, {side}: {len(image_paths)-1} vs {len(flow_paths)}"

                    for i in range(len(image_paths) - self.sequence_length + 1):
                        self.img_paths.append(image_paths[i : i + self.sequence_length])
                        if len(flow_paths) > 0:
                            self.flow_paths.append(
                                flow_paths[i : i + self.sequence_length - 1]
                            )
                        if self.get_backward and len(flow_b_paths) > 0:
                            self.flow_b_paths.append(
                                flow_b_paths[i : i + self.sequence_length - 1]
                            )
                        self.metadata.append(
                            {
                                "image_paths": [
                                    str(p)
                                    for p in image_paths[i : i + self.sequence_length]
                                ],
                                "is_val": False,
                                "misc": seq_name,
                                "is_seq_start": i == 0,
                            }
                        )

        # Sanity check
        if split != "test":
            assert len(self.img_paths) == len(
                self.flow_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"

        self._log_status()


class MiddleburyDataset3DCC(BaseFlowDataset):
    # TODO: currently unchanged
    """Handle the Middlebury dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        split: str = "train",
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_meta: bool = True,
    ) -> None:
        """Initialize MiddleburyDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the Middlebury dataset.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval', 'test'}.
        pass_names : Union[str, List[str]], default 'clean'
            Which passes should be loaded. It can be one of {'clean', 'final', ['clean', 'final']}.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_occlusion_mask : bool, default True
            Whether to get occlusion masks.
        get_meta : bool, default True
            Whether to get metadata.
        """
        super().__init__(
            dataset_name="Middlebury",
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=False,
            get_motion_boundary_mask=False,
            get_backward=False,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.split = split
        self.sequence_length = 2

        # Get sequence names for the given split
        if split == "test":
            split_dir = "eval"
        else:
            split_dir = "other"

        sequence_names = sorted(
            [p.stem for p in (Path(root_dir) / f"{split_dir}-gt-flow").glob("*")]
        )

        # Read paths from disk
        for seq_name in sequence_names:
            image_paths = sorted(
                (Path(self.root_dir) / f"{split_dir}-data" / seq_name).glob("*.png")
            )
            flow_paths = []
            if split != "test":
                flow_paths = sorted(
                    (Path(self.root_dir) / f"{split_dir}-gt-flow" / seq_name).glob(
                        "*.flo"
                    )
                )
                assert len(image_paths) - 1 == len(
                    flow_paths
                ), f"{seq_name}: {len(image_paths)-1} vs {len(flow_paths)}"
            for i in range(len(image_paths) - self.sequence_length + 1):
                self.img_paths.append(image_paths[i : i + self.sequence_length])
                if len(flow_paths) > 0:
                    self.flow_paths.append(flow_paths[i : i + self.sequence_length - 1])
                self.metadata.append(
                    {
                        "image_paths": [
                            str(p) for p in image_paths[i : i + self.sequence_length]
                        ],
                        "is_val": False,
                        "misc": seq_name,
                        "is_seq_start": True,
                    }
                )

        # Sanity check
        if split != "test":
            assert len(self.img_paths) == len(
                self.flow_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"

        self._log_status()


class MonkaaDataset3DCC(BaseFlowDataset):
    # TODO: currently unchanged
    """Handle the Monkaa dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        pass_names: Union[str, List[str]] = "clean",
        side_names: Union[str, List[str]] = "left",
        add_reverse: bool = True,
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 1000.0,
        get_valid_mask: bool = True,
        get_backward: bool = True,
        get_meta: bool = True,
        sequence_length: int = 2,
        sequence_position: str = "first",
    ) -> None:
        """Initialize MonkaaDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the Monkaa dataset.
        pass_names : Union[str, List[str]], default 'clean'
            Which passes should be loaded. It can be one of {'clean', 'final', ['clean', 'final']}.
        side_names : Union[str, List[str]], default 'left'
             Samples from which side view should be loaded. It can be one of {'left', 'right', ['left', 'right']}.
        add_reverse : bool, default True
            If True, double the number of samples by appending the backward samples as additional samples.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_backward : bool, default True
            Whether to get the occluded version of the inputs.
        get_meta : bool, default True
            Whether to get metadata.
        sequence_length : int, default 2
            How many consecutive images are loaded per sample. More than two images can be used for model which exploit more
            temporal information.
        sequence_position : str, default "first"
            Only used when sequence_length > 2.
            Determines the position where the main image frame will be in the sequence. It can one of three values:
            - "first": the main frame will be the first one of the sequence,
            - "middle": the main frame will be in the middle of the sequence (at position sequence_length // 2),
            - "last": the main frame will be the penultimate in the sequence.
        """
        super().__init__(
            dataset_name="Monkaa",
            split_name="trainval",
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=False,
            get_motion_boundary_mask=False,
            get_backward=get_backward,
            get_semantic_segmentation_labels=False,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.add_reverse = add_reverse
        self.pass_names = pass_names
        self.sequence_length = sequence_length
        self.sequence_position = sequence_position
        if isinstance(self.pass_names, str):
            self.pass_names = [self.pass_names]
        self.side_names = side_names
        if isinstance(self.side_names, str):
            self.side_names = [self.side_names]

        pass_dirs = [f"frames_{p}pass" for p in self.pass_names]

        directions = [("into_future", "into_past")]
        reverts = [False]
        if self.add_reverse:
            directions.append(("into_past", "into_future"))
            reverts.append(True)

        # Read paths from disk
        for passd in pass_dirs:
            pass_path = Path(self.root_dir) / passd
            for seq_path in pass_path.glob("*"):
                for direcs, rev in zip(directions, reverts):
                    for side in self.side_names:
                        image_paths = sorted(
                            (seq_path / side).glob("*.png"), reverse=rev
                        )
                        image_paths = self._extend_paths_list(
                            image_paths, sequence_length, sequence_position
                        )
                        flow_paths = sorted(
                            (
                                Path(str(seq_path).replace(passd, "optical_flow"))
                                / direcs[0]
                                / side
                            ).glob("*.pfm"),
                            reverse=rev,
                        )
                        flow_paths = self._extend_paths_list(
                            flow_paths, sequence_length, sequence_position
                        )

                        flow_b_paths = []
                        if self.get_backward:
                            flow_b_paths = sorted(
                                (
                                    Path(str(seq_path).replace(passd, "optical_flow"))
                                    / direcs[1]
                                    / side
                                ).glob("*.pfm"),
                                reverse=rev,
                            )
                            flow_b_paths = self._extend_paths_list(
                                flow_b_paths, sequence_length, sequence_position
                            )

                        for i in range(len(image_paths) - self.sequence_length + 1):
                            self.img_paths.append(
                                image_paths[i : i + self.sequence_length]
                            )
                            if len(flow_paths) > 0:
                                self.flow_paths.append(
                                    flow_paths[i : i + self.sequence_length - 1]
                                )
                            self.metadata.append(
                                {
                                    "image_paths": [
                                        str(p)
                                        for p in image_paths[
                                            i : i + self.sequence_length
                                        ]
                                    ],
                                    "is_val": False,
                                    "misc": "",
                                    "is_seq_start": i == 0,
                                }
                            )
                            if self.get_backward:
                                if len(flow_b_paths) > 0:
                                    self.flow_b_paths.append(
                                        flow_b_paths[i + 1 : i + self.sequence_length]
                                    )

        assert len(self.img_paths) == len(
            self.flow_paths
        ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"
        assert len(self.occ_paths) == 0 or len(self.img_paths) == len(
            self.occ_paths
        ), f"{len(self.img_paths)} vs {len(self.occ_paths)}"
        assert len(self.mb_paths) == 0 or len(self.img_paths) == len(
            self.mb_paths
        ), f"{len(self.img_paths)} vs {len(self.mb_paths)}"
        if self.get_backward:
            assert len(self.img_paths) == len(
                self.flow_b_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_b_paths)}"
            assert len(self.occ_b_paths) == 0 or len(self.img_paths) == len(
                self.occ_b_paths
            ), f"{len(self.img_paths)} vs {len(self.occ_b_paths)}"
            assert len(self.mb_b_paths) == 0 or len(self.img_paths) == len(
                self.mb_b_paths
            ), f"{len(self.img_paths)} vs {len(self.mb_b_paths)}"

        self._log_status()
