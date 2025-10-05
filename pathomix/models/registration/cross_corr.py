import torch
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
from kornia.filters import box_blur
from torch.nn.functional import conv2d
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from kornia.enhance import equalize_clahe as equalize
from kornia.geometry.transform import get_rotation_matrix2d, warp_affine

from pathomix.gpu.torch import Manager as GPUManager

DEFAULT_SCALE = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

class XCorr(GPUManager):
    def __init__(self, gpu_id=0, device_type="gpu"):
        super().__init__(gpu_id=gpu_id, device_type=device_type)

    def prepare_for_registration(
        self,
        grayscale_target: torch.Tensor,
        grayscale_source: torch.Tensor,
        source_region_mask: torch.Tensor = None,
    ):
        assert (
            grayscale_source.ndimension() == 4
        ), f"grayscale_source must have 4 dimensions, but got {grayscale_source.ndimension()}"

        if source_region_mask is not None:
            assert (
                grayscale_source.shape == source_region_mask.shape
            ), f"grayscale_source and source_region_mask should have same shape got,\ngrayscale_source:{grayscale_source.shape} vs source_region_mask:{source_region_mask.shape}"

        assert (
            grayscale_target.ndimension() == 4
        ), f"grayscale_target must have 4 dimensions, but got {grayscale_target.ndimension()}"
        assert (
            grayscale_target.shape[1] == 1
        ), f"grayscale_target must be grayscale, but got {grayscale_target.shape}"

        self.padding_logs = {}
        self.padding_logs["source"] = []
        self.padding_logs["target"] = []
        self.origin = np.array([0, 0])

        self.source_region_mask = source_region_mask

        self._grayscale_source = grayscale_source
        self._grayscale_target = grayscale_target

        self._set_template_source()  # Source Template to be created first always
        self._set_template_target()

        self.original_target_shape = self.template_target.shape

        self._center_source = torch.tensor(
            [[self.template_source.shape[3] // 2, self.template_source.shape[2] // 2]],
            dtype=torch.float32,
        )

    def _set_template_source(self):

        equalized = equalize(self._grayscale_source) < 0.2
        template_source = 1 - self._grayscale_source
        template_source[equalized] = 1
        if self.source_region_mask is not None:
            template_source[self.source_region_mask == 0] = 0

        self.template_source = template_source
        self._pad_template_source()

    def _set_template_target(self):
        template_target = self._grayscale_target.clone()
        template_target[template_target < 0.2] = 1
        template_target = box_blur(template_target, kernel_size=3)
        self.template_target = 1 - template_target
        self._pad_template_target()

    def _pad_template_source(self):

        assert (
            len(self.template_source.shape) == 4
        ), f"input tensor must be of shape b*c*h*w"

        padding_list = []

        # Ensure height is even
        if self.template_source.shape[2] % 2 == 1:
            pad = 1
            # Add padding to the bottom (height dimension)
            self.template_source = F.pad(
                self.template_source, (0, 0, 0, pad), mode="constant", value=0
            )
            padding_list.append({"bottom": pad})

        # Ensure width is even
        if self.template_source.shape[3] % 2 == 1:
            pad = 1
            # Add padding to the right (width dimension)
            self.template_source = F.pad(
                self.template_source, (0, pad, 0, 0), mode="constant", value=0
            )
            padding_list.append({"right": pad})

        # Balance padding for height > width
        if self.template_source.shape[2] > self.template_source.shape[3]:
            pad = (self.template_source.shape[2] - self.template_source.shape[3]) // 2
            # Add padding equally to left and right (width dimension)
            self.template_source = F.pad(
                self.template_source, (pad, pad, 0, 0), mode="constant", value=0
            )
            padding_list.append({"left": pad, "right": pad})

        # Balance padding for width > height
        if self.template_source.shape[3] > self.template_source.shape[2]:
            pad = (self.template_source.shape[3] - self.template_source.shape[2]) // 2
            # Add padding equally to top and bottom (height dimension)
            self.template_source = F.pad(
                self.template_source, (0, 0, pad, pad), mode="constant", value=0
            )
            padding_list.append({"top": pad, "bottom": pad})

        # Add additional padding around the image based on the largest dimension
        pad = max(self.template_source.shape[2], self.template_source.shape[3]) // 4
        self.template_source = F.pad(
            self.template_source, (pad, pad, pad, pad), mode="constant", value=0
        )
        padding_list.append({"top": pad, "bottom": pad, "left": pad, "right": pad})

        self.padding_logs["source"].append(padding_list)

    def _pad_template_target(self):
        padding_list = []
        pad = max(self.template_source.shape[2:]) // 2
        self.template_target = F.pad(
            self.template_target, (pad, pad, pad, pad), mode="constant", value=0
        )
        padding_list.append({"top": pad, "bottom": pad, "left": pad, "right": pad})
        self.padding_logs["target"].append(padding_list)

    def _register(self, dataloader, show_progress=True, desc="Registering"):
        stride = dataloader.dataset.stride
        self.template_target = self.template_target.to(self.device)
        if show_progress:
            iterator = tqdm(dataloader, desc=desc)
        else:
            iterator = dataloader

        with torch.inference_mode():
            max_conv_score = float("-inf")
            registered_data = None

            for batch_angles, batch_rotated_template_source in iterator:

                batch_rotated_template_source = batch_rotated_template_source.to(
                    self.device
                )

                batch_conv_score = conv2d(
                    input=self.template_target,
                    weight=batch_rotated_template_source,
                    stride=stride,
                )
                batch_max_conv_score = torch.max(batch_conv_score)
                if batch_max_conv_score > max_conv_score:
                    max_conv_score = batch_max_conv_score
                    max_idx = torch.argmax(batch_conv_score)

                    num_angles, _, conv_scores_height, conv_scores_width = (
                        batch_conv_score.shape
                    )

                    batch_idx = max_idx // (
                        num_angles * conv_scores_height * conv_scores_width
                    )

                    offset_idx = max_idx % (
                        num_angles * conv_scores_height * conv_scores_width
                    )

                    angle_idx = offset_idx // (conv_scores_height * conv_scores_width)
                    spatial_idx = offset_idx % (conv_scores_height * conv_scores_width)

                    x_idx = spatial_idx // conv_scores_width
                    y_idx = spatial_idx % conv_scores_width

                    x_new = stride * x_idx.item()
                    y_new = stride * y_idx.item()

                    angle = batch_angles[angle_idx]

                    registered_data = (x_new, y_new, angle.item())

                del batch_rotated_template_source

        self.template_target = self.template_target.to("cpu")

        return registered_data

class RegistrationDataset(BaseDataset):
    def __init__(
        self,
        angles,
        stride,
        center_source,
        template_source,
        scale=DEFAULT_SCALE,
    ):
        self.scale = scale
        self.angles = angles
        self.stride = stride
        self.center_source = center_source
        self.template_source = template_source

    def __len__(self):
        return len(self.angles)

    def __getitem__(self, idx):
        angle = self.angles[idx]
        rotated_template_source = self._get_rotated_template_source(angle=angle)

        return angle, rotated_template_source

    def _get_rotated_template_source(self, angle):
        rotation_matrix = get_rotation_matrix2d(
            self.center_source, torch.tensor([angle], dtype=torch.float32), self.scale
        )
        rotated_template_source = warp_affine(
            self.template_source,
            rotation_matrix,
            (self.template_source.shape[3], self.template_source.shape[2]),
        ).squeeze(0)

        return rotated_template_source

