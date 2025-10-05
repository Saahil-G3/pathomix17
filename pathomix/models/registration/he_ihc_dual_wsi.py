import torch
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from shapely.affinity import translate, scale

from fileverse.logger import Logger
from .cross_corr import XCorr, RegistrationDataset
from pathomix.geometry.plotting import plot_overlay
from pathomix.geometry.tools.shapely import get_numpy_mask_for_geom

to_tensor = transforms.ToTensor()
logger = Logger(name="he_x_ihc").get_logger()

class HExIHCxDualWSI(XCorr):
    """
    wsi_target: Where you want to reach.
    wsi_source: Source you want to send.
    """

    def __init__(self, wsi_target, wsi_source, geom_source, target_mpp):
        super().__init__()
        self.wsi_target = wsi_target
        self.wsi_source = wsi_source
        self.source_geom = geom_source
        self.target_mpp = target_mpp

        self.wsi_source_name = str(self.wsi_source.name)
        self.wsi_target_name = str(self.wsi_target.name)

        self._update_wsi_paths_dirs()        

    def _update_wsi_paths_dirs(self, replace=False):
        # Source WSI
        self.wsi_source.logs.setdefault("he_x_ihc", {})
        self.wsi_source.logs["he_x_ihc"].setdefault(self.wsi_target_name, {})
        self.wsi_source.logs["he_x_ihc"][self.wsi_target_name].setdefault(
            "metadata_saved", False
        )

        self.wsi_source.dirs["inference"].setdefault("he_x_ihc", {})
        self.wsi_source.dirs["inference"]["he_x_ihc"].setdefault(
            "main", self.wsi_source.dirs["inference"]["main"] / "he_x_ihc"
        )
        self.wsi_source.dirs["inference"]["he_x_ihc"]["main"].mkdir(
            exist_ok=True, parents=True
        )

        self.wsi_source.paths["inference"].setdefault("he_x_ihc", {})
        self.wsi_source.paths["inference"]["he_x_ihc"].setdefault(
            self.wsi_target_name, {}
        )

        # Target WSI
        self.wsi_target.logs.setdefault("he_x_ihc", {})
        self.wsi_target.logs["he_x_ihc"].setdefault(self.wsi_source_name, {})
        self.wsi_target.logs["he_x_ihc"][self.wsi_source_name].setdefault(
            "metadata_saved", False
        )
        self.wsi_target.logs["he_x_ihc"][self.wsi_source_name].setdefault(
            "run_complete", False
        )

        self.wsi_target.dirs["inference"].setdefault("he_x_ihc", {})
        self.wsi_target.dirs["inference"]["he_x_ihc"].setdefault(
            "main", self.wsi_target.dirs["inference"]["main"] / "he_x_ihc"
        )
        self.wsi_target.dirs["inference"]["he_x_ihc"]["main"].mkdir(
            exist_ok=True, parents=True
        )

        self.wsi_target.paths["inference"].setdefault("he_x_ihc", {})
        self.wsi_target.paths["inference"]["he_x_ihc"].setdefault(
            self.wsi_source_name, {}
        )

        if (
            not self.wsi_source.logs["he_x_ihc"][self.wsi_target_name]["metadata_saved"]
            or replace
        ):
            self.wsi_source.logs["he_x_ihc"][self.wsi_target_name][
                "metadata_saved"
            ] = True
            self.wsi_source.save_metadata(replace=True)

        if (
            not self.wsi_target.logs["he_x_ihc"][self.wsi_source_name]["metadata_saved"]
            or replace
        ):
            self.wsi_target.logs["he_x_ihc"][self.wsi_source_name][
                "metadata_saved"
            ] = True

            scale_factor_source = self.wsi_source.get_factor_for_mpp(
                target_mpp=self.target_mpp
            )

            scale_factor_target = self.wsi_target.get_factor_for_mpp(
                target_mpp=self.target_mpp
            )

            self.wsi_target.logs["he_x_ihc"][self.wsi_source_name][
                "target_mpp"
            ] = self.target_mpp
            self.wsi_target.logs["he_x_ihc"][self.wsi_source_name][
                "scale_factor_source"
            ] = scale_factor_source
            self.wsi_target.logs["he_x_ihc"][self.wsi_source_name][
                "scale_factor_target"
            ] = scale_factor_target
            self.wsi_target.save_metadata(replace=True)

        self.wsi_target.logs["he_x_ihc"][self.wsi_source_name]["template_corrected"] = False

    def _configure(self):
        logger.info(f"Configuring for registration.")
        scale_factor_source = self.wsi_source.get_factor_for_mpp(
            target_mpp=self.target_mpp
        )
        mask_dims = self.wsi_source.get_dims_for_mpp(self.target_mpp)[::-1]

        source_region_mask = get_numpy_mask_for_geom(
            geom=self.source_geom,
            scale_factor=1 / scale_factor_source,
            mask_dims=mask_dims,
        )

        grayscale_source = self.wsi_source.get_thumbnail_at_dims(
            self.wsi_source.get_dims_for_mpp(self.target_mpp)
        ).convert("L")

        grayscale_target = self.wsi_target.get_thumbnail_at_dims(
            self.wsi_target.get_dims_for_mpp(self.target_mpp)
        ).convert("L")

        grayscale_source = to_tensor(grayscale_source).unsqueeze(0)
        grayscale_target = to_tensor(grayscale_target).unsqueeze(0)
        source_region_mask = to_tensor(source_region_mask).unsqueeze(0)

        self.prepare_for_registration(
            grayscale_target=grayscale_target,
            grayscale_source=grayscale_source,
            source_region_mask=source_region_mask,
        )

        logger.info(f"Configuration complete.")

    def register(self, angles, stride, desc, **kwargs):

        dataset = RegistrationDataset(
            angles=angles,
            stride=stride,
            template_source=self.template_source,
            center_source=self._center_source,
        )
        dataloader = DataLoader(dataset, shuffle=False, **kwargs)

        registered_data = self._register(dataloader=dataloader, desc=desc)
        return registered_data

    def get_quadrant_angles(self, stride=10, astride=10, angles=[45, 135, 225, 315]):
        registered_data = self.register(
            angles=angles,
            stride=stride,
            desc="Finding Quadrant",
            batch_size=4,
            num_workers=4,
        )
        x_new, y_new, angle = registered_data
        angles = self.get_quadrant_range(angle, astride)
        return angles

    def correct_template_target(self, registered_data, pad=None):
        if pad is None:
            pad = max(self.template_source.shape) // 2

        delta_pad = round(pad * 0.15)

        x_new, y_new, angle = registered_data

        self.template_target = self.template_target[
            :,
            :,
            x_new - delta_pad : x_new + delta_pad + (2 * pad),
            y_new - delta_pad : y_new + delta_pad + (2 * pad),
        ]

        self.wsi_target.logs["he_x_ihc"][self.wsi_source_name][
            "template_corrected"
        ] = True

        return pad, delta_pad

    def run(
        self,
        num_workers=4,
        stride_coarse=10,
        stride_fine=1,
        astride_coarse=10,
        astride_fine=1,
        batch_size=16,
        replace=False,
    ):
        if (
            self.wsi_target.logs["he_x_ihc"][self.wsi_source_name]["run_complete"]
            and not replace
        ):
            logger.warning(
                f"he_x_ihc cycle complete for target wsi: {self.wsi_target_name} wrt source wsi: {self.wsi_source_name}. Set replace=True to run again."
            )
            return
            
        self._configure()

        angles_coarse = self.get_quadrant_angles()

        registered_data_coarse = self.register(
            angles=angles_coarse,
            stride=stride_coarse,
            batch_size=batch_size,
            num_workers=num_workers,
            desc="Coarse Registration",
        )

        self.wsi_target.logs["he_x_ihc"][self.wsi_source_name][
            "coarse"
        ] = registered_data_coarse

        pad, delta_pad = self.correct_template_target(
            registered_data=registered_data_coarse
        )

        self.wsi_target.logs["he_x_ihc"][str(self.wsi_source.name)]["pad"] = pad
        self.wsi_target.logs["he_x_ihc"][str(self.wsi_source.name)][
            "delta_pad"
        ] = delta_pad

        _, _, angle_coarse = registered_data_coarse

        angle_strided = astride_coarse * angle_coarse
        angles_fine = torch.arange(
            start=max(0, angle_strided - 10), end=angle_strided + 10, step=astride_fine
        )

        registered_data_fine = self.register(
            angles=angles_fine,
            stride=stride_fine,
            batch_size=batch_size,
            num_workers=num_workers,
            desc="Fine Tuning",
        )

        self.wsi_target.logs["he_x_ihc"][self.wsi_source_name][
            "fine"
        ] = registered_data_fine

        shift_dims = self.get_shift()
        self.wsi_target.logs["he_x_ihc"][self.wsi_source_name]["shift"] = shift_dims

        self.wsi_target.logs["he_x_ihc"][self.wsi_source_name]["run_complete"] = True
        self.wsi_target.save_metadata(replace=True)

        logger.info(
            f"he_x_ihc cycle completed successfully for target wsi: {self.wsi_target_name} wrt source wsi: {self.wsi_source_name}."
        )

    def plot_registration_overlay(self, mask=True):

        if self.wsi_target.logs["he_x_ihc"][self.wsi_source_name]["run_complete"]:
            if not self.wsi_target.logs["he_x_ihc"][self.wsi_source_name][
                "template_corrected"
            ]:
                self._configure()
                _ = self.correct_template_target(
                    registered_data=self.wsi_target.logs["he_x_ihc"][
                        self.wsi_source_name
                    ]["coarse"]
                )

        x_new, y_new, angle = self.wsi_target.logs["he_x_ihc"][self.wsi_source_name][
            "fine"
        ]

        template_he = self.template_source.clone()
        template_ihc = self.template_target.clone()

        template_ihc = template_ihc[
            0,
            0,
            y_new : y_new + template_he.shape[2],
            x_new : x_new + template_he.shape[3],
        ]
        template_he = template_he[0, 0, :, :]

        template_ihc = template_ihc.to("cpu").numpy()
        template_he = template_he.to("cpu").numpy()

        if mask:
            plot_overlay(template_ihc, template_he != 0, axis="on")
        else:
            plot_overlay(template_ihc, template_he, axis="on")

    def get_shift(self):
        source_padding_logs = self.padding_logs["source"]

        x_new, y_new, angle = self.wsi_target.logs["he_x_ihc"][self.wsi_source_name][
            "fine"
        ]

        origin_source = torch.tensor([0, 0])

        for padding_logs in source_padding_logs:
            for padding_log in padding_logs:
                if "top" in padding_log:
                    origin_source[0] = origin_source[0] + padding_log["top"]
                if "left" in padding_log:
                    origin_source[1] = origin_source[1] + padding_log["left"]

        origin_source[0] = origin_source[0] + x_new
        origin_source[1] = origin_source[1] + y_new

        x_new, y_new, _ = self.wsi_target.logs["he_x_ihc"][self.wsi_source_name][
            "coarse"
        ]
        delta_pad = self.wsi_target.logs["he_x_ihc"][self.wsi_source_name]["delta_pad"]
        pad = self.wsi_target.logs["he_x_ihc"][self.wsi_source_name]["pad"]

        new_origin_x = pad - (x_new - delta_pad)
        new_origin_y = pad - (y_new - delta_pad)

        origin_target = torch.tensor([new_origin_x, new_origin_y])

        shift_x = (origin_source[0] - origin_target[0]).item()
        shift_y = (origin_source[1] - origin_target[1]).item()

        return (shift_x, shift_y)

    def get_shifted_geom_source2target(self, geom_source):
        shift = self.wsi_target.logs["he_x_ihc"][self.wsi_source_name]["shift"]
        scale_factor_target = self.wsi_target.logs["he_x_ihc"][self.wsi_source_name][
            "scale_factor_target"
        ]

        shift_x, shift_y = shift
        shift_x = shift_x * scale_factor_target
        shift_y = shift_y * scale_factor_target

        source2target_factor = self.wsi_target.get_factor_for_mpp(
            target_mpp=self.wsi_target.mpp, source_mpp=self.wsi_source.mpp
        )

        geom_target = scale(
            geom=geom_source,
            xfact=source2target_factor,
            yfact=source2target_factor,
            origin=(0, 0),
        )

        geom_target = translate(
            geom_target,
            xoff=shift_y,
            yoff=shift_x,
        )

        return geom_target

    @staticmethod
    def get_quadrant_range(angle, astride):
        quadrants = {
            45: torch.arange(0, 100, step=astride),
            135: torch.arange(90, 190, step=astride),
            225: torch.arange(180, 280, step=astride),
            315: torch.arange(270, 370, step=astride),
        }

        return quadrants[angle]
