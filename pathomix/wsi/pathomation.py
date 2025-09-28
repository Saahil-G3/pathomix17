"""
for documentaion refer to -
https://docs.pathomation.com/sdk/pma.python.documentation/pma_python.html
"""

import numpy as np
from pma_python import core

from base import Base
from fileverse.logger import Logger

logger = Logger(
    name="PathomationWSI",
    log_folder="logs/pathomation",
    log_to_txt=True,
    log_to_console=False,
).get_logger()


class Pathomation(Base):
    def __init__(self, wsi_path, sessionID, tissue_geom=None):
        super().__init__(wsi_path=wsi_path, tissue_geom=tissue_geom)
        self._slideRef = wsi_path
        self.sessionID = sessionID

        self._configure()
        self._set_level_mpp_dict()

        logger.info(
            f"Initiated session with sessionID: {self.sessionID} for WSI at {self._wsi_path}."
        )

    def get_thumbnail_at_mpp(self, target_mpp=50):
        factor = self.get_factor_for_mpp(target_mpp=target_mpp)
        dims = (int(self.dims[0] // factor), int(self.dims[1] // factor))
        return self.get_thumbnail_at_dims(dims)

    def get_thumbnail_at_dims(self, dims):
        thumbnail = core.get_thumbnail_image(
            self._slideRef,
            width=dims[0],
            height=dims[1],
            sessionID=self.sessionID,
            verify=True,
        )
        return thumbnail

    def get_region_native(self, x: int, y: int, w: int, h: int, scale: float = 1):
        region = core.get_region(
            self._slideRef,
            x=x,
            y=y,
            width=w,
            height=h,
            scale=scale,
            sessionID=self.sessionID,
        )
        return region

    def get_region_for_slicer(self, coordinate, patchify_params):

        x, y = coordinate

        factor_dict = patchify_params["factor"]
        extraction_dict = patchify_params["extraction"]

        context = extraction_dict["context"]
        source2target = factor_dict["source2target"]

        if context is not None:
            y_context, x_context = context
            y_context_scaled = int(y_context * source2target)
            x_context_scaled = int(x_context * source2target)

            x = x - x_context_scaled
            y = y - y_context_scaled

        y_extraction, x_extraction = extraction_dict["extraction_dims"]
        y_extraction_scaled = int(y_extraction * source2target)
        x_extraction_scaled = int(x_extraction * source2target)

        region = self.get_region_native(
            x=x,
            y=y,
            w=x_extraction_scaled,
            h=y_extraction_scaled,
            scale=1 / source2target,
        )

        return region

    def _configure(self):
        self.dims = core.get_pixel_dimensions(
            self._slideRef, zoomlevel=None, sessionID=self.sessionID
        )
        self._mpp_x, self._mpp_y = core.get_pixels_per_micrometer(
            self._slideRef, zoomlevel=None, sessionID=self.sessionID
        )
        self.zoomlevels = core.get_zoomlevels_list(
            self._slideRef, sessionID=self.sessionID, min_number_of_tiles=0
        )
        if self._mpp_x != self._mpp_y:
            logger.warning("mpp_x is not equal to mpp_y.")
        self.mpp = self._mpp_x
        self.level_count = len(self.zoomlevels)

    def _set_level_mpp_dict(self):
        level_mpp_dict = {}
        for level in self.zoomlevels:
            temp_dict = {}
            mpp_x, mpp_y = core.get_pixels_per_micrometer(
                self._slideRef, zoomlevel=level, sessionID=self.sessionID
            )

            temp_dict["level"] = level
            temp_dict["mpp"] = mpp_x
            # factor to go from original mpp to mpp_x
            temp_dict["factor"] = self.get_factor_for_mpp(temp_dict["mpp"])
            temp_dict["dims"] = (
                int(self.dims[0] // temp_dict["factor"]),
                int(self.dims[1] // temp_dict["factor"]),
            )
            level_mpp_dict[self.level_count - level - 1] = temp_dict
            if mpp_x != mpp_y:
                logger.warning(f"mpp_x is not equal to mpp_y at level {level}")

        self.level_mpp_dict = level_mpp_dict
