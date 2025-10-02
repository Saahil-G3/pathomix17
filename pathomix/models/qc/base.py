import gc
import cv2
import h5py
import json
import hashlib
import numpy as np
from tqdm.auto import tqdm
from shapely.affinity import scale
from collections import defaultdict
from abc import ABC, abstractmethod
from shapely.affinity import translate
from shapely.geometry import Polygon, MultiPolygon

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from torch.utils.data import Dataset as BaseDataset


from fileverse.formats.pickle import BasePickle
from pathomix.models.base import Base as PathomixBase
from pathomix.geometry.tools.cv import get_shapely_poly

base_pickle = BasePickle()


def close_open_h5_files():
    """Force close any open HDF5 files"""
    for obj in gc.get_objects():
        try:
            if isinstance(obj, h5py.File):
                obj.close()
        except:
            pass


def get_patching_uid(params: dict) -> str:
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:8]


class Base(PathomixBase, ABC):
    def __init__(self, gpu_id=0, device_type="gpu"):
        super().__init__(gpu_id=gpu_id, device_type=device_type)
        self._set_model_specific_params()
        self._set_model_class()
        self._set_paths_dirs()

        self._model_purpose = "qc"

    def _set_paths_dirs(self):
        self.dirs = {}
        # self.dirs['base'] = Path(base_dir)
        # self.dirs['base'].mkdir(exist_ok=True, parents=True)

        # self.dirs['model'] = self.dirs['base']/self._model_name
        # self.dirs['model'].mkdir(exist_ok=True, parents=True)

        self.paths = {}

    def _update_wsi_paths_dirs(self, wsi, extraction_dict, replace):

        uid = get_patching_uid(extraction_dict)

        if self._model_name not in wsi.dirs["inference"] or replace:

            wsi.dirs["inference"][self._model_name] = {}
            wsi.dirs["inference"][self._model_name]["main"] = (
                wsi.dirs["inference"]["main"] / self._model_purpose / self._model_name
            )
            wsi.dirs["inference"][self._model_name]["main"].mkdir(
                exist_ok=True, parents=True
            )

            wsi.dirs["logits"][
                self._model_name
            ] = {}  # / f"{self._model_name}_{uid}.h5"
            wsi.dirs["logits"][self._model_name]["main"] = (
                wsi.dirs["logits"]["main"] / self._model_name
            )
            wsi.dirs["logits"][self._model_name]["main"].mkdir(
                exist_ok=True, parents=True
            )

            wsi.paths["inference"][self._model_name] = {}

        if uid not in wsi.dirs["inference"][self._model_name] or replace:
            wsi.dirs["inference"][self._model_name][uid] = (
                wsi.dirs["inference"][self._model_name]["main"] / uid
            )
            wsi.dirs["inference"][self._model_name][uid].mkdir(
                exist_ok=True, parents=True
            )

            wsi.paths["inference"][self._model_name][uid] = {}

            wsi.paths["inference"][self._model_name][uid]["logits"] = (
                wsi.dirs["logits"][self._model_name]["main"]
                / f"{self._model_name}_{uid}.h5"
            )

            wsi.paths["inference"][self._model_name][uid]["predictions"] = (
                wsi.dirs["inference"][self._model_name][uid] / "predictions.h5"
            )

            wsi.paths["inference"][self._model_name][uid]["patchify_params"] = (
                wsi.dirs["inference"][self._model_name][uid] / "patchify_params.pkl"
            )

            wsi.paths["inference"][self._model_name][uid]["wkt"] = (
                wsi.dirs["inference"][self._model_name][uid] / "wkt.json"
            )

            wsi.save_metadata(replace=replace)

        return uid

    def infer(
        self,
        wsi,
        patch_size,
        overlap,
        context,
        save_logits=False,
        compression="gzip",
        compression_opts=0,
        replace=False,
        **kwargs,
    ):

        dataloader = wsi.get_default_dataloader(
            target_mpp=self._mpp,
            patch_size=patch_size,
            overlap=overlap,
            context=context,
            **kwargs,
        )

        # if self._model_name not in wsi.dirs["inference"]:
        uid = self._update_wsi_paths_dirs(
            wsi=wsi,
            extraction_dict=dataloader.dataset.patchify_params["extraction"],
            replace=replace,
        )

        base_pickle.save(
            data=dataloader.dataset.patchify_params,
            path=wsi.paths["inference"][self._model_name][uid]["patchify_params"],
            replace=False,
        )

        logits_shape = (
            len(dataloader.dataset),
            self._args["classes"],
            dataloader.dataset.extraction_dict["extraction_dims"][0],
            dataloader.dataset.extraction_dict["extraction_dims"][1],
        )

        predictions_shape = (
            len(dataloader.dataset),
            dataloader.dataset.extraction_dict["extraction_dims"][0],
            dataloader.dataset.extraction_dict["extraction_dims"][1],
        )

        coordinates_shape = (len(dataloader.dataset), 2)

        if save_logits:
            f_logits = h5py.File(
                wsi.paths["inference"][self._model_name][uid]["logits"], "w"
            )

            logits_h5py = f_logits.create_dataset(
                "logits",
                shape=logits_shape,
                dtype="float32",
                compression=compression,
                compression_opts=compression_opts if compression == "gzip" else None,
                chunks=True if compression else None,
            )

        f_predictions = h5py.File(
            wsi.paths["inference"][self._model_name][uid]["predictions"], "w"
        )

        coords_h5py = f_predictions.create_dataset(
            "coordinates",
            shape=coordinates_shape,
            dtype="int64",
            compression=compression,
            compression_opts=compression_opts if compression == "gzip" else None,
            chunks=True if compression else None,
        )

        patch_predictions_h5py = f_predictions.create_dataset(
            "patch_predictions",
            shape=predictions_shape,
            dtype="uint8",
            compression=compression,
            compression_opts=compression_opts if compression == "gzip" else None,
            chunks=True if compression else None,
        )

        self.model.eval()
        with torch.inference_mode():
            start = 0
            for batch_coordinates, patches in tqdm(dataloader):
                patches = patches.to(self.device) - 0.5
                logits = self.model(patches)
                blur = F.gaussian_blur(logits, kernel_size=self._blur_ksize, sigma=5)
                predictions = torch.argmax(blur, dim=1).float()

                # predictions = torch.argmax(logits, dim=1).float()
                # predictions = kornia.filters.median_blur(predictions.unsqueeze(1), kernel_size=self._med_blur_ksize)

                end = start + logits.shape[0]
                if save_logits:
                    logits = logits.detach().cpu().numpy()
                    logits_h5py[start:end] = logits

                batch_coordinates = torch.stack(batch_coordinates, dim=1)
                batch_coordinates = batch_coordinates.detach().cpu().numpy()
                coords_h5py[start:end] = batch_coordinates

                predictions = predictions.to(torch.uint8).detach().cpu().numpy()
                patch_predictions_h5py[start:end] = predictions

                start = end

            if save_logits:
                f_logits.close()

            f_predictions.close()
        return uid

    def post_process(self, wsi, uid, batch_size=16):

        dataset = PostProcessingDatasetShapely(wsi=wsi, model=self, uid=uid)

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
        )

        polys_dict = defaultdict(list)
        for patch_polys_dict in tqdm(dataloader):
            # polys is a list of dictionaries
            for d in patch_polys_dict:
                for k, v in d.items():
                    polys_dict[k].extend(v)

        wkt_dict = {}
        scale_factor = dataset.source2target
        for k, polygons in polys_dict.items():
            if polygons:
                scaled_poly = scale(
                    MultiPolygon(polygons).buffer(0),
                    xfact=scale_factor,
                    yfact=scale_factor,
                    origin=(0, 0),
                )
                wkt_dict[k] = scaled_poly.wkt
        # polys_dict = dict(polys_dict)

        # mask_dict = {}
        # polys = []
        # for k, v in polys_dict.items():
        #     polys.extend(v)
        #     mpoly = MultiPolygon(v).buffer(0)
        #     mask_dict[k] = mpoly

        # wkt_dict = {}

        # for k, v in mask_dict.items():
        #     scaled_poly = scale(
        #         v,
        #         xfact=dataset.source2target,
        #         yfact=dataset.source2target,
        #         origin=(0, 0),
        #     )
        #     wkt_dict[k] = scaled_poly.wkt
        json_path = wsi.paths["inference"][self._model_name][uid]["wkt"]
        with open(json_path, "w") as f:
            json.dump(wkt_dict, f, indent=2)

    def get_wkt_dict(self, wsi, uid):
        with open(wsi.paths["inference"][self._model_name][uid]["wkt"], "r") as f:
            wkt_dict = json.load(f)
        return wkt_dict

    @abstractmethod
    def _set_model_specific_params(self):
        pass

    @abstractmethod
    def _set_model_class(self):
        pass


class PostProcessingDatasetShapely(BaseDataset):
    def __init__(self, wsi, model, uid):
        super().__init__()
        self.wsi = wsi
        self.model_name = model._model_name
        self.model_class_map = model._class_map
        self.f_predictions = h5py.File(
            self.wsi.paths["inference"][self.model_name][uid]["predictions"], "r"
        )
        self.coordinates = self.f_predictions["coordinates"]
        self.patch_predictions = self.f_predictions["patch_predictions"]

        self.patchify_params = base_pickle.load(
            path=self.wsi.paths["inference"][self.model_name][uid]["patchify_params"]
        )

        self.source2target = self.patchify_params["factor"]["source2target"]
        self.context = self.patchify_params["extraction"]["context"]
        self.overlap = self.patchify_params["extraction"]["overlap"]

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        x_context, y_context = self.context
        x_overlap, y_overlap = self.overlap
        coordinate = self.coordinates[idx] / self.source2target
        patch_prediction = self.patch_predictions[idx]
        prediction = patch_prediction[y_context:-y_context, x_context:-x_context]
        # prediction =  prediction[y_overlap//4:-y_overlap//4, x_overlap//4:-x_overlap//4]

        keys_to_remove = ["bg"]
        patch_polys_dict = {
            k: [] for k, v in self.model_class_map.items() if k not in keys_to_remove
        }
        patch_polys_dict["combined"] = {}
        x_off, y_off = coordinate[0], coordinate[1]

        for class_name, class_idx in self.model_class_map.items():
            if class_name == "bg":
                continue
            class_mask = (prediction == class_idx).astype(np.uint8)
            contours, hierarchy = cv2.findContours(
                class_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) > 0:
                polys = get_shapely_poly(contours, hierarchy)
                polys = [translate(poly, xoff=x_off, yoff=y_off) for poly in polys]

                patch_polys_dict[class_name].extend(polys)

        contours, hierarchy = cv2.findContours(
            prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 0:
            polys = get_shapely_poly(contours, hierarchy)
            patch_polys_dict["combined"] = [
                translate(poly, xoff=x_off, yoff=y_off) for poly in polys
            ]

        return patch_polys_dict
