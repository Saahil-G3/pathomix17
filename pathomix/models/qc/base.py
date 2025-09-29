import h5py
from tqdm.auto import tqdm
from abc import ABC, abstractmethod

import torch
import torchvision.transforms.functional as F

from fileverse.formats.pickle import BasePickle
from pathomix.models.base import Base as PathomixBase

base_pickle = BasePickle()


class Base(PathomixBase, ABC):
    def __init__(self, base_dir, gpu_id=0, device_type="gpu"):
        super().__init__(gpu_id=gpu_id, device_type=device_type)
        self._set_model_specific_params()
        self._set_model_class()
        self._set_paths_dirs(base_dir=base_dir)

        self._model_purpose = "qc"

    def _set_paths_dirs(self, base_dir):
        self.dirs = {}
        # self.dirs['base'] = Path(base_dir)
        # self.dirs['base'].mkdir(exist_ok=True, parents=True)

        # self.dirs['model'] = self.dirs['base']/self._model_name
        # self.dirs['model'].mkdir(exist_ok=True, parents=True)

        self.paths = {}

    def _update_wsi_paths_dirs(self, wsi):
        wsi.dirs["inference"][self._model_name] = (
            wsi.dirs["inference"]["main"] / self._model_purpose / self._model_name
        )
        wsi.dirs["inference"][self._model_name].mkdir(exist_ok=True, parents=True)

        wsi.paths["inference"][self._model_name] = {}
        wsi.paths["inference"][self._model_name]["logits"] = (
            wsi.dirs["logits"] / f"{self._model_name}.h5"
        )
        # wsi.paths["inference"][self._model_name]["coordinates"] = (
        #     wsi.dirs["inference"][self._model_name] / "coordinates.h5"
        # )
        wsi.paths["inference"][self._model_name]["predictions"] = (
            wsi.dirs["inference"][self._model_name] / "predictions.h5"
        )

        wsi.paths["inference"][self._model_name]["patchify_params"] = (
            wsi.dirs["inference"][self._model_name] / "patchify_params.pkl"
        )
        wsi.save_metadata(replace=True)

    def infer(
        self,
        wsi,
        patch_size,
        overlap,
        context,
        save_logits=False,
        compression="gzip",
        compression_opts=0,
        **kwargs
    ):
    
        dataloader = wsi.get_default_dataloader(
            target_mpp=self._mpp,
            patch_size=patch_size,
            overlap=overlap,
            context=context,
            **kwargs
        )
    
        if self._model_name not in wsi.dirs["inference"]:
            self._update_wsi_paths_dirs(wsi=wsi)

        base_pickle.save(
            data=dataloader.dataset.patchify_params,
            path=wsi.paths["inference"][self._model_name]["patchify_params"],
            replace=False,
        )
    
        logits_shape = (
            len(dataloader.dataset), 
            self._args['classes'],
            dataloader.dataset.extraction_dict['extraction_dims'][0], 
            dataloader.dataset.extraction_dict['extraction_dims'][1]
        )
        
        predictions_shape = (
            len(dataloader.dataset),
            dataloader.dataset.extraction_dict['extraction_dims'][0], 
            dataloader.dataset.extraction_dict['extraction_dims'][1]
        )
        
        coordinates_shape = (
            len(dataloader.dataset),
            2
        )
        
        if save_logits:
            f_logits = h5py.File(wsi.paths["inference"][self._model_name]["logits"], "w")
            
            logits_h5py = f_logits.create_dataset(
                "logits",
                shape=logits_shape,
                dtype="float32",
                compression=compression,
                compression_opts=compression_opts if compression == "gzip" else None,
                chunks=True if compression else None,
            )
        
        
        f_predictions = h5py.File(wsi.paths['inference'][self._model_name]['predictions'], "w")
        
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

        self.model.eval();
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

    @abstractmethod
    def _set_model_specific_params(self):
        pass

    @abstractmethod
    def _set_model_class(self):
        pass
