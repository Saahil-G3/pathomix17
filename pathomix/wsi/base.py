import numpy as np
from abc import ABC
from pathlib import Path
from shapely.prepared import prep as prep_geom_for_query

from pathomix.geometry.tools.shapely import get_box
from fileverse.formats.pickle import BasePickle
base_pickle = BasePickle()

class Base(ABC):
    def __init__(self, wsi_path: Path, base_dir, tissue_geom=None):
        super().__init__()
        self.tissue_geom = tissue_geom
        self._wsi_path = Path(wsi_path)
        self.name = Path(self._wsi_path.name)
        self.stem = Path(self._wsi_path.stem)
        self.base_dir = base_dir

        self._load_metadata()

    def _load_metadata(self):
        base_dir = Path(self.base_dir) if self.base_dir is not None else Path()
        metadata_path = Path('pathomix')/base_dir/f"metadata/{self.stem}"/f'{self.stem}.pkl'
        if metadata_path.exists():
            metadata = base_pickle.load(path = metadata_path)
            self.dirs = metadata["dirs"]
            self.paths = metadata["paths"]
            self.logs = metadata["logs"]
        else:
            self._set_dirs_paths()
            self.save_metadata(replace=True)

    def _set_dirs_paths(self):
        self.dirs = {}

        if self.base_dir is not None:
            self.dirs["main"] = Path(f"pathomix/{self.base_dir}")
        else:
            self.dirs["main"] = Path("pathomix")

        self.dirs["metadata"] = self.dirs["main"] / f"metadata/{self.stem}"
        self.dirs["metadata"].mkdir(exist_ok=True, parents=True)

        self.dirs["inference"] = {}
        self.dirs["inference"]["main"] = self.dirs["main"] / f"inference/{self.stem}"
        self.dirs["inference"]["main"].mkdir(exist_ok=True, parents=True)

        self.dirs['logits'] = {}
        self.dirs['logits']["main"] = self.dirs['main']/f'logits/{self.stem}'
        self.dirs['logits']["main"].mkdir(exist_ok=True, parents=True)

        self.paths = {}
        self.paths["inference"] = {}
        self.paths['metadata'] = self.dirs['metadata']/f'{self.stem}.pkl'

        self.logs = {}
    
    def save_metadata(self, replace):
        metadata = {}
        metadata['dirs'] = self.dirs
        metadata['paths'] = self.paths
        metadata['logs'] = self.logs
        
        base_pickle.save(data=metadata, path=self.paths['metadata'], replace=replace)

    def get_dims_for_mpp(self, target_mpp):
        scale, rescale = self.get_scale_rescale_pair(target_mpp)
        scaled_dims = self.get_dims_for_scale(scale)
        return scaled_dims

    def get_dims_for_scale(self, scale):
        return (
            int((np.array(self.dims) * scale)[0]),
            int((np.array(self.dims) * scale)[1]),
        )

    def get_factor_for_mpp(self, target_mpp, source_mpp=None):
        if source_mpp is None:
            factor = target_mpp / self.mpp
        else:
            factor = target_mpp / source_mpp
        return factor

    def get_scale_rescale_pair(self, target_mpp):
        rescale = self.get_factor_for_mpp(target_mpp)
        scale = 1 / rescale
        return scale, rescale

    def get_level_for_downsample(self, factor):
        for key, value in self.level_mpp_dict.items():
            if value["factor"] < factor:
                break
        return key

    def get_patchify_params(self, target_mpp, patch_size, overlap, context):

        _patch_size_ = _validate_and_convert_to_tuple(patch_size, "patch_size")
        _overlap_ = _validate_and_convert_to_tuple(overlap, "overlap")
        if context is not None:
            _context_ = _validate_and_convert_to_tuple(context, "context")
        else:
            _context_ = (0,0)

        stride = (_patch_size_[0] - _overlap_[0], _patch_size_[1] - _overlap_[1])

        if _context_ is not None:
            extraction_dims = (
                _patch_size_[0] + 2 * _context_[0],
                _patch_size_[1] + 2 * _context_[1],
            )
        else:
            extraction_dims = _patch_size_

        factor_source2target = self.get_factor_for_mpp(target_mpp=target_mpp)
        level = self.get_level_for_downsample(factor=factor_source2target)
        level_dims = self.level_mpp_dict[level]["dims"]
        level_mpp = self.level_mpp_dict[level]["mpp"]
        factor_source2level = self.level_mpp_dict[level]["factor"]
        factor_level2target = self.get_factor_for_mpp(
            target_mpp=target_mpp, source_mpp=level_mpp
        )

        extraction_dict = {}
        extraction_dict["extraction_dims"] = extraction_dims
        extraction_dict["patch_size"] = _patch_size_
        extraction_dict["overlap"] = _overlap_
        extraction_dict["context"] = _context_
        extraction_dict["stride"] = stride

        factor_dict = {}
        factor_dict["source2target"] = factor_source2target
        factor_dict["source2level"] = factor_source2level
        factor_dict["level2target"] = factor_level2target

        level_dict = {}
        level_dict["level"] = level
        level_dict["dims"] = level_dims
        level_dict["mpp"] = level_mpp

        params = {}
        params["extraction"] = extraction_dict
        params["factor"] = factor_dict
        params["level"] = level_dict

        return params

    def get_patchify_coordinates(self, patchify_params):

        factor_dict = patchify_params["factor"]
        extraction_dict = patchify_params["extraction"]

        # y_patch_size, x_patch_size = extraction_dict['patch_size']
        x_stride, y_stride = extraction_dict["stride"]
        x_extraction, y_extraction = extraction_dict["extraction_dims"]

        source2target = factor_dict["source2target"]

        
        x_extraction_scaled = int(x_extraction * source2target)
        y_extraction_scaled = int(y_extraction * source2target)

        
        x_stride_scaled = int(x_stride * source2target)
        y_stride_scaled = int(y_stride * source2target)

        x_max, y_max = self.dims

        x_coords = np.arange(0, x_max, x_stride_scaled)
        x_coords = np.where(
            x_coords + x_extraction_scaled > x_max,
            x_max - x_extraction_scaled,
            x_coords,
        )

        y_coords = np.arange(0, y_max, y_stride_scaled)
        y_coords = np.where(
            y_coords + y_extraction_scaled > y_max,
            y_max - y_extraction_scaled,
            y_coords,
        )

        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        coordinates = list(zip(X.ravel(), Y.ravel()))

        return coordinates

    def get_filtered_coordinates(self, patchify_params):
        if self.tissue_geom is None:
            logger.warning(f"No tissue_geom found for the wsi object.")
            return
            
        coordinates = self.get_patchify_coordinates(
            patchify_params=patchify_params
        )
    
        context = patchify_params['extraction']["context"]
        source2target = patchify_params["factor"]["source2target"]
        x_extraction, y_extraction = patchify_params['extraction']["extraction_dims"]
        
        x_extraction_scaled = int(x_extraction * source2target)
        y_extraction_scaled = int(y_extraction * source2target)

        if context is not None:
            x_context_scaled = int(context[0] * source2target)
            y_context_scaled = int(context[1] * source2target)
        else:
            x_context_scaled = y_context_scaled = 0

        
        prepped_tissue_geom = prep_geom_for_query(self.tissue_geom)
        
        contained_coordinates = []
        boundary_coordinates = []
        
        for x, y in coordinates:
            
            x_start = x - x_context_scaled
            y_start = y - y_context_scaled
            
            box = get_box(x_start, y_start, x_extraction_scaled, y_extraction_scaled)
            if prepped_tissue_geom.intersects(box):
                if prepped_tissue_geom.contains(box):
                    contained_coordinates.append((x, y))
                else:
                    boundary_coordinates.append((x, y))
    
        return contained_coordinates, boundary_coordinates

    @staticmethod
    def round_to_nearest_even(x):
        return round(x / 2) * 2


def _validate_and_convert_to_tuple(value, name: str) -> tuple[int, int]:
    """A helper method to validate and convert params to a tuple of two integers."""
    if isinstance(value, int):
        return (value, value)

    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(
                f"'{name}' must be a tuple of length 2, but got length {len(value)}."
            )
        return value

    raise TypeError(
        f"'{name}' must be an int or a tuple, but got {type(value).__name__}."
    )
