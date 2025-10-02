from pathlib import Path

script_dir = Path(__file__).resolve().parent
toolkit_weights_dir = Path("pathomix/models/weights/qc/qc_models_v1")
WEIGHTS_DIR = None
for parent in script_dir.parents[::-1]:
    #print(parent)
    if "pathomix" in str(parent):
        WEIGHTS_DIR = parent / toolkit_weights_dir
        break
        
if WEIGHTS_DIR is None or not WEIGHTS_DIR.exists():
    raise FileNotFoundError("Could not find the 'toolkit_x' directory or the weights path does not exist.")

from .base import Base as QCBase

class TissueDetectionV1(QCBase):
    def __init__(
        self,
        gpu_id=0,
        device_type="gpu",
    ):
        super().__init__(gpu_id=gpu_id, device_type=device_type)
        #self.WEIGHTS_DIR = WEIGHTS_DIR

    def _set_model_specific_params(self) -> None:
        self._detects_tissue = True
        self._model_name = "tissue_detection_v1"
        self._class_map = {"bg": 0, "adipose": 1, "non_adipose": 2}
        self._mpp = 4
        self._blur_ksize = 25
        self._model_class = "smp"

        self.state_dict_path = WEIGHTS_DIR / "tissue_model_v1.pt"

    def _set_model_class(self):
        self._model_class = "smp"
        self._architecture = "UnetPlusPlus"
        self._args = {
            "encoder_name": "resnet18",
            "encoder_weights": "imagenet",
            "in_channels": 3,
            "classes": 3,
        }