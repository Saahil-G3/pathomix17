import torch
import segmentation_models_pytorch as smp

from fileverse.logger import Logger
from pathomix.gpu.torch import Manager as GPUManager

logger = Logger(name="base_model").get_logger()

class Base(GPUManager):
    def __init__(self, gpu_id=0, device_type="gpu"):
        super().__init__(gpu_id=gpu_id, device_type=device_type)

    def load_model(self):
        if self._model_class == "smp":
            if self._architecture == "UnetPlusPlus":
                self.model = smp.UnetPlusPlus(**self._args)
                state_dict = torch.load(
                    f=self.state_dict_path, map_location=self.device, weights_only=True
                )
                self.model.load_state_dict(state_dict)

                logger.info(f"Model {self._model_name} loaded.")
            else:
                raise ValueError(
                    f"Architecture {self._architecture} from {self._model_class} not defined."
                )

        elif self._model_class == "custom":
            raise ValueError(f"model class custom not defined yet")

        else:
            raise ValueError("Unimplemented model class.")

        
        self.model = self.model.to(self.device)