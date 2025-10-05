from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset as BaseDataset

pil_to_tensor = ToTensor()


class InferenceDataset(BaseDataset):
    def __init__(self, wsi, target_mpp, patch_size, overlap, context, base_dir=None):
        super().__init__()

        self.wsi = wsi
        self.patchify_params = self.wsi.get_patchify_params(
            target_mpp=target_mpp,
            patch_size=patch_size,
            overlap=overlap,
            context=context,
        )
        self.extraction_dict = self.patchify_params["extraction"]
        self.factor_dict = self.patchify_params["factor"]
        self.level_dict = self.patchify_params["level"]

        self.coordinates = self.wsi.get_patchify_coordinates(
            patchify_params=self.patchify_params
        )

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        coordinate = self.coordinates[idx]
        region = self.wsi.get_region_for_dataloader(
            coordinate=coordinate, patchify_params=self.patchify_params
        )
        region = region.resize(
            size=self.extraction_dict["extraction_dims"], resample=Image.BICUBIC
        )
        region = pil_to_tensor(region)
        # region = resize(region, self.params["extraction_dims"])
        return (coordinate, region)
