"""PyTorch dataset pairing X-ray frames with their binary spatter masks."""

from torch.utils.data import Dataset
import cv2


class SegmentationDataset(Dataset):
    """Image/mask pairs for U-Net training.

    Images are read as 3-channel RGB (OpenCV replicates single-channel frames
    across the three channels); masks are read as single-channel grayscale.
    ``transforms`` is applied to the image and the mask separately.
    """

    def __init__(self, imagePaths, maskPaths, transforms):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.maskPaths[idx], 0)

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return image, mask
