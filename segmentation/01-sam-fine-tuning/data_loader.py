import numpy as np
from typing import List
from torch.utils.data import Dataset

def get_bounding_box_center(binary_mask):
    """Get bounding box coordinates from binary mask."""
    # get bounding box from mask
    y_indices, x_indices = np.where(binary_mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    # add perturbation to bounding box coordinates
    H, W = binary_mask.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    # Get center of bounding box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    return [center_y, center_x]


class SAMDataset(Dataset):
    def __init__(self, dataset, processor):
        """Init the SAM dataset for pytorch."""
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        binary_mask = np.array(item["label"])

        # get bounding box prompt
        point_prompt = get_bounding_box_center(binary_mask)

        # prepare image and prompt for the model
        inputs = self.processor(image, input_points=[[point_prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = binary_mask

        return inputs