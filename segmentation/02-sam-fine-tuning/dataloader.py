import os
import numpy as np
import torch
from torch.utils.data import Dataset


join = os.path.join


class NpzDataset(Dataset):
    """create a dataset class to load npz data and return back image embeddings and ground truth"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.npz_files = sorted(os.listdir(self.data_path))
        self.npz_data = [np.load(join(data_path, f)) for f in self.npz_files]

        # this implementation is ugly but it works (and is also fast for feeding data to GPU) if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = np.vstack([data["gts"] for data in self.npz_data])
        self.img_embeddings = np.vstack(
            [data["img_embeddings"] for data in self.npz_data]
        )
        print(f"{self.img_embeddings.shape=}, {self.ori_gts.shape=}")

    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        xy_center_points = np.array(get_center_points(bboxes))

        # convert img embedding, mask, bounding box, center point to torch tensor
        return (
            torch.tensor(img_embed).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            torch.tensor(xy_center_points).float(),
        )


def get_center_points(bbox: np.ndarray):
    """get the center points(xy format) of the bounding box"""
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return [x_center, y_center]


def get_bbox_from_mask(mask):
    """Returns a bounding box from a mask"""
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))

    return np.array([x_min, y_min, x_max, y_max])


def get_center_points_from_mask(mask):
    """Returns a center point from a mask"""
    bbox = get_bbox_from_mask(mask)
    return get_center_points(bbox)