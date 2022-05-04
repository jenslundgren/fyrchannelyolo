import numpy as np
import os
import pandas as pd
import torch
import config

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
] 


# label_dir = "C:/coding/ExJobb/FourChannelYolo/data/labels"
# tensor_dir = "C:/coding/ExJobb/FourChannelYolo/data/rgbDistance"

class DistanceDataset(Dataset):
    def __init__(
        self,
        tensor_dir,
        label_dir,
        anchors,
        image_size=416,
        S = [13, 26, 52],
        C=3
    ):
        self.tensor_dir = tensor_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
    
    def __len__(self):
        return 1
        
    def __getitem__(self, index):
        label_path = self.label_dir + f"/{index+1}.txt"
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), -1, axis=1).tolist()
        tensor_path = self.tensor_dir + f"/{index+1}.pt"
        tensor = torch.load(tensor_path)
        
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                    
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return tensor, tuple(targets)