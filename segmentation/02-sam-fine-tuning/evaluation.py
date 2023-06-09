import os
import numpy as np
import torch
from skimage import io
from segment_anything import SamPredictor, sam_model_registry
from utils.loss import compute_dice_coefficient
from dataloader import get_center_points_from_mask

join = os.path.join

# Set global vairiable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "vit_b"
sam_checkpoint = "models/sam_vit_b_01ec64.pth"
medsam_checkpoint = "outputs/sam_model_best.pth"
test_img_path = "data/MedSAMDemo_2D/test/images"
test_gt_path = "data/MedSAMDemo_2D/test/labels"
test_data_names = sorted(os.listdir(test_img_path))

# Load model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
sam_predictor = SamPredictor(sam)

med_sam = sam_model_registry[model_type](checkpoint=medsam_checkpoint).to(device)
med_sam_predictor = SamPredictor(med_sam)

# Validation
total_sam_dice_coef = []
total_med_sam_dice_coef = []
for img_name in test_data_names:
    image_data = io.imread(join(test_img_path, img_name))
    gt_data = io.imread(join(test_gt_path, img_name))
    gt_points = get_center_points_from_mask(gt_data)
    gt_points = np.array([gt_points])
    gt_label = np.array([1])

    # Preprocessing
    lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(
        image_data, 99.5
    )
    preprocessed_image = np.clip(
        image_data, lower_bound, upper_bound
    )  ## Adaptive thresholding
    preprocessed_image = (
        (preprocessed_image - np.min(preprocessed_image))
        / (np.max(preprocessed_image) - np.min(preprocessed_image))
        * 255.0
    )
    preprocessed_image[image_data == 0] = 0
    preprocessed_image = preprocessed_image.astype(np.uint8)
    h, w, _ = preprocessed_image.shape

    # SAM Inference
    sam_predictor.set_image(preprocessed_image)
    sam_seg, _, _ = sam_predictor.predict(
        point_coords=gt_points, point_labels=gt_label, box=None, multimask_output=False
    )
    med_sam_predictor.set_image(preprocessed_image)
    med_sam_seg, _, _ = med_sam_predictor.predict(
        point_coords=gt_points, point_labels=gt_label, box=None, multimask_output=False
    )

    # Validation
    sam_dice_coef = compute_dice_coefficient(gt_data > 0, sam_seg > 0)
    medsam_dice_coef = compute_dice_coefficient(gt_data > 0, med_sam_seg > 0)
    total_sam_dice_coef.append(sam_dice_coef)
    total_med_sam_dice_coef.append(medsam_dice_coef)

print("SAM Dice Coefficient: ", np.mean(total_sam_dice_coef))
print("MEDSAM Dice Coefficient: ", np.mean(total_med_sam_dice_coef))
