import os
import torch
import numpy as np
import cv2
from skimage import io
from segment_anything import SamPredictor, sam_model_registry
from dataloader import get_bbox_from_mask
from utils.loss import compute_dice_coefficient


join = os.path.join

# Set global vairiable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "vit_b"
sam_checkpoint = "models/sam_vit_b_01ec64.pth"
medsam_checkpoint = "outputs/sam_model_best.pth"
test_img_path = "data/MedSAMDemo_2D/test/images"
test_gt_path = "data/MedSAMDemo_2D/test/labels"
test_data_names = sorted(os.listdir(test_img_path))

# Load random image and gt from testset
img_idx = np.random.randint(len(test_data_names))
image_data = io.imread(join(test_img_path, test_data_names[img_idx]))
gt_data = io.imread(join(test_gt_path, test_data_names[img_idx]))
gt_bbox = get_bbox_from_mask(gt_data)

# Load model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
sam_predictor = SamPredictor(sam)

med_sam = sam_model_registry[model_type](checkpoint=medsam_checkpoint).to(device)
med_sam_predictor = SamPredictor(med_sam)

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
    point_coords=None, box=gt_bbox, multimask_output=False
)
med_sam_predictor.set_image(preprocessed_image)
med_sam_seg, _, _ = med_sam_predictor.predict(
    point_coords=None, box=gt_bbox, multimask_output=False
)

# Visualize using contour in original image
## Reshape b h w to h w b
sam_seg = np.transpose(sam_seg, (1, 2, 0))
sam_seg = (sam_seg * 255).astype(np.uint8)
med_sam_seg = np.transpose(med_sam_seg, (1, 2, 0))
med_sam_seg = (med_sam_seg * 255).astype(np.uint8)

## Draw polygon
sam_contour = cv2.findContours(sam_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
sam_contour = sorted(sam_contour, key=cv2.contourArea, reverse=True)
sam_contour = np.array(sam_contour[0]).reshape(-1, 2)
med_sam_contour = cv2.findContours(med_sam_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[
    -2
]
med_sam_contour = sorted(med_sam_contour, key=cv2.contourArea, reverse=True)
med_sam_contour = np.array(med_sam_contour[0]).reshape(-1, 2)
gt_contour = cv2.findContours(gt_data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
gt_contour = sorted(gt_contour, key=cv2.contourArea, reverse=True)
gt_contour = np.array(gt_contour[0]).reshape(-1, 2)

## Draw contour
draw_img = image_data.copy()
contour_img = cv2.drawContours(draw_img, [sam_contour], -1, (255, 255, 0), 2)  ### Cyan
contour_img = cv2.drawContours(
    draw_img, [med_sam_contour], -1, (0, 255, 255), 2
)  ### Yellow
contour_img = cv2.drawContours(
    draw_img, [gt_contour], -1, (255, 255, 255), 2
)  ### White

## Save
cv2.imwrite("outputs/contour.png", contour_img)

# Validation
total_sam_dice_coef = []
total_med_sam_dice_coef = []
for img_name in test_data_names:
    image_data = io.imread(join(test_img_path, img_name))
    gt_data = io.imread(join(test_gt_path, img_name))
    gt_bbox = get_bbox_from_mask(gt_data)

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
        point_coords=None, box=gt_bbox, multimask_output=False
    )
    med_sam_predictor.set_image(preprocessed_image)
    med_sam_seg, _, _ = med_sam_predictor.predict(
        point_coords=None, box=gt_bbox, multimask_output=False
    )

    # Validation
    sam_dice_coef = compute_dice_coefficient(gt_data > 0, sam_seg > 0)
    medsam_dice_coef = compute_dice_coefficient(gt_data > 0, med_sam_seg > 0)
    total_sam_dice_coef.append(sam_dice_coef)
    total_med_sam_dice_coef.append(medsam_dice_coef)

print("SAM Dice Coefficient: ", np.mean(total_sam_dice_coef))
print("MEDSAM Dice Coefficient: ", np.mean(total_med_sam_dice_coef))
