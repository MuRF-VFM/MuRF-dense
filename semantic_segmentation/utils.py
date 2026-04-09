import numpy as np
import torch
from itertools import product
from typing import Tuple, Optional

import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass


def intersection_union(imPred, imLab, numClass) -> Tuple[
    np.ndarray, np.ndarray]:
    # https://github.com/linusericsson/ssl-transfer/blob/main/semantic-segmentation/mit_semseg/utils.py#L140-L141
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass)
    )

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return area_intersection, area_union


def correct_total(imPred, imLab, numClass) -> Tuple[np.ndarray, np.ndarray]:
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    class_total = np.bincount(imPred.flatten(), minlength=numClass)[1:]
    class_correct = np.bincount(imLab[imPred == imLab].flatten(),
                                minlength=numClass)[
                    1:
                    ]

    return class_correct, class_total


def get_macc_miou(
        preds: np.ndarray | list,
        labels: np.ndarray | list,
        num_classes: int,
) -> dict:
    intersections, unions, corrects, totals = [], [], [], []

    for pred, label in zip(preds, labels):
        i, u = intersection_union(pred, label, num_classes)
        c, t = correct_total(pred, label, num_classes)
        intersections.append(i)
        unions.append(u)
        corrects.append(c)
        totals.append(t)

    return {
        "mean_accuracy": np.nanmean(
            np.sum(corrects, axis=0) / np.sum(totals, axis=0)),
        "mean_iou": np.nanmean(
            np.sum(intersections, axis=0) / np.sum(unions, axis=0), axis=0
        ),
    }


def sliding_window_inference(
        model: torch.nn.Module,
        num_classes: int,
        input_data: torch.Tensor,
        crop_size,
        stride,
        batch_size: int,
        encoder: Optional[torch.nn.Module] = None,
        return_logits: bool = False,
) -> torch.Tensor:
    softmax = torch.nn.Softmax(dim=1)

    _, _, height, width = input_data.size()
    output = torch.zeros((1, num_classes, height, width)).to(input_data.device)
    freq = output.clone()

    xs = np.unique(
        [
            x if x + crop_size[1] <= width else width - crop_size[1]
            for x in range(0, width, stride[1])
        ]
    )
    ys = np.unique(
        [
            y if y + crop_size[0] <= height else height - crop_size[0]
            for y in range(0, height, stride[0])
        ]
    )

    crops = []
    for x, y in product(xs, ys):
        crops.append(
            input_data[:, :, y: y + crop_size[0], x: x + crop_size[1]])

    if return_logits:
        preds = []
        for i in range(0, len(crops), batch_size):  # this should be faster :)
            batch = torch.cat(crops[i:])

            pred = model(batch)
            preds += [softmax(p.unsqueeze(0)) for p in pred]

        for i, (x, y) in enumerate(product(xs, ys)):
            output[:, :, y: y + crop_size[0], x: x + crop_size[1]] += preds[i]
            freq[:, :, y: y + crop_size[0], x: x + crop_size[1]] += 1

        output /= freq

        return output

    else:
        assert encoder is not None, "Encoder is required for patch embeddings"
        embeddings = []
        for i in range(0, len(crops), batch_size):
            batch = torch.cat(crops[i:])

            embedding = encoder.get_patch_embeddings(batch)
            embeddings += [embedding]

        return torch.cat(embeddings).unsqueeze(0)


# Standard Pascal VOC color map (21 classes)
VOC_COLORMAP = np.array([
    [0, 0, 0],  # 0 = background
    [128, 0, 0],  # 1 = aeroplane
    [0, 128, 0],  # 2 = bicycle
    [128, 128, 0],  # 3 = bird
    [0, 0, 128],  # 4 = boat
    [128, 0, 128],  # 5 = bottle
    [0, 128, 128],  # 6 = bus
    [128, 128, 128],  # 7 = car
    [64, 0, 0],  # 8 = cat
    [192, 0, 0],  # 9 = chair
    [64, 128, 0],  # 10 = cow
    [192, 128, 0],  # 11 = diningtable
    [64, 0, 128],  # 12 = dog
    [192, 0, 128],  # 13 = horse
    [64, 128, 128],  # 14 = motorbike
    [0, 255, 255],  # 15 = person
    [0, 64, 0],  # 16 = potted plant
    [128, 64, 0],  # 17 = sheep
    [0, 192, 0],  # 18 = sofa
    [128, 192, 0],  # 19 = train
    [0, 64, 128],  # 20 = tv/monitor
], dtype=np.uint8)

ADE20K_COLORMAP = np.array([
    [0, 0, 0],
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 0, 255],
    [0, 20, 255],
    [2, 255, 255],
    [0, 255, 20],
    [184, 255, 0],
    [0, 255, 204],
    [92, 0, 255],
    [0, 92, 255],
    [0, 255, 184],
    [184, 0, 255],
    [0, 184, 255],
    [92, 255, 0],
    [0, 255, 92],
    [255, 0, 184],
    [255, 0, 92],
    [255, 153, 0],
    [153, 0, 255],
    [0, 153, 255],
    [255, 0, 153],
    [153, 255, 0],
    [0, 255, 153],
    [70, 70, 70],
    [80, 70, 70],
    [90, 70, 70],
    [0, 0, 230],
    [0, 0, 254],
    [0, 40, 40],
    [0, 140, 140],
    [224, 0, 0],
    [244, 0, 0],
    [204, 0, 0],
    [34, 0, 0],
    [0, 34, 0],
    [0, 34, 34],
    [0, 10, 10],
    [0, 112, 112],
    [51, 0, 51],
    [0, 0, 112],
    [0, 0, 51],
    [34, 34, 0],
    [80, 80, 0],
    [70, 0, 70],
    [0, 80, 80],
    [0, 50, 50],
    [80, 0, 80],
    [0, 0, 70],
    [0, 60, 0],
    [0, 0, 80],
    [0, 80, 0],
    [0, 80, 50],
    [80, 0, 50],
    [0, 50, 80],
    [50, 0, 80],
    [0, 20, 40],
    [0, 20, 80],
    [0, 20, 120],
    [0, 20, 160],
    [0, 20, 200],
    [0, 20, 240],
    [0, 20, 255],
    [20, 0, 40],
    [40, 0, 80],
    [60, 0, 120],
    [80, 0, 160],
    [100, 0, 200],
    [120, 0, 240],
    [140, 0, 255],
    [255, 255, 0],
    [255, 255, 40],
    [255, 255, 80],
    [255, 255, 120],
    [255, 255, 160],
    [255, 255, 200],
    [255, 255, 240],
    [255, 200, 0],
    [255, 160, 0],
    [255, 120, 0],
    [255, 80, 0],
    [255, 40, 0],
    [200, 255, 0],
    [160, 255, 0],
    [120, 255, 0],
    [80, 255, 0],
    [40, 255, 0],
    [200, 0, 255],
    [160, 0, 255],
    [120, 0, 255],
    [80, 0, 255],
    [40, 0, 255],
    [255, 0, 200],
    [255, 0, 160],
    [255, 0, 120],
    [255, 0, 80],
    [255, 0, 40],
    [0, 255, 200],
    [0, 255, 160],
    [0, 255, 120],
    [0, 255, 80],
    [0, 255, 40],
    [200, 200, 200],
    [180, 180, 180],
    [160, 160, 160],
    [140, 140, 140],
    [120, 120, 120],
    [100, 100, 100],
    [80, 80, 80],
    [60, 60, 60],
    [40, 40, 40],
    [20, 20, 20],
    [220, 20, 60],
    [119, 11, 32],
    [0, 0, 142],
    [0, 0, 230],
    [106, 0, 228],
    [0, 0, 70],
    [0, 60, 100],
    [0, 0, 90],
    [0, 0, 110],
    [0, 0, 130],
    [110, 110, 70],
    [120, 120, 80],
    [0, 0, 0],
], dtype=np.uint8)

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

ADE20K_CLASSES = [
    "background",  # 0
    "wall",  # 1
    "building",  # 2
    "sky",  # 3
    "floor",  # 4
    "tree",  # 5
    "ceiling",  # 6
    "road",  # 7
    "bed",  # 8
    "windowpane",  # 9
    "grass",  # 10
    "cabinet",  # 11
    "sidewalk",  # 12
    "person",  # 13
    "earth",  # 14
    "door",  # 15
    "table",  # 16
    "mountain",  # 17
    "plant",  # 18
    "curtain",  # 19
    "chair",  # 20
    "car",  # 21
    "water",  # 22
    "painting",  # 23
    "sofa",  # 24
    "shelf",  # 25
    "house",  # 26
    "sea",  # 27
    "mirror",  # 28
    "rug",  # 29
    "field",  # 30
    "armchair",  # 31
    "seat",  # 32
    "fence",  # 33
    "desk",  # 34
    "rock",  # 35
    "wardrobe",  # 36
    "lamp",  # 37
    "bathtub",  # 38
    "railing",  # 39
    "cushion",  # 40
    "base",  # 41
    "box",  # 42
    "column",  # 43
    "signboard",  # 44
    "chest_of_drawers",  # 45
    "counter",  # 46
    "sand",  # 47
    "sink",  # 48
    "skyscraper",  # 49
    "fireplace",  # 50
    "refrigerator",  # 51
    "grandstand",  # 52
    "path",  # 53
    "stairs",  # 54
    "runway",  # 55
    "case",  # 56
    "pool_table",  # 57
    "pillow",  # 58
    "screen_door",  # 59
    "stairway",  # 60
    "river",  # 61
    "bridge",  # 62
    "bookcase",  # 63
    "blind",  # 64
    "coffee_table",  # 65
    "toilet",  # 66
    "flower",  # 67
    "book",  # 68
    "hill",  # 69
    "bench",  # 70
    "countertop",  # 71
    "stove",  # 72
    "palm",  # 73
    "kitchen_island",  # 74
    "computer",  # 75
    "swivel_chair",  # 76
    "boat",  # 77
    "bar",  # 78
    "arcade_machine",  # 79
    "hovel",  # 80
    "bus",  # 81
    "towel",  # 82
    "light",  # 83
    "truck",  # 84
    "tower",  # 85
    "chandelier",  # 86
    "awning",  # 87
    "streetlight",  # 88
    "booth",  # 89
    "television_receiver",  # 90
    "airplane",  # 91
    "dirt_track",  # 92
    "apple_tree",  # 93
    "column_alt",  # 94
    "bannister",  # 95
    "escalator",  # 96
    "ottoman",  # 97
    "bottle",  # 98
    "buffet",  # 99
    "poster",  # 100
    "stage",  # 101
    "van",  # 102
    "ship",  # 103
    "fountain",  # 104
    "conveyer_belt",  # 105
    "canopy",  # 106
    "washer",  # 107
    "plaything",  # 108
    "swimming_pool",  # 109
    "stool",  # 110
    "barrel",  # 111
    "basket",  # 112
    "waterfall",  # 113
    "tent",  # 114
    "bag",  # 115
    "minibike",  # 116
    "cradle",  # 117
    "oven",  # 118
    "ball",  # 119
    "food",  # 120
    "step",  # 121
    "tank",  # 122
    "trade_name",  # 123
    "microwave",  # 124
    "pot",  # 125
    "animal",  # 126
    "bicycle",  # 127
    "lake",  # 128
    "dishwasher",  # 129
    "screen",  # 130
    "blanket",  # 131
    "sculpture",  # 132
    "hood",  # 133
    "sconce",  # 134
    "vase",  # 135
    "traffic_light",  # 136
    "tray",  # 137
    "ashcan",  # 138
    "fan",  # 139
    "pier",  # 140
    "crt_screen",  # 141
    "plate",  # 142
    "monitor",  # 143
    "bulletin_board",  # 144
    "shower",  # 145
    "radiator",  # 146
    "glass",  # 147
    "clock",  # 148
    "flag"  # 149
]


def decode_segmap(mask: np.ndarray, colormap) -> np.ndarray:
    """
    Convert class indices (H, W) into RGB color image using the Pascal VOC colormap.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label in range(len(colormap)):
        color_mask[mask == label] = colormap[label]
    return color_mask


def _denormalize(image,
                 mean=(123.675, 116.28, 103.53),
                 std=(58.395, 57.12, 57.375),
                 dataset="voc"):
    """
    Undo normalization applied in dataset pipeline.
    """
    mean = torch.tensor(mean, device=image.device).view(3, 1, 1)
    std = torch.tensor(std, device=image.device).view(3, 1, 1)

    # undo normalization
    img = image * std + mean
    img = img / 255.0
    if dataset == 'voc':
        img = img[[2, 1, 0]]
    return img.clamp(0, 1)


def annotate_segmentation(ax, mask, class_names, min_region_size=500):
    """
    Adds in-image text labels on the largest segmentation region per class.
    """
    unique_classes = np.unique(mask)
    for cls in unique_classes:
        if cls == 0:  # skip bg
            continue

        binary_mask = (mask == cls).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary_mask)

        # finds the largest region per-class
        max_region_size = 0
        max_region_idx = -1
        for region_idx in range(1, num_labels):
            region_size = (labels == region_idx).sum()
            if region_size > max_region_size:
                max_region_size = region_size
                max_region_idx = region_idx

        if max_region_idx == -1 or max_region_size < min_region_size:
            continue

        region = (labels == max_region_idx)
        cy, cx = np.round(center_of_mass(region)).astype(int)
        ax.text(
            cx, cy, class_names[cls],
            color="white", fontsize=8, weight="bold",
            ha="center", va="center",
            bbox=dict(facecolor="black", alpha=0.5, pad=1,
                      edgecolor="none")
        )


def visualize_segmentation(
        images,
        models,
        model_names,
        pred_segs=None,
        output_dir="images/segmentation_comparison",
        epoch=1,
        alpha=0.5,
        device=None,
        dataset="voc",
):
    """
    Visualize segmentation predictions from multiple models in a clean grid format.

    Each row corresponds to one image.
    Each column corresponds to a model's output (plus original & optional GT).
    """
    os.makedirs(output_dir, exist_ok=True)
    assert len(models) == len(
        model_names), "models and model_names must match in length"

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if dataset.lower() == "voc":
        colormap = VOC_COLORMAP
        classes = VOC_CLASSES
    else:
        colormap = ADE20K_COLORMAP
        classes = ADE20K_CLASSES

    # put into eval
    for model in models:
        model.to(device).eval()

    num_imgs = min(len(images), 4)
    num_models = len(models)
    include_gt = pred_segs is not None
    num_cols = 1 + num_models + int(
        include_gt)  # +1 for original, +1 for GT if given

    fig, axes = plt.subplots(num_imgs, num_cols,
                             figsize=(4 * num_cols, 4 * num_imgs))

    # Handle 1D cases gracefully
    if num_imgs == 1:
        axes = axes[None, :]
    if num_cols == 1:
        axes = axes[:, None]

    col_titles = ["Input"] + model_names + (
        ["Ground Truth"] if include_gt else [])

    for row_idx in range(num_imgs):
        img_t = images[row_idx].unsqueeze(0).to(device)
        img_rgb = _denormalize(images[row_idx], dataset=dataset).permute(1, 2, 0).cpu().numpy()
        img_rgb = np.clip(img_rgb, 0, 1)

        # original image
        axes[row_idx, 0].imshow(img_rgb)
        if row_idx == 0:
            axes[row_idx, 0].set_title(col_titles[0], fontsize=35, pad=8)

        # Predictions
        for j, (model, name) in enumerate(zip(models, model_names), start=1):
            with torch.no_grad():
                with torch.amp.autocast(device_type=device, enabled=False):
                    output = model(img_t)
                    logits = output["logits"] if isinstance(output,
                                                            dict) else output
                    pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(
                        np.uint8)

            color_mask = decode_segmap(pred, colormap)
            overlay = (1 - alpha) * img_rgb + alpha * (color_mask / 255.0)
            axes[row_idx, j].imshow(overlay)
            annotate_segmentation(axes[row_idx, j], pred, classes)
            if row_idx == 0:
                axes[row_idx, j].set_title(col_titles[j], fontsize=35, pad=8)

        # Ground truth if available
        if include_gt:
            gt = pred_segs[row_idx].cpu().numpy().astype(np.uint8)
            gt_color = decode_segmap(gt, colormap)
            axes[row_idx, -1].imshow(gt_color / 255.0)
            axes[row_idx, -1].set_title(col_titles[-1], fontsize=35, pad=8)

    # Clean up borders
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    plt.tight_layout(pad=2.0)
    filename = os.path.join(output_dir, f"seg_comparison_epoch_{epoch}.pdf")
    plt.savefig(filename, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

