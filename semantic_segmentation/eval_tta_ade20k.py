import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation
import numpy as np
from tqdm import tqdm
from PIL import Image
import sys

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Use Hugging Face transformers
from transformers import AutoImageProcessor, AutoModel

from dataset.ade import ADE20KDataset
import torchvision.transforms as T

import transforms

import functools
import semantic_segmentation.utils as utils
from semantic_segmentation.model import *

crop_sizes = [266, 518, 784]

def evaluate(models, dataloader, device):
    for model in models:
        model.eval()

    preds = []
    masks_tot = []
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            images = data["images"]
            masks = data["masks"]
            images = images.to(device)
            masks = masks.to(device)

            assert images.shape[0] == 1  # Ensure batch size is 1 for evaluation

            ori_size = images.shape[2:]
    
            # output = utils.sliding_window_inference(model, 151, images, (518, 518), (341, 341), 32, return_logits=True)
            stacked_output = torch.zeros((1, 151, ori_size[0], ori_size[1])).to(device)

            for j, model in enumerate(models):
                resized_images = T.Resize((crop_sizes[j], crop_sizes[j]))(images)
                output = model(resized_images)
                output = F.interpolate(output, size=ori_size, mode='nearest')
                stacked_output = stacked_output + output
            pred = torch.argmax(stacked_output, dim=1)

            assert pred.shape == masks.shape
            # loss = criterion(output, masks)
            preds.append(pred.cpu())
            masks_tot.append(masks.cpu())
            # running_loss += loss.item()

    # loss = running_loss / len(dataloader)

    return preds, masks_tot

def collate_fn_single_res(batch, transform):
    transformed = [ transform.transform(image=item['image'], mask=item['mask']) for item in batch]
    images = torch.stack([torch.tensor(item['image']).permute(2, 0, 1) for item in transformed])
    masks = torch.stack([torch.LongTensor(item['mask']) for item in transformed])
    return {"images": images, "masks": masks}

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    num_classes = 151  # For ADE20K

    # Model name for DinoV2 from Hugging Face
    model_name = "facebook/dinov2-base"
    
    # Load processor
    processor = AutoImageProcessor.from_pretrained(model_name)

    train_transform = transforms.SegmentationTransform(augment=True)
    val_transform = transforms.SegmentationTransform(augment=False)
    
    # Load datasets
    train_dataset = ADE20KDataset(root="../datasets/ADEChallengeData2016", split="training")
    val_dataset = ADE20KDataset(root="../datasets/ADEChallengeData2016", split="validation")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=functools.partial(collate_fn_single_res, transform=train_transform))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=functools.partial(collate_fn_single_res, transform=val_transform))

    # Initialize model
    # models = []
    # model_266 = DinoV2SegmentationModel(num_classes=num_classes, model_name=model_name).to(device)
    # model_266.seg_head.load_state_dict(torch.load("ckpt/bilinear_266/checkpoint_epoch_44.pth")['model_state_dict'])
    # models.append(model_266)
    # model_518 = DinoV2SegmentationModel(num_classes=num_classes, model_name=model_name).to(device)
    # model_518.seg_head.load_state_dict(torch.load("ckpt/bilinear_518/checkpoint_epoch_44.pth")['model_state_dict'])
    # models.append(model_518)
    # model_784 = DinoV2SegmentationModel(num_classes=num_classes, model_name=model_name).to(device)
    # model_784.seg_head.load_state_dict(torch.load("ckpt/bilinear_784/checkpoint_epoch_48.pth")['model_state_dict'])
    # models.append(model_784)
    
    models = []
    model_266 = DinoV2SegmentationModel(num_classes=num_classes, model_name=model_name).to(device)
    model_266.seg_head.load_state_dict(torch.load("ckpt/bilinear_784/checkpoint_epoch_48.pth")['model_state_dict'])
    models.append(model_266)
    model_518 = DinoV2SegmentationModel(num_classes=num_classes, model_name=model_name).to(device)
    model_518.seg_head.load_state_dict(torch.load("ckpt/bilinear_784/checkpoint_epoch_48.pth")['model_state_dict'])
    models.append(model_518)
    model_784 = DinoV2SegmentationModel(num_classes=num_classes, model_name=model_name).to(device)
    model_784.seg_head.load_state_dict(torch.load("ckpt/bilinear_784/checkpoint_epoch_48.pth")['model_state_dict'])
    models.append(model_784)


    max_miou = 0.0
    max_idx = -1

    preds, masks_tot = evaluate(models, val_loader, device)
    metrics = utils.get_macc_miou(preds, masks_tot, num_classes)

    print(f"Validation mIoU: {metrics["mean_iou"]:.4f}")

if __name__ == "__main__":
    main()
