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
from . import utils
from .model import *

crop_size = 518
ckpt_size = 518

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

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
            resized_images = T.Resize((crop_size, crop_size))(images)
            output = model(resized_images)
            pred = torch.argmax(output, dim=1)
            pred = F.interpolate(pred.unsqueeze(1).float(), size=ori_size, mode='nearest')[:, 0, :, :].long()
            # print(pred.shape, masks.shape)
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

    val_transform = transforms.SegmentationTransform(augment=False)
    
    # Load datasets
    train_dataset = ADE20KDataset(root="../datasets/ADEChallengeData2016", split="training")
    val_dataset = ADE20KDataset(root="../datasets/ADEChallengeData2016", split="validation")

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=functools.partial(collate_fn_single_res, transform=val_transform))

    # Initialize model
    model = DinoV2SegmentationModel(num_classes=num_classes, model_name=model_name).to(device)

    max_miou = 0.0
    max_idx = -1

    for i in range(1, 51):
        model.seg_head.load_state_dict(torch.load(f"ckpt/bilinear_{ckpt_size}/checkpoint_epoch_{i}.pth")['model_state_dict'])

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()

        preds, masks_tot = evaluate(model, val_loader, criterion, device)
        metrics = utils.get_macc_miou(preds, masks_tot, num_classes)

        print(f"{i}: Validation mIoU: {metrics["mean_iou"]:.4f}")

        if metrics["mean_iou"] > max_miou:
            max_miou = metrics["mean_iou"]
            max_idx = i

    print(f"Best mIoU: {max_miou:.4f} at epoch {max_idx}")

if __name__ == "__main__":
    main()
