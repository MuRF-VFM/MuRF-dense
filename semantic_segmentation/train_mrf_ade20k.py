import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

from dataset.ade import ADE20KDataset

import transforms

import functools
import utils

from model import *

crop_size = 784

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        in_channels: number of channels after concatenation (e.g. embed_dim * num_resolutions)
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.bn(x)
        x = self.classifier(x)
        return x

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for data in tqdm(dataloader):
        images1 = data["images1"]
        images2 = data["images2"]
        images3 = data["images3"]
        masks = data["masks"]
        images1 = images1.to(device)
        images2 = images2.to(device)
        images3 = images3.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images1, images2, images3)  # logits at original image spatial size
        loss = criterion(outputs, masks)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    preds = []
    masks_tot = []
    
    with torch.no_grad():
        for data in tqdm(dataloader):
            images = data["images"]
            masks = data["masks"]
            images = images.to(device)
            masks = masks.to(device)

            ori_size = images.shape[2:]

            # Pass original images to model; model internally resizes to multi-res.
            output = model(images)  # (B, num_classes, H, W) where H,W == ori_size
            pred = torch.argmax(output, dim=1)  # (B, H, W)
            # ensure shapes match
            assert pred.shape == masks.shape
            preds.append(pred.cpu())
            masks_tot.append(masks.cpu())

    return preds, masks_tot

def collate_fn_single_res(batch, transform):
    # transform.transform returns dict-like with 'image' and 'mask' presumably
    transformed = [ transform.transform(image=item['image'], mask=item['mask']) for item in batch]

    # image expected to be H x W x C ; convert to (C, H, W) float tensor
    images = torch.stack([torch.tensor(item['image']).permute(2, 0, 1).float() for item in transformed])
    # masks expected to be H x W ; LongTensor
    masks = torch.stack([torch.LongTensor(item['mask']) for item in transformed])
    return {"images": images, "masks": masks}

def collate_fn_multi_res(batch, transform):
    transformed = [ transform.transform(image=item['image'], mask=item['mask']) for item in batch]
    masks = torch.stack([torch.LongTensor(item['mask']) for item in transformed])
    if transform.processor is not None:
        images = torch.from_numpy(np.stack([item['image'] for item in transformed]))
        images1 = F.interpolate(images.permute(0,3,1,2), size=(256,256), mode='bilinear', align_corners=False).permute(0,2,3,1)
        images1 = transform.processor[0](
            images=[item['image'] for item in transformed],
            return_tensors="pt"
        )['pixel_values']
        images2 = F.interpolate(images.permute(0,3,1,2), size=(512,512), mode='bilinear', align_corners=False).permute(0,2,3,1)
        images2 = transform.processor[1](
            images=[item['image'] for item in transformed],
            return_tensors="pt"
        )['pixel_values']
        images3 = F.interpolate(images.permute(0,3,1,2), size=(768,768), mode='bilinear', align_corners=False).permute(0,2,3,1)
        images3 = transform.processor[2](
            images=[item['image'] for item in transformed],
            return_tensors="pt"
        )['pixel_values']

        return {"images1": images1, "images2": images2, "images3": images3, "masks": masks}
    else:
        images = torch.stack([torch.tensor(item['image']).permute(2, 0, 1) for item in transformed])
    
        return {"images": images, "masks": masks}



def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.001
    num_classes = 151  # For ADE20K

    # Model name for DinoV2 from Hugging Face
    model_name = "google/siglip2-base-patch16-naflex"

    # train_transform = transforms.SegmentationTransform(augment=True, crop_size = crop_size)
    # val_transform = transforms.SegmentationTransform(augment=False, crop_size = crop_size)

    train_transform = transforms.SegmentationTransformSiglipMRF(augment=True)
    val_transform = transforms.SegmentationTransformSiglipMRF(augment=False)

    
    # Load datasets
    train_dataset = ADE20KDataset(root="../datasets/ADEChallengeData2016", split="training")
    val_dataset = ADE20KDataset(root="../datasets/ADEChallengeData2016", split="validation")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=functools.partial(collate_fn_multi_res, transform=train_transform))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=functools.partial(collate_fn_multi_res, transform=val_transform))

    # Initialize model (runs backbone at 266, 518, 784 by default)
    # model = DinoV2SegmentationModelMRF(num_classes=num_classes, model_name=model_name, pretrained=True, resolutions=(266, 518, 784)).to(device)
    model = Siglip2SegmentationModelMRF(num_classes=num_classes, model_name=model_name, pretrained=True, resolutions=(256, 512, 768)).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # Only optimize segmentation head parameters (backbone is frozen)
    optimizer = optim.AdamW(model.seg_head.parameters(), lr=learning_rate, weight_decay=0.0001, betas=(0.9, 0.999))
    
    # Training loop
    train_losses = []
    max_val_miou = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss = train(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        print(f"Training Loss: {train_loss:.4f}")

        preds, masks_tot = evaluate(model, val_loader, criterion, device)

        # Calculate mIoU for validation set
        print("Calculating mIoU...")
        metrics = utils.get_macc_miou(preds, masks_tot, num_classes)
        
        max_val_miou = max(max_val_miou, metrics["mean_iou"])
        print(f"Validation mIoU: {metrics['mean_iou']:.4f}, Max mIoU: {max_val_miou:.4f}")

        # Save checkpoint
        os.makedirs("ckpt", exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.seg_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
        }, f"ckpt/checkpoint_epoch_{epoch+1}.pth")
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.savefig('loss_curve.png')

if __name__ == "__main__":
    main()
