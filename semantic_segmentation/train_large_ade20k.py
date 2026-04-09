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

# crop_size = 784
crop_size = 256


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for data in tqdm(dataloader):
        images = data["images"]
        masks = data["masks"]
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images, original_size=(crop_size, crop_size))
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

            ori_size = (crop_size, crop_size)

            # output = utils.sliding_window_inference(model, 151, images, (518, 518), (341, 341), 32, return_logits=True)
            output = model(images, original_size=ori_size)
            pred = torch.argmax(output, dim=1)
            pred = F.interpolate(pred.unsqueeze(1).float(), size=ori_size, mode='nearest')[:, 0, :, :].long()
            assert pred.shape == masks.shape
            # loss = criterion(output, masks)
            preds.append(pred.cpu())
            masks_tot.append(masks.cpu())
            # running_loss += loss.item()

    # loss = running_loss / len(dataloader)

    return preds, masks_tot

def collate_fn_single_res(batch, transform):
    transformed = [ transform.transform(image=item['image'], mask=item['mask']) for item in batch]
    if transform.processor is not None:
        processed_images = transform.processor(
            images=[item['image'] for item in transformed],
            return_tensors="pt"
        )
        images = processed_images['pixel_values']
    else:
        images = torch.stack([torch.tensor(item['image']).permute(2, 0, 1) for item in transformed])
    masks = torch.stack([torch.LongTensor(item['mask']) for item in transformed])
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
    # model_name = "facebook/dinov2-base"
    model_name = "google/siglip2-base-patch16-naflex"
    
    # train_transform = transforms.SegmentationTransform(augment=True, crop_size = crop_size)
    # val_transform = transforms.SegmentationTransform(augment=False, crop_size = crop_size)

    train_transform = transforms.SegmentationTransformSiglip(augment=True, crop_size = crop_size, model_name = model_name)
    val_transform = transforms.SegmentationTransformSiglip(augment=False, crop_size = crop_size, model_name = model_name)
    
    # Load datasets
    train_dataset = ADE20KDataset(root="../datasets/ADEChallengeData2016", split="training")
    val_dataset = ADE20KDataset(root="../datasets/ADEChallengeData2016", split="validation")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=functools.partial(collate_fn_single_res, transform=train_transform))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=functools.partial(collate_fn_single_res, transform=val_transform))

    # Initialize model
    # model = DinoV2SegmentationModel(num_classes=num_classes, model_name=model_name).to(device)
    model = Siglip2SegmentationModel(num_classes=num_classes, model_name=model_name).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.seg_head.parameters(), lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999))
    
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
        print(f"Validation mIoU: {metrics["mean_iou"]:.4f}, Max mIoU: {max_val_miou:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.seg_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
        }, f"ckpt/checkpoint_epoch_{epoch+1}.pth")
    
    # Save final model
    # torch.save(model.seg_head.state_dict(), "dinov2_segmentation_model.pth")
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_curve.png')

if __name__ == "__main__":
    main()
