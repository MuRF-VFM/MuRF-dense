import os

import torch
from .dataset.voc import get_voc_dataloader
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR, LinearLR, SequentialLR
from torch.amp.grad_scaler import GradScaler
from torch.amp import autocast

from tqdm import tqdm
from tqdm.auto import tqdm

import time

from .model import DinoV2SegmentationModelMRF
from .utils import sliding_window_inference
from .segmetric import SegmentationMetrics

# remove Adafactor if it exists to avoid conflicts
if hasattr(torch.optim, "Adafactor"):
    delattr(torch.optim, "Adafactor")

from .losses.cross_entropy_loss import CrossEntropyLoss


def Criterion():
    return CrossEntropyLoss(use_sigmoid=False, loss_weight=1.0)


def evaluate(model, dataloader, device):
    """Runs evaluation loop and returns averaged segmentation metrics."""
    model.eval().to(device)
    metrics = SegmentationMetrics(num_classes=21)
    total_loss = 0.0
    criterion = Criterion().to(device)

    with torch.no_grad(), autocast(device, enabled=False):
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            gt_seg = batch["labels"].to(device, non_blocking=True)

            preds = sliding_window_inference(model,
                                             21,
                                             pixel_values,
                                             (512, 512),
                                             (341, 341),
                                             1,
                                             return_logits=True)

            loss = criterion(preds.squeeze(1), gt_seg.squeeze(1).long(),
                             ignore_index=255)
            total_loss += loss.item()

            # get predicted class indices (argmax over channels)
            preds = torch.argmax(preds, dim=1)

            # update metric
            metrics.update(preds, gt_seg.squeeze(1))

    results = metrics.compute()
    avg_loss = total_loss / len(dataloader)
    return avg_loss, results


def train_mrf(
        model=None,
        root_dir='/home/ubuntu/coding/datasets/VOC2012',
        max_iters=40000,
        log_interval=7680,
        val_interval=10000,
        ckpt_interval=7680,
        batch_size=16,  # 2 per GPU with 8 GPUs
        checkpoint_path='./checkpoints',
        num_workers=6,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(checkpoint_path, exist_ok=True)

    train_dataloader = get_voc_dataloader(root_dir,
                                          batch_size,
                                          'trainaug',
                                          shuffle=True,
                                          num_workers=num_workers)
    val_dataloader = get_voc_dataloader(root_dir,
                                        1,
                                        'val',
                                        shuffle=False, num_workers=num_workers)
    test_dataloader = get_voc_dataloader(root_dir,
                                         1,
                                         'val',
                                         shuffle=False,
                                         num_workers=num_workers)

    model.to(device)
    criterion = Criterion().to(device)
    optimizer = AdamW(params=model.seg_head.parameters(),
                      lr=0.001,
                      weight_decay=0.0001,
                      betas=(0.9, 0.999))

    # Warmup scheduler
    warmup_scheduler = LinearLR(optimizer,
                                start_factor=1e-6,
                                end_factor=1.0,
                                total_iters=1500)
    poly_scheduler = PolynomialLR(optimizer,
                                  total_iters=max_iters - 1500,
                                  power=1.0)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_scheduler, poly_scheduler],
                             milestones=[1500])
    scaler = GradScaler(device)

    # iteration counter
    iter_idx = 0
    train_iter = iter(train_dataloader)

    # put model in training mode
    model.train()

    start_time = time.time()
    # iteration-based training loop
    with tqdm(total=max_iters, desc="Training") as pbar:
        while iter_idx < max_iters:
            try:
                batch = next(train_iter)
            except StopIteration:
                # restart dataloader if necessary
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            iter_idx += 1
            optimizer.zero_grad(set_to_none=True)

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            gt_seg = batch["labels"].to(device, non_blocking=True)
            img_metas = batch["img_metas"]

            with autocast(device):
                predicted_depth = model(pixel_values)

                # compute loss
                loss = criterion(predicted_depth.squeeze(1),
                                 gt_seg.squeeze(1).long(), ignore_index=255)

            # backward + optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # logging
            if iter_idx is not None and iter_idx % log_interval == 0:
                tqdm.write(f"[Iter {iter_idx}] loss={loss.item():.4f}")

            # validating
            if val_interval and iter_idx % val_interval == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                print(f"Time elapsed for {val_interval} iters: {elapsed:.2f} seconds")
                val_loss, val_metrics = evaluate(model, val_dataloader, device)
                tqdm.write(f"Validation @ {iter_idx}: loss={val_loss:.4f}, "
                           f"mIoU={val_metrics['mIoU']:.3f}, "
                           f"Pixel Acc={val_metrics['aAcc']:.3f}"
                           f"mFscore={val_metrics['mFscore']:.3f}")
                model.train()

            # checkpointing
            if ckpt_interval and iter_idx % ckpt_interval == 0:
                os.makedirs(os.path.join(checkpoint_path,
                                         str((model.resolutions).replace(',', '_'))), exist_ok=True)
                ckpt_path = os.path.join(checkpoint_path,
                                            str((model.resolutions).replace(',', '_')),
                                         f"iter_{iter_idx}.pth")
                torch.save({
                    "iter": iter_idx,
                    "seg_head": model.seg_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "resolutions": model.resolutions,
                }, ckpt_path)
                tqdm.write(f"Saved checkpoint to {ckpt_path}")

            pbar.update(1)

    end_time = time.time()
    total_time = end_time - start_time

    # run test
    model.eval()
    print("Running final evaluation on test set...")
    test_loss, test_metrics = evaluate(model, test_dataloader, device)
    print(f"[TEST] loss={test_loss:.4f}, "
          f"mIoU={test_metrics['mIoU']:.3f}, "
          f"Pixel Acc={test_metrics['aAcc']:.3f}"
          f"mFscore={test_metrics['mFscore']:.3f}")

    os.makedirs(os.path.join(checkpoint_path,
                             str((model.resolutions)).replace(',', '_')), exist_ok=True)
    with open(os.path.join(checkpoint_path, str((model.resolutions)).replace(',', '_'), "res.txt"), "a") as f:
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")

    with open(os.path.join(checkpoint_path, "res.txt"), "a") as f:
        f.write(f"MRF Resolutions: {model.resolutions}\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")
        print(f"Total training time: {total_time:.2f} seconds")
        

    torch.save({
        "iter": iter_idx,
        "seg_head": model.seg_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "resolutions": model.resolutions,
    }, os.path.join(checkpoint_path, str((model.resolutions)).replace(',', '_'), f"final_checkpoint.pth"))


def set_seed():
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


if __name__ == "__main__":
    set_seed()
    # checkpoint = './weights/mrf.pth'
    # head_state_dict = torch.load(checkpoint, map_location='cpu')['seg_head']
    # 148 266 518
    model = DinoV2SegmentationModelMRF(num_classes=21, resolutions=[140])
    # model.seg_head.load_state_dict(head_state_dict)
    train_mrf(
        model=model,
        max_iters=6_000,#40_000 is config, 6000 seems sufficient
        val_interval=500,
        batch_size=16,
        ckpt_interval=40_000,
        root_dir='/home/ubuntu/coding/datasets/VOC2012',
        checkpoint_path='./reference_mrf_voc'
    )

    # evaluate(model, get_voc_dataloader('/home/ubuntu/coding/datasets/VOC2012', 1, 'val', shuffle=False, num_workers=6), device="cuda" if torch.cuda.is_available() else "cpu")




