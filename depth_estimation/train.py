import os

import torch
from dataset.NYU import get_nyu_dataloader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from tqdm.auto import tqdm

from dataset.SUNRGBD import get_sunrgb_dataloader
from callbacks.vismap import VisualizeDepthMap
from dino_model import DINOv2DepthEstimation
from losses.gradientloss import GradientLoss
from losses.sigloss import SigLoss
from val import val
from tests import test, flip_test, flip_test_sun


class Criterion(torch.nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.sig_loss = SigLoss(valid_mask=True,
                                loss_weight=1.0,
                                max_depth=10,
                                warm_up=True,
                                warm_iter=1000)
        self.grad_loss = GradientLoss(valid_mask=True, loss_weight=0.5)

    def forward(self, pred, target):
        return self.sig_loss(pred, target) + self.grad_loss(pred, target)


def train(
        root_dir='/nobackup/shared/datasets/nyu',
        model_name='facebook/dinov2-base',
        model=None,
        max_iters=38400,
        log_interval=7680,
        val_interval=3840,
        ckpt_interval=7680,
        save_images=False,
        batch_size=2,
        checkpoint_path='checkpoints',
        sun_test=False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataloaders
    train_dataloader = get_nyu_dataloader(
        root_dir=root_dir,
        split='train',
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = get_nyu_dataloader(
        root_dir=root_dir,
        split='val',
        batch_size=batch_size,
        shuffle=False,
    )
    test_dataloader = get_nyu_dataloader(
        root_dir=root_dir,
        split='test',
        batch_size=batch_size,
        shuffle=False,
    )

    if model is None:
        model = DINOv2DepthEstimation.from_pretrained(model_name, norm_strategy="nonlinear",).to(device)
    else:
        model = model.to(device)
    print(model)

    criterion = Criterion().to(device)
    optimizer = AdamW(params=model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device)

    callbacks = []
    if save_images:
        callbacks.append(VisualizeDepthMap(output_dir="images/images_train", model=model, frequency=1))

    # early_stopping = EarlyStopping()
    loss_history, val_loss_history = [], []

    # iteration counter
    iter_idx = 0
    train_iter = iter(train_dataloader)

    # put model in training mode
    model.train()

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
            gt_depths = batch["labels"].to(device, non_blocking=True)
            img_metas = batch["img_metas"]

            with torch.amp.autocast(device):
                outputs = model(pixel_values)
                predicted_depth = outputs["predicted_depth"]

                # compute loss
                loss = criterion(predicted_depth.squeeze(1), gt_depths.squeeze(1))

            # backward + optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # logging
            if log_interval is not None and iter_idx % log_interval == 0:
                print(f"Iter {iter_idx}/{max_iters} - Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6e}")
                loss_history.append(loss.item())

            # validation
            if val_interval is not None and iter_idx % val_interval == 0:
                val_results = val(model, val_dataloader)
                val_loss_history.append(val_results)
                print(f"validation at {iter_idx}] loss={val_results['loss']}, val rmse={val_results['rmse']}")

                # if early_stopping(val_results['loss'], val_results['rmse']):
                #     print("Early stopping triggered.")
                #     break

            # checkpointing
            if ckpt_interval is not None and iter_idx % ckpt_interval == 0:
                os.makedirs(checkpoint_path, exist_ok=True)
                ckpt_path = f"{checkpoint_path}/iter{iter_idx}.pth"
                torch.save({
                    "iter": iter_idx,
                    "head_state_dict": model.depth_estimation_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                    "val_loss": val_results['loss'] if len(val_loss_history) > 0 else None,
                    "val_rmse": val_results['rmse'] if len(val_loss_history) > 0 else None,
                }, ckpt_path)
                print(f"checkpoint saved at: {ckpt_path}")

            loss_history.append(loss.item())
            pbar.update(1)

    with open("loss_history_dino.txt", "a") as f:
        f.write(f"Iteration: {iter_idx}\n")
        for loss in loss_history:
            f.write(f"train loss: {loss}\n")
        for val_loss in val_loss_history:
            f.write(
                f"val loss: {val_loss['loss']}, val_rmse: {val_loss['rmse']}\n\n")

    # run test
    print('running test')
    test_res = test(model, save_images=save_images, dataloader=val_dataloader)

    print('running flip test (aggregates predictions over hflip aug)')
    flip_test_res = flip_test(model, test_dataloader=test_dataloader)

    # save final model
    os.makedirs(checkpoint_path, exist_ok=True)
    final_ckpt_path = os.path.join(checkpoint_path,
                                   f"dino_model.pth")

    with open(os.path.join(checkpoint_path,
                                   f"results.txt"), 'a') as f:
        f.write("DINOv2DepthEstimationMRF\n")
        for k, v in test_res.items():
            f.write(f"{k}: {v}\n")
        for k, v in flip_test_res.items():
            f.write(f"flip_{k}: {v}\n")
        f.write("\n")

    if sun_test:
        val_dataloader = get_sunrgb_dataloader(
            split='val',
            root_dir='/nobackup/shared/datasets/',
            split_file='./SUNRGBD_val_splits.txt',
            batch_size=batch_size,
            shuffle=False)
        test_dataloader = get_sunrgb_dataloader(
            split='test',
            batch_size=batch_size,
            root_dir='/nobackup/shared/datasets/',
            split_file='./SUNRGBD_val_splits.txt',
            shuffle=False)
        print("running SUNRGBD val set test")
        sun_test_res = test(model, val_dataloader)
        print("running SUNRGBD test set flip test")
        sun_flip_test_res = flip_test_sun(model, test_dataloader)

        with open(f"{checkpoint_path}/sunrgbd_results.txt", "a") as f:
            f.write("SUNRGBD Dino MRF results:\n")
            for k, v in sun_test_res.items():
                f.write(f"{k}: {v}\n")
            f.write("SUNRGBD Flip Test Results:\n")
            for k, v in sun_flip_test_res.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

    torch.save({
        "iter": iter_idx,
        "head_state_dict": model.depth_estimation_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_history[-1] if len(loss_history) > 0 else -1,
        "loss_history": loss_history,
        "val_loss": val_results['loss'] if len(val_loss_history) > 0 else None,
        "val_rmse": val_results['rmse'] if len(val_loss_history) > 0 else None,
        "out_indices": model.out_indices,
    }, final_ckpt_path)


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
    model = DINOv2DepthEstimation.from_pretrained('facebook/dinov2-base',
                                                  norm_strategy="nonlinear",
                                                  out_indices=[11])
    train(root_dir='/nobackup/shared/datasets/nyu',
          checkpoint_path='./giant/standard',
          ckpt_interval=None,
          model=model,
          sun_test=True
          )
