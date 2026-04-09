import os
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from tqdm.auto import tqdm

# from depth_estimation.callbacks.vismap import VisualizeDepthMap
from depth_estimation.dataset.NYU import get_nyu_dataloader
from depth_estimation.dataset.SUNRGBD import get_sunrgb_dataloader
from depth_estimation.losses.gradientloss import GradientLoss
from depth_estimation.losses.sigloss import SigLoss
from depth_estimation.mrf_model import DINOv2DepthEstimationMRF
from depth_estimation.tests import test
from depth_estimation.val import val


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


def train_mrf(
        root_dir='/nobackup/shared/datasets/nyu',
        model: torch.nn.Module = None,
        max_iters=38400,
        model_name='facebook/dinov2-base',
        log_interval=None,
        val_interval=3200,
        ckpt_interval=6400,
        save_images=False,
        batch_size=2,
        checkpoint_path='checkpoints_mrf',
        scales: float | list[float] = 1,
        sun_test=False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataloaders
    train_dataloader = get_nyu_dataloader(
        root_dir=root_dir,
        split='train',
        batch_size=batch_size,
        shuffle=True,
        # list=False
    )
    val_dataloader = get_nyu_dataloader(
        root_dir=root_dir,
        split='val',
        batch_size=batch_size,
        shuffle=False,
        # list=False
    )
    test_dataloader = get_nyu_dataloader(
        root_dir=root_dir,
        split='test',
        batch_size=batch_size,
        shuffle=False,
        # list=False
    )

    sun_mode = "sun_new" # or sun_mode = "sun_old" for old test

    if sun_test:
        # ensure the split file is at the current root
        assert os.path.exists(
            os.path.join(os.curdir, './depth_estimation/SUNRGBD_val_splits.txt')), "SUNRGBD split file not found in root_dir"
    os.makedirs(checkpoint_path, exist_ok=True)

    if model is None:
        model = DINOv2DepthEstimationMRF.from_pretrained(model_name,
                                                         scales=scales)

    criterion = Criterion().to(device)
    optimizer = AdamW(params=model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device)

    model.to(device)
    criterion.to(device)

    # early_stopping = EarlyStopping()
    loss_history, val_loss_history = [], []

    # iteration counter
    iter_idx = 0
    train_iter = iter(train_dataloader)
    file_name = "_".join([str(scale) for scale in model.scales])

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
            gt_depths = batch["labels"].to(device, non_blocking=True)
            img_metas = batch["img_metas"]

            with torch.amp.autocast(device):
                outputs = model(pixel_values)
                predicted_depth = outputs["predicted_depth"]

                # compute loss
                loss = criterion(predicted_depth.squeeze(1),
                                 gt_depths.squeeze(1))

            # backward + optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # logging
            if log_interval and iter_idx % log_interval == 0:
                print(
                    f"Iter {iter_idx}/{max_iters} - Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6e}")
                loss_history.append(loss.item())

            # validation
            if val_interval and iter_idx % val_interval == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                print(
                    f"Elapsed time for {iter_idx} iters: {elapsed:.2f} seconds")
                val_results = val(model, val_dataloader)
                val_loss_history.append(val_results)
                print(
                    f"validation at {iter_idx}] loss={val_results['loss']}, val rmse={val_results['rmse']}")

                # if early_stopping(val_results['loss'], val_results['rmse']):
                #     print("Early stopping triggered.")
                #     break

            # checkpointing
            if iter_idx % ckpt_interval == 0:
                os.makedirs(os.path.join(checkpoint_path, str(model.scales)),
                            exist_ok=True)
                ckpt_path = os.path.join(checkpoint_path, str(model.scales),
                                         f"iter_mrf{iter_idx}_{file_name}.pth")
                torch.save({
                    "iter": iter_idx,
                    "head_state_dict": model.depth_estimation_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                    "val_loss": val_results['loss'] if len(
                        val_loss_history) > 0 else None,
                    "val_rmse": val_results['rmse'] if len(
                        val_loss_history) > 0 else None,
                }, ckpt_path)
                print(f"checkpoint saved at: {ckpt_path}")

            loss_history.append(loss.item())
            pbar.update(1)
    
    end_time = time.time()

    with open(f"{checkpoint_path}/loss_history_mrf.txt", "a") as f:
        f.write(f"Iteration: {iter_idx}\n")
        for loss in loss_history:
            f.write(f"train_mrf loss: {loss}\n")
        for val_loss in val_loss_history:
            f.write(
                f"val loss: {val_loss['loss']}, val_rmse: {val_loss['rmse']}\n\n")

    # run test
    print('Running NYU Test (No TTA)')
    test_res = test(model, test_dataloader, use_tta=False, mode="nyu")
    print(test_res)
    print()

    print("Running NYUd Test Using TTA (Horiztontal flip inference aggregation)")
    flip_test_res = test(model, test_dataloader, use_tta=True, mode="nyu")
    print(flip_test_res)
    print()


    os.makedirs(f"{checkpoint_path}/{str(model.scales)}", exist_ok=True)
    with open(f"{checkpoint_path}/{str(model.scales)}/mrf_results.txt", "a") as f:
        f.write("Dino MRF results:\n")
        for k, v in test_res.items():
            f.write(f"{k}: {v}\n")
        f.write("Flip Test Results:\n")
        for k, v in flip_test_res.items():
            f.write(f"{k}: {v}\n")
        f.write("Time taken (seconds): ")
        f.write(f"{end_time - start_time}\n")
        f.write("\n")

    with open(f"{checkpoint_path}/results.txt", "a") as f:
        f.write(f"Scales: {model.scales}\n")
        f.write("NYU RMSE: ")
        f.write(f"{test_res['rmse']}\n")
        f.write("NYU Flip Test RMSE: ")
        f.write(f"{flip_test_res['rmse']}\n")
        f.write("\n")

    if sun_test:
        val_dataloader = get_sunrgb_dataloader(
            split='val',
            root_dir='/nobackup/shared/datasets',
            # root_dir='./sun',
            split_file='./depth_estimation/SUNRGBD_val_splits.txt',
            batch_size=batch_size,
            shuffle=False)
        test_dataloader = get_sunrgb_dataloader(
            split='test',
            root_dir='/nobackup/shared/datasets',
            # root_dir='./sun',
            split_file='./depth_estimation/SUNRGBD_val_splits.txt',
            batch_size=batch_size,
            shuffle=False)
        
        print("Running SUNRGBD Base Test (No TTA)")
        sun_test_res = test(model, test_dataloader, use_tta=False, mode=sun_mode)
        print(sun_test_res)
        print()
        
        print("Running SUNRGBD Test Using TTA (Flip inference aggregation)")
        sun_flip_test_res = test(model, test_dataloader, use_tta=True, mode=sun_mode)
        print(sun_flip_test_res)
        print()

        with open(f"{checkpoint_path}/{str(model.scales)}/mrf_sunrgbd_results.txt", "a") as f:
            f.write("SUNRGBD Dino MRF results:\n")
            for k, v in sun_test_res.items():
                f.write(f"{k}: {v}\n")
            f.write("SUNRGBD Flip Test Results:\n")
            for k, v in sun_flip_test_res.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

        with open(f"{checkpoint_path}/results.txt", "a") as f:
            f.write("SUNRGBD Dino RMSE: ")
            f.write(f"{sun_test_res['rmse']}\n")
            f.write("SUNRGBD Dino Flip Test RMSE: ")
            f.write(f"{sun_flip_test_res['rmse']}\n")
            f.write("\n")
        
    with open(f"{checkpoint_path}/time_logs.txt", "a") as f:
        f.write(f"Scales: {model.scales}\n")
        f.write(f"Time taken for {iter_idx} iters: {end_time - start_time} seconds\n")
        f.write("\n")

    # save final model
    os.makedirs(checkpoint_path, exist_ok=True)
    final_ckpt_path = os.path.join(checkpoint_path, str(model.scales), f"mrf_{file_name}.pth")

    torch.save({
        "iter": iter_idx,
        "head_state_dict": model.depth_estimation_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_history[-1] if len(loss_history) > 0 else -1,
        "loss_history": loss_history,
        "val_loss": val_results['loss'] if len(val_loss_history) > 0 else None,
        "val_rmse": val_results['rmse'] if len(val_loss_history) > 0 else None,
        "scales": model.scales,
        "out_indices": model.out_indices,
        "upsample_factor": model.upsample_factor
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
    # import combinations
    from itertools import combinations
    set_seed()
    scales = [1.0]

    possible_indices = [2, 5, 8, 11]
    combs_of_three = combinations(possible_indices, 3)
    # combs_of_three.append([3, 7, 11]) # add a custom
    for combination in combs_of_three:
        indices = list(combination)
        if 11 not in indices:
            continue
        indices = [11]

        underscore_string = "_".join(str(i) for i in indices)
        model = DINOv2DepthEstimationMRF.from_pretrained(
            'facebook/dinov2-base',
            scales=scales,
            norm_strategy="nonlinear",
            out_indices=indices)

        train_mrf(scales=scales,
                  root_dir='/nobackup/shared/datasets/nyu',
                  # checkpoint_path=f'checkpoints_mrf/{underscore_string}',
                  # root_dir='./nyu_depth_v2/nyu',
                  checkpoint_path=f'./results',
                  ckpt_interval=38400,
                  val_interval=10_000,
                  max_iters=38_400,
                  model=model,
                  sun_test=True
                  )

        break

