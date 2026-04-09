import os
from tqdm import tqdm
from torch import nn
import torch

from depth_estimation.metric import AllMetrics
from depth_estimation.callbacks.vismap import VisualizeDepthMap
import torch.nn.functional as F

def test(model, dataloader, use_tta=False, mode="nyu"):
    """
    Test the depth estimation model on the provided dataloader.
    Opptional test-time augmentation (TTA) with prediction aggregation over
    horizontal flip is supported.
    """
    valid_modes = ["nyu", "sun_old", "sun_new", "sun"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
    if mode == "sun": mode ="sun_new"

    metric = AllMetrics()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch["pixel_values"].to(device, non_blocking=True)
            
            if mode == "sun_old":
                gt = batch["labels"].to(device, non_blocking=True)
                if gt.ndim == 4: gt = gt.squeeze(1)
            else:
                gt = batch["original_depths"].to(device, non_blocking=True)
                if gt.ndim == 4: gt = gt.squeeze(1)

            with torch.amp.autocast(device):
                pred_final = model(images)["predicted_depth"]

                if use_tta:
                    images_flipped = batch["pixel_values_flip"].to(device, non_blocking=True)
                    pred_flipped = model(images_flipped)["predicted_depth"]
                    pred_flipped_back = torch.flip(pred_flipped, dims=[-1])
                    

                    pred_final = (pred_final + pred_flipped_back) / 2.0

            if pred_final.ndim == 3:
                pred_final = pred_final.unsqueeze(1)

            if mode == "sun_new":
                # sunrgbd has different size images so we scale to those
                for p, g in zip(pred_final, gt):
                    p, g = p.unsqueeze(0), g.unsqueeze(0)
                    
                    if p.shape[-2:] != g.shape[-2:]:
                        p = F.interpolate(
                            p, 
                            size=g.shape[-2:], 
                            mode='bilinear', 
                            align_corners=False
                        )
                    metric.update(p.squeeze(1), g.squeeze(1))
            else:
                # Standard batch-wise update for NYU or old_sun
                metric.update(pred_final, gt)

    metrics = metric.compute()
    return metrics


if __name__ == "__main__":
    from depth_estimation.mrf_model import DINOv2DepthEstimationMRF
    from depth_estimation.dataset.NYU import get_nyu_dataloader
    from depth_estimation.dataset.SUNRGBD import get_sunrgb_dataloader


    paths = ["./res/pathnamebase/mrf_05/[0.5]/mrf_0.5.pth",
             "./res/pathnamebase/mrf_10/[1.0]/mrf_1.0.pth",
             "./res/pathnamebase/mrf_15/[1.5]/mrf_1.5.pth",
             "./res/pathnamebase/mrf_05_10_15/[0.5, 1.0, 1.5]/mrf_0.5_1.0_1.5.pth"]
    
    test_dataloader = get_sunrgb_dataloader(
            split='test',
            root_dir='../datasets',
            # root_dir='./sun',
            split_file='./depth_estimation/SUNRGBD_val_splits.txt',
            batch_size=2,
            shuffle=False)

    for model_path in paths:
        print('running test')
        ckpt = torch.load(model_path, weights_only=False)
        head_state_dict = ckpt['head_state_dict']
        scales = ckpt['scales']
        out_indices = ckpt['out_indices']

        model = DINOv2DepthEstimationMRF.from_pretrained(
            'facebook/dinov2-base',
            scales=scales,
            out_indices=out_indices,
            norm_strategy="nonlinear",
        )

        model.depth_estimation_head.load_state_dict(head_state_dict)

        new_res = test(model, test_dataloader, use_tta=True, mode="sun_old")

        print(new_res)


