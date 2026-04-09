"""
CLA training script to automate fast testing.
Was not used for final results.
"""

import os
from depth_estimation.mrf_model import DINOv2DepthEstimationMRF
from depth_estimation.train_mrf import train_mrf

if __name__ == '__main__':
    import argparse

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", required=True)
    parser.add_argument("--model-size", required=True, choices=["small", "base", "large", "giant"])
    parser.add_argument("--lin", choices=["lin1", "lin4", "lin3"], default="lin1")
    parser.add_argument("--max-iters", type=int, default=38400)
    parser.add_argument("--upsample", type=int, default=1)
    args = parser.parse_args()

    # Parse scales
    scales = [float(s) for s in args.scales.split(",")]

    model_sizes = {
        'small': ('facebook/dinov2-small', ([11], [2, 5, 8, 11], [3, 8, 11])),
        'base': ('facebook/dinov2-base', ([11], [2, 5, 8, 11], [3, 7, 11])),
        'large': ('facebook/dinov2-large', ([23], [4, 11, 17, 23], [4, 11, 23])),
        'giant': ('facebook/dinov2-giant', ([39], [9, 19, 29, 39], [9, 19, 39]))
    }

    model_id, out_indices_all = model_sizes[args.model_size]
    if args.lin == 'lin1':
        out_indices = out_indices_all[0]
    elif args.lin == 'lin4':
        out_indices = out_indices_all[1]
    else:  # lin3
        out_indices = out_indices_all[2]
    model_size_name = args.model_size + ('' if args.lin == 'lin1' else '_lin4')

    # mrf model
    model = DINOv2DepthEstimationMRF.from_pretrained(
        model_id,
        scales=scales,
        out_indices=out_indices,
        norm_strategy="nonlinear",
        upsample_factor=args.upsample
    )

    checkpoint_path = (
        f'test'
        f'{model_size_name}/mrf_{"_".join([str(s).replace(".", "") for s in scales])}'
    )
    os.makedirs(checkpoint_path, exist_ok=True)

    print(f'Going to train_mrf on {scales} {model_size_name} model, saving to {checkpoint_path}')
    train_mrf(
        # root_dir='./nyu_depth_v2/nyu',
        root_dir = '/nobackup/shared/datasets/nyu',
        checkpoint_path=checkpoint_path,
        model=model,
        ckpt_interval=38400,
        val_interval=10_000,
        sun_test=True,
        max_iters=args.max_iters
    )
    print(f'Finished to train_mrf on {scales} {model_size_name} model, saving to {checkpoint_path}')
