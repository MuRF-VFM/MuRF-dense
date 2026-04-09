"""
Training script for training with CLA on VOC2012 dataset.

"""

from .train_mrf_voc import train_mrf
from .train_voc import train

from .model import DinoV2SegmentationModel, DinoV2SegmentationModelMRF

def main(resolutions, root_dir, max_iters, val_interval, batch_size, checkpoint_path):
    if resolutions == [0]:
        # train a default model
        model = DinoV2SegmentationModel(num_classes=21)
        train(
            model=model,
            root_dir=root_dir,
            max_iters=max_iters,
            val_interval=val_interval,
            batch_size=batch_size,
            checkpoint_path=checkpoint_path
        )
    else:
        # train a MRF model with given resolutions
        model = DinoV2SegmentationModelMRF(num_classes=21, resolutions=resolutions)
        train_mrf(
            model=model,
            root_dir=root_dir,
            max_iters=max_iters,
            val_interval=val_interval,
            batch_size=batch_size,
            checkpoint_path=checkpoint_path
        )

if __name__ == '__main__':
    # take clas for resolutions
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--res',
        type=str,
        nargs='+',
        default='0',
        help='List of resolutions for MRF model. Use [0] for default model.'
    )
    args = parser.parse_args()
    resolution = [int(res) for res in args.res[0].split(',')]
    root_dir = '/home/ubuntu/coding/datasets/VOC2012'
    checkpoint_path = './checkpoints/cla_voc'
    max_iters = 6000
    val_interval = 1000
    batch_size = 16

    main(resolution, root_dir, max_iters, val_interval, batch_size, checkpoint_path)