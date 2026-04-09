import os

import matplotlib.pyplot as plt
import torch

from depth_estimation.ops import fill_depth_colorization


class VisualizeDepthMap:
    """
    Callback to visualize depth maps during training.
    """
    def __init__(self,
                 output_dir: str,
                 model: torch.nn.Module,
                 frequency: int = 1,
                 num_images: int = 3):
        self.output_dir = output_dir
        self.model = model
        self.frequency = frequency
        self.num_images = num_images
        os.makedirs(self.output_dir, exist_ok=True)

    def _denormalize(self, image,
                     mean=(123.675, 116.28, 103.53),
                     std=(58.395, 57.12, 57.375)):
        """
        Undo normalization applied in dataset pipeline.
        """
        mean = torch.tensor(mean, device=image.device).view(3, 1, 1)
        std = torch.tensor(std, device=image.device).view(3, 1, 1)

        # undo normalization
        img = image * std + mean
        img = img / 255.0
        return img.clamp(0, 1)

    @torch.no_grad()
    def __call__(self,
                 epoch: int,
                 example: dict,
                 detailed_image=True):
        """
        Creates and saves a visualization of the input, ground truth,
        and predicted depth maps.

        Args:
            epoch (int): Current epoch number.
            example (dict): A batch of examples containing 'pixel_values' and 'labels'.
            detailed_image (bool): If True, includes filled depth map in visualization.
        """
        if epoch % self.frequency != 0:
            return

        image_batches = example["pixel_values"]
        gt_depth_batches = example["labels"]

        self.model.eval()
        outputs = self.model(image_batches)
        pred_depths = outputs["predicted_depth"]

        for i in range(min(self.num_images, image_batches.size(0))):
            if detailed_image:
                img = image_batches[i].cpu()
                img = self._denormalize(img)
                img = img.permute(1, 2, 0)

                depth = gt_depth_batches[i].squeeze(0).cpu()
                depth_filled = fill_depth_colorization(
                    imgRgb=img.numpy(),
                    imgDepthInput=depth.numpy(), alpha=1)
                pred_depth = pred_depths[i].cpu()

                # plot the four images into a 4 row grid
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                axes[0, 0].imshow(img)
                axes[0, 0].set_title("Original Image")
                axes[0, 1].imshow(depth, cmap="plasma")
                axes[0, 1].set_title("Ground Truth Depth")
                axes[1, 0].imshow(depth_filled, cmap="plasma")
                axes[1, 0].set_title("Filled Ground Truth Depth")
                axes[1, 1].imshow(pred_depth, cmap="plasma")
                axes[1, 1].set_title("Predicted Depth")

                for ax in axes.flatten():
                    ax.axis("off")
                plt.tight_layout()

                filepath_name = os.path.join(self.output_dir,
                                             f"epoch_{epoch}_img_{i}.png")
                while os.path.exists(filepath_name):
                    filepath_name = filepath_name.replace(".png", "_1.png")

                plt.savefig(os.path.join(filepath_name))
                plt.close()

            else:
                img = image_batches[i].cpu()
                img = self._denormalize(img)
                img = img.permute(1, 2, 0)

                depth = gt_depth_batches[i].squeeze(0).cpu()
                pred_depth = pred_depths[i].cpu()

                # plot the four images into a 3 row grid
                fig, axes = plt.subplots(1, 3, figsize=(10, 10))
                axes[0, 0].imshow(img)
                axes[0, 0].set_title("Original Image")
                axes[0, 1].imshow(depth, cmap="plasma")
                axes[0, 1].set_title("Ground Truth Depth")
                axes[1, 1].imshow(pred_depth, cmap="plasma")
                axes[1, 1].set_title("Predicted Depth")

                for ax in axes.flatten():
                    ax.axis("off")
                plt.tight_layout()

                filepath_name = os.path.join(self.output_dir,
                                             f"epoch_{epoch}_img_{i}.png")
                while os.path.exists(filepath_name):
                    filepath_name = filepath_name.replace(".png", "_1.png")

                plt.savefig(os.path.join(filepath_name))
                plt.close()

        self.model.train()

if __name__ == "__main__":
    from ..dino_model import DINOv2DepthEstimation
    from ..dataset import get_nyu_dataloader
    from pathlib import Path

    print(Path.cwd() / 'checkpoints')
    train_dataloader = get_nyu_dataloader(
        root_dir='/nobackup/shared/datasets/nyu',
        split='train_mrf',
        batch_size=16,
        shuffle=True)
    example = next(iter(train_dataloader))
    model = DINOv2DepthEstimation.from_pretrained("facebook/dinov2-base")
    callback = VisualizeDepthMap(output_dir="images", model=model, frequency=1)
    callback(epoch=1, example=example)

