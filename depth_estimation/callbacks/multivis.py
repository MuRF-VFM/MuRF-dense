import os

import matplotlib.pyplot as plt
import torch

from ops import fill_depth_colorization


class VisualizeMultiDepthMap:
    """
    Callback to visualize depth maps during with multiple comparison models.
    """
    def __init__(self,
                 output_dir: str,
                 models: list[torch.nn.Module],
                 model_names: list[str],
                 frequency: int = 1
                 ):
        self.output_dir = output_dir
        self.models = models
        self.model_names = model_names
        if self.model_names is None or len(self.model_names) != len(self.models):
            self.model_names = [f"model_{i}" for i in range(len(self.models))]
        self.frequency = frequency
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
    def __call__(self, epoch: int, example: dict):
        """
        Creates and saves a visualization comparing multiple model predictions.
        Each row corresponds to one image in the batch.
        Each column corresponds to one model's prediction (plus the input image).
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if epoch % self.frequency != 0:
            return

        image_batches = example["pixel_values"].to(device)
        batch_size = image_batches.size(0)

        num_models = len(self.models)
        num_cols = num_models + 1  # +1 for the input image column

        # Prepare all model predictions
        pred_depths_all = []
        for model in self.models:
            model.eval()
            model.to(device)
            outputs = model(image_batches)
            pred_depths_all.append(outputs["predicted_depth"].cpu())

        fig, axes = plt.subplots(batch_size, num_cols,
                                 figsize=(4 * num_cols, 4 * batch_size))
        if batch_size == 1:
            axes = axes[None, :]
        if num_cols == 1:
            axes = axes[:, None]

        for i in range(batch_size):
            # input
            img = self._denormalize(image_batches[i].cpu())
            img = img.permute(1, 2, 0)
            axes[i, 0].imshow(img)
            if i == 0:
                axes[i, 0].set_title("Input", fontsize=35)

            # prediction visualization per model
            for j, (pred_depth, model_name) in enumerate(
                    zip(pred_depths_all, self.model_names), start=1):
                axes[i, j].imshow(pred_depth[i], cmap="plasma_r")
                if i == 0:
                    axes[i, j].set_title(model_name, fontsize=35)

        for ax_row in axes:
            for ax in ax_row:
                ax.axis("off")

        plt.tight_layout()
        filename = os.path.join(self.output_dir, f"epoch_{epoch}_comparison.pdf")
        plt.savefig(filename)
        plt.close()


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
