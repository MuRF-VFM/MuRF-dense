"""
Dinov2 Depth Estimation Model

    From the paper DINOv2 Paper:

    lin. 1: we extract
    the last layer of the frozen transformer and concatenate the [CLS] token to each patch token. Then we
    bi-linearly upsample the tokens by a factor of 4 to increase the resolution. Finally we train_mrf a simple linear
    layer using a classification loss by dividing the depth prediction range in 256 uniformly distributed bins and
    use a linear normalization following Bhat et al.

Hopefully this is a faithful implementation of the above description.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model, Dinov2PreTrainedModel


class DepthEstimationHeadLin1(torch.nn.Module):
    """
    Below is what I hope is a faithful implementation of this description.
    """
    def __init__(self,
                 embed_dim,
                 num_bins=256,
                 depth_range=(1e-3, 10),
                 upsample_factor=4,
                 patch_size=14,
                 norm_strategy="linear"):
        super().__init__()
        self.num_bins = num_bins
        self.norm_strategy = norm_strategy
        self._setup_bins(depth_range)

        self.upsample_factor = upsample_factor
        self.patch_size = patch_size

        # bathcnorm the tokens before conv layer
        # self.norm = nn.BatchNorm2d(2 * embed_dim)
        self.head = nn.Conv2d(2 * embed_dim, num_bins, kernel_size=1)

    def _setup_bins(self, depth_range):
        """
        Sets up the depth bins and their centers.
        Used for normalization and loss computation.
        """
        self.depth_min, self.depth_max = depth_range
        self.bin_edges = torch.linspace(self.depth_min, self.depth_max,
                                        self.num_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    def forward(self, tokens, H, W):
        """
        Applies the depth estimation head on the token embeddings.
        This depth estimation forward pass does NOT return the full dimensional depth map.
        """
        device = tokens.device
        B, N, D = tokens.shape

        # store original H and W
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size

        # bilinearly upsample the patch tokens
        patch_tokens = tokens.reshape(B, H_patch, W_patch, D)
        patch_tokens = patch_tokens.permute(0, 3, 1, 2)  # B, C, H, W
        patch_tokens = F.interpolate(
            patch_tokens,
            scale_factor=self.upsample_factor,
            mode="bilinear",
            align_corners=False,
        )

        # normalize the patch tokens
        # patch_tokens = self.norm(patch_tokens)

        # apply the linear classifier
        logits = self.head(patch_tokens)

        # normalizes
        if self.norm_strategy == "linear":
            logits = torch.relu(logits)
            eps = 0.1
            logits = logits + eps
            probs = logits / logits.sum(dim=1, keepdim=True)
        else:
            probs = F.softmax(logits, dim=1)

        # averages the depth bins according to the predicted probabilities
        # e.g. centers: [1, 2, 3], probs: [0.1, 0.3, 0.6] -> depth: 2.5
        depth = torch.sum(probs * self.bin_centers.view(1, -1, 1, 1).to(device), dim=1)

        return depth, logits


class DINOv2DepthEstimation(Dinov2PreTrainedModel):
    def __init__(self,
                 config,
                 freeze_encoder=True,
                 norm_strategy="nonlinear",
                 feature_fusion="concat",
                 out_indices=None,
                 **kwargs):
        super().__init__(config)

        if out_indices is None:
            out_indices = getattr(config, "out_indices", [11])

        self.out_indices = out_indices
        self.feature_fusion = feature_fusion

        self.dinov2 = Dinov2Model(config)
        self.depth_estimation_head = DepthEstimationHeadLin1(
            config.hidden_size * len(out_indices) if feature_fusion == "concat" else config.hidden_size,
            norm_strategy=norm_strategy,
            **kwargs)

        if freeze_encoder:
            for param in self.dinov2.parameters():
                param.requires_grad = False
    def forward(self,
                pixel_values: torch.Tensor,
                output_attentions=False):
        # save original size
        original_size = pixel_values.shape[2:]

        # extract features with DINOv2
        with torch.no_grad():
            outputs = self.dinov2(pixel_values,
                                  output_hidden_states=True,
                                  output_attentions=output_attentions)

        # collect multiple hidden states
        selected_hidden = [outputs.hidden_states[i] for i in self.out_indices]

        # concat each layers cls token to each patch token
        prepared_layers = []
        for hidden_state in selected_hidden:
            cls_token = hidden_state[:, 0:1, :]
            patch_tokens = hidden_state[:, 1:, :]

            cls_expanded = cls_token.expand(-1, patch_tokens.shape[1], -1)
            prepared_layer_tokens = torch.cat([patch_tokens, cls_expanded],
                                              dim=-1)
            prepared_layers.append(prepared_layer_tokens)

        # fuse
        if len(prepared_layers) > 1:
            if self.feature_fusion == "concat":
                tokens = torch.cat(prepared_layers, dim=-1)
            elif self.feature_fusion == "mean":
                tokens = torch.stack(prepared_layers, dim=0).mean(dim=0)
            else:
                raise ValueError(
                    f"unknown method ")
        else:
            tokens = prepared_layers[0]

        # apply the depth estimation head
        depth, logits = self.depth_estimation_head(tokens,
                                                   H=pixel_values.shape[2],
                                                   W=pixel_values.shape[3])

        # upscale the depth to the original image size
        upsampled_depth = F.interpolate(depth.unsqueeze(1),
                                        size=original_size,
                                        mode="bilinear",
                                        align_corners=False).squeeze(1)

        return {"predicted_depth": upsampled_depth,
                "normalized_depth": depth,
                "logits": logits}


if __name__ == "__main__":
    from dataset import get_nyu_dataloader

    model = DINOv2DepthEstimation.from_pretrained("facebook/dinov2-base",
                                                  input_image_size=(240, 320))

    train_dataloader = get_nyu_dataloader(
        root_dir='../datasets/nyu',
        split='train_mrf',
        batch_size=2,
        shuffle=True
        )

    example = next(iter(train_dataloader))
    outputs = model(example["resized_images"])