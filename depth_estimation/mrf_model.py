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

import math
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
                 scales,
                 num_bins=256,
                 patch_size=14):
        super().__init__()
        self.scales = scales
        self.num_bins = num_bins
        self.patch_size = patch_size

        # depth estimation head for multiple resolutions
        self.head = nn.Conv2d(2 * embed_dim * len(scales), num_bins, kernel_size=1)

    def forward(self, tokens):
        """
        Applies the depth estimation head on the token embeddings.
        This depth estimation forward pass does NOT return the full dimensional depth map.
        """
        # apply the linear classifier
        logits = self.head(tokens)
        return logits


class DINOv2DepthEstimationMRF(Dinov2PreTrainedModel):
    def __init__(self,
                 config,
                 freeze_encoder=True,
                 upsample_factor=1,
                 num_bins=256,
                 depth_range=(1e-3, 10),
                 scales: tuple[float]=(0.5, 1.0),
                 out_indices=None,
                 norm_strategy="nonlinear",
                 feature_fusion="concat",
                 **kwargs):
        super().__init__(config)
        self.feature_fusion = feature_fusion

        if isinstance(scales, int) or isinstance(scales, float):
            self.scales = [scales]
        else:
            self.scales = list(scales)
        if out_indices is None:
            out_indices = getattr(config, "out_indices", [11])

        self.upsample_factor = upsample_factor
        self.patch_size = config.patch_size
        self.norm_strategy = norm_strategy
        self.num_bins = num_bins
        self.out_indices = out_indices
        self._setup_bins(depth_range)

        self.dinov2 = Dinov2Model(config)
        self.depth_estimation_head = DepthEstimationHeadLin1(
            embed_dim=config.hidden_size * len(out_indices),
            num_bins=num_bins,
            scales=self.scales,
            patch_size=self.patch_size,
        )

        if freeze_encoder:
            for param in self.dinov2.parameters():
                param.requires_grad = False

    def _setup_bins(self, depth_range):
        """
        Sets up the depth bins and their centers.
        Used for normalization and loss computation.
        """
        self.depth_min, self.depth_max = depth_range
        self.bin_edges = torch.linspace(self.depth_min, self.depth_max,
                                        self.num_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    def _features_from_backbone(self, x, H_patch, W_patch, B,
                                D) -> torch.Tensor:
        """
        Extract features from multiple transformer layers (self.out_indices).
        For each layer, concatenate [CLS] token with patch tokens, then upsample.
        Finally, concatenate all layer features along the channel dimension.
        """
        outputs = self.dinov2(x, output_hidden_states=True)
        layer_feats = []

        for idx in self.out_indices:
            hidden = outputs['hidden_states'][idx]

            # separate CLS and patch tokens
            cls_token, patch_tokens = hidden[:, 0:1, :], hidden[:, 1:, :]

            # concatenate CLS token to each patch token
            cls_expanded = cls_token.expand(-1, patch_tokens.shape[1], -1)
            patch_tokens = torch.cat([patch_tokens, cls_expanded], dim=-1)

            # reshape to (B, C, H, W)
            patch_tokens = patch_tokens.reshape(B, H_patch, W_patch, 2 * D)
            patch_tokens = patch_tokens.permute(0, 3, 1, 2)  # B, C, H, W

            # upsample for this layer
            patch_tokens = F.interpolate(
                patch_tokens,
                scale_factor=self.upsample_factor,
                mode="bilinear",
                align_corners=False,
            )

            layer_feats.append(patch_tokens)

        # concatenate features from all chosen layers
        return torch.cat(layer_feats, dim=1)


    def forward(self,
                pixel_values: torch.Tensor,
                output_attentions=False):
        device = pixel_values.device
        # save original size
        original_size = pixel_values.shape[2:]

        feats = []
        feat_sizes = []
        for scale in self.scales:
            scaled_height = math.ceil(original_size[0] * scale)
            scaled_width = math.ceil(original_size[1] * scale)

            if scale == 1:
                x_res = pixel_values
            else:
                # use bilinear down/up sampling for continuous images
                x_res = F.interpolate(pixel_values,
                                      size=(scaled_height,
                                            scaled_width),
                                      mode='bilinear',
                                      align_corners=False)

            H_patch = scaled_height // self.patch_size
            W_patch = scaled_width // self.patch_size
            B, C, H, W = x_res.shape
            D = self.dinov2.config.hidden_size

            patch_tokens = self._features_from_backbone(x_res, H_patch, W_patch, B, D)

            feats.append(patch_tokens)
            feat_sizes.append(patch_tokens.shape[-2:])

        # Upsample all features to the largest spatial size among them before concatenation
        max_h = max(s[0] for s in feat_sizes)
        max_w = max(s[1] for s in feat_sizes)

        feats_upsampled = []
        for f in feats:
            f_up = F.interpolate(f, size=(max_h, max_w), mode='bilinear', align_corners=False)
            feats_upsampled.append(f_up)

        # Concatenate along channel axis
        x_cat = torch.cat(feats_upsampled, dim=1)  # (B, 2 * embed_dim * num_res, max_h, max_w)

        # apply depth estimation head
        logits = self.depth_estimation_head(x_cat)

        # norm strategy for probabilities
        if self.norm_strategy == "linear":
            logits = torch.relu(logits)
            eps = 0.1
            logits = logits + eps
            probs = logits / logits.sum(dim=1, keepdim=True)
        else:
            probs = F.softmax(logits, dim=1)

        # discretizes depth with the bin centers and probabilities
        # e.g. centers: [1, 2, 3], probs: [0.1, 0.3, 0.6] -> depth: 2.5
        depth = torch.sum(probs * self.bin_centers.view(1, -1, 1, 1).to(device), dim=1)

        # upscale the depth to the original image size
        upsampled_depth = F.interpolate(depth.unsqueeze(1),
                                        size=original_size,
                                        mode="bilinear",
                                        align_corners=False).squeeze(1)

        return {"predicted_depth": upsampled_depth,
                "normalized_depth": depth,
                "logits": logits}


if __name__ == "__main__":
    pass