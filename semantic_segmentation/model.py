import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import AutoModel, Dinov2Model
import math

class SegmentationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(embed_dim)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.bn(x)
        x = self.classifier(x)
        return x

class DinoV2SegmentationModel(nn.Module):
    def __init__(self, num_classes, model_name="facebook/dinov2-base", pretrained=True):
        super().__init__()
        
        # Load DinoV2 backbone from Hugging Face
        self.backbone = Dinov2Model.from_pretrained(model_name)
        
        # Freeze backbone parameters
        if pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.backbone.eval()
            
        # Get embedding dimension
        self.embed_dim = self.backbone.config.hidden_size
        
        # Linear segmentation head
        self.seg_head = SegmentationHead(self.embed_dim, num_classes)

    def forward(self, x):
        # Store original size for upsampling
        original_size = x.shape[2:]
        
        # Extract features from DinoV2
        outputs = self.backbone(x)
        features = outputs.last_hidden_state
        
        # Remove CLS token if present
        features = features[:, 1:]  # Remove CLS token
            
        # Reshape to spatial features
        features = features.reshape(features.shape[0], original_size[0] // 14, original_size[1] // 14, -1).permute(0, 3, 1, 2)

        # Apply segmentation head
        logits = self.seg_head(features)
        
        # Upsample to original image size
        logits = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)
        
        return logits

class DinoV2SegmentationModelMRF(nn.Module):
    def __init__(self, num_classes, model_name="facebook/dinov2-base", pretrained=True, resolutions=(266, 518, 784)):
        super().__init__()
        
        # Resolutions we'll run the backbone at (list or tuple)
        self.resolutions = list(resolutions)
        assert len(self.resolutions) >= 1, "Need at least one resolution"

        # Load DinoV2 backbone from Hugging Face
        self.backbone = Dinov2Model.from_pretrained(model_name)
        
        # Freeze backbone parameters if pretrained
        if pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
        # Embedding dimension (hidden_size)
        self.embed_dim = self.backbone.config.hidden_size
        
        # Segmentation head input channels = embed_dim * number_of_resolutions
        concat_channels = self.embed_dim * len(self.resolutions)
        self.seg_head = SegmentationHead(concat_channels, num_classes)

    def _features_from_backbone(self, x):
        """
        Run backbone and reshape last_hidden_state (remove cls) into (B, C, H, W).
        x: tensor (B, 3, H, W)
        returns: feature map (B, embed_dim, H_feat, W_feat)
        """
        outputs = self.backbone(x)
        # last_hidden_state shape: (B, seq_len, embed_dim)
        features = outputs.last_hidden_state

        # If model has a CLS token, commonly it's first token. Remove it if seq_len is not a perfect square.
        # Many ViTs include a CLS token; we handle both cases robustly.
        seq_len = features.shape[1]
        # Try removing first token when seq_len isn't a perfect square (i.e., there's a CLS)
        spatial_len = int(round(math.sqrt(seq_len)))
        if spatial_len * spatial_len != seq_len:
            # assume first token is CLS
            features = features[:, 1:, :]
            seq_len = features.shape[1]
            spatial_len = int(round(math.sqrt(seq_len)))

        # Now reshape to (B, embed_dim, H_feat, W_feat)
        B = features.shape[0]
        embed = features.shape[2]
        H_feat = spatial_len
        W_feat = spatial_len
        feat = features.reshape(B, H_feat, W_feat, embed).permute(0, 3, 1, 2).contiguous()
        return feat  # (B, embed_dim, H_feat, W_feat)

    def forward(self, x):
        """
        x: input images tensor (B, 3, H, W). Typically H=W=crop_size (784),
           but can be any spatial size. Model will internally create resized versions
           at the resolutions specified in self.resolutions.
        """
        original_size = x.shape[2:]  # (H, W)

        feats = []
        feat_sizes = []
        # compute features for each resolution
        for res in self.resolutions:
            # Resize input to (res, res)
            if (x.shape[2], x.shape[3]) == (res, res):
                x_res = x
            else:
                # use bilinear down/up sampling for continuous images
                x_res = F.interpolate(x, size=(res, res), mode='bilinear', align_corners=False)

            # Backbone expects pixel_values ordering that was used previously; pass x_res directly as before
            feat = self._features_from_backbone(x_res)  # (B, embed_dim, Hf, Wf)
            feats.append(feat)
            feat_sizes.append(feat.shape[-2:])  # (Hf, Wf)

        # Upsample all features to the largest spatial size among them before concatenation
        max_h = max(s[0] for s in feat_sizes)
        max_w = max(s[1] for s in feat_sizes)

        feats_upsampled = []
        for f in feats:
            if (f.shape[2], f.shape[3]) == (max_h, max_w):
                f_up = f
            else:
                f_up = F.interpolate(f, size=(max_h, max_w), mode='bilinear', align_corners=False)
            feats_upsampled.append(f_up)

        # Concatenate along channel axis
        x_cat = torch.cat(feats_upsampled, dim=1)  # (B, embed_dim * num_res, max_h, max_w)

        # Pass through segmentation head (Conv + BN)
        logits = self.seg_head(x_cat)  # (B, num_classes, max_h, max_w)

        # Upsample logits to original input image size
        logits = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)

        return logits

class Siglip2SegmentationModel(nn.Module):
    def __init__(self, num_classes, model_name="google/siglip2-base-patch16-naflex", pretrained=True):
        super().__init__()
        
        # Load SigLip2 backbone from Hugging Face
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Freeze backbone parameters
        if pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
        # Get embedding dimension
        self.embed_dim = self.backbone.config.vision_config.hidden_size
        
        # Linear segmentation head
        self.seg_head = SegmentationHead(self.embed_dim, num_classes)

    def forward(self, x, original_size):
        # Store original size for upsampling
        
        # Extract features from SigLip2
        original_size = torch.tensor(original_size)
        outputs = self.backbone.vision_model(x, spatial_shapes=(original_size//16).repeat(x.shape[0], 1), attention_mask=None)
        features = outputs.last_hidden_state
            
        # Reshape to spatial features
        features = features.reshape(features.shape[0], original_size[0] // 16, original_size[1] // 16, -1).permute(0, 3, 1, 2)

        # Apply segmentation head
        logits = self.seg_head(features)
        
        # Upsample to original image size
        logits = F.interpolate(logits, size=(original_size[0].item(), original_size[1].item()), mode='bilinear', align_corners=False)
        
        return logits

class Siglip2SegmentationModelMRF(nn.Module):
    def __init__(self, num_classes, model_name="google/siglip2-base-patch16-naflex", pretrained=True, resolutions=(256, 512, 768)):
        super().__init__()
        
        # Resolutions we'll run the backbone at (list or tuple)
        self.resolutions = list(resolutions)
        assert len(self.resolutions) >= 1, "Need at least one resolution"

        # Load DinoV2 backbone from Hugging Face
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Freeze backbone parameters if pretrained
        if pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
        # Embedding dimension (hidden_size)
        self.embed_dim = self.backbone.config.vision_config.hidden_size
        
        # Segmentation head input channels = embed_dim * number_of_resolutions
        concat_channels = self.embed_dim * len(self.resolutions)
        self.seg_head = SegmentationHead(concat_channels, num_classes)

    def _features_from_backbone(self, x, res):
        """
        Run backbone and reshape last_hidden_state (remove cls) into (B, C, H, W).
        x: tensor (B, 3, H, W)
        returns: feature map (B, embed_dim, H_feat, W_feat)
        """
        outputs = self.backbone.vision_model(x, spatial_shapes=torch.tensor([res//16, res//16]).repeat(x.shape[0], 1), attention_mask=None)
        # last_hidden_state shape: (B, seq_len, embed_dim)
        features = outputs.last_hidden_state

        # If model has a CLS token, commonly it's first token. Remove it if seq_len is not a perfect square.
        # Many ViTs include a CLS token; we handle both cases robustly.
        seq_len = features.shape[1]
        # Try removing first token when seq_len isn't a perfect square (i.e., there's a CLS)
        spatial_len = int(round(math.sqrt(seq_len)))
        if spatial_len * spatial_len != seq_len:
            # assume first token is CLS
            features = features[:, 1:, :]
            seq_len = features.shape[1]
            spatial_len = int(round(math.sqrt(seq_len)))

        # Now reshape to (B, embed_dim, H_feat, W_feat)
        B = features.shape[0]
        embed = features.shape[2]
        H_feat = spatial_len
        W_feat = spatial_len
        feat = features.reshape(B, H_feat, W_feat, embed).permute(0, 3, 1, 2).contiguous()
        return feat  # (B, embed_dim, H_feat, W_feat)

    def forward(self, x1, x2, x3):
        """
        x: input images tensor (B, 3, H, W). Typically H=W=crop_size (784),
           but can be any spatial size. Model will internally create resized versions
           at the resolutions specified in self.resolutions.
        """

        x= [x1, x2, x3]

        feats = []
        feat_sizes = []
        # compute features for each resolution
        for i, res in enumerate(self.resolutions):

            # Backbone expects pixel_values ordering that was used previously; pass x_res directly as before
            feat = self._features_from_backbone(x[i], res)  # (B, embed_dim, Hf, Wf)
            feats.append(feat)
            feat_sizes.append(feat.shape[-2:])  # (Hf, Wf)

        # Upsample all features to the largest spatial size among them before concatenation
        max_h = max(s[0] for s in feat_sizes)
        max_w = max(s[1] for s in feat_sizes)

        feats_upsampled = []
        for f in feats:
            if (f.shape[2], f.shape[3]) == (max_h, max_w):
                f_up = f
            else:
                f_up = F.interpolate(f, size=(max_h, max_w), mode='bilinear', align_corners=False)
            feats_upsampled.append(f_up)

        # Concatenate along channel axis
        x_cat = torch.cat(feats_upsampled, dim=1)  # (B, embed_dim * num_res, max_h, max_w)

        # Pass through segmentation head (Conv + BN)
        logits = self.seg_head(x_cat)  # (B, num_classes, max_h, max_w)

        # Upsample logits to original input image size
        logits = F.interpolate(logits, size=(768, 768), mode='bilinear', align_corners=False)

        return logits
