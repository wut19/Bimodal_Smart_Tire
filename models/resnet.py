"""
    Modified from https://github.com/JunzheJosephZhu/see_hear_feel
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from omegaconf import OmegaConf

def make_encoder(args):
    vision_extractor = resnet50(pretrained=True)
    vision_extractor.conv1 = nn.Conv2d(
        in_channels=args.in_channels+2, 
        out_channels=args.out_channels, 
        kernel_size=args.kernel_size, 
        stride=args.kernel_size, 
        padding=args.padding, 
        bias=args.bias,
    )
    vision_extractor = create_feature_extractor(vision_extractor, ["layer4.1.relu_1"])
    return Encoder(vision_extractor, args.feature_dim)

class CoordConv(nn.Module):
    """Add coordinates in [0,1] to an image, like CoordConv paper."""

    def forward(self, x):
        # B x C x H x W inputs
        assert x.ndim == 4
        h, w = x.shape[2:]
        ones_h = x.new_ones((h, 1))
        type_dev = dict(dtype=x.dtype, device=x.device)
        lin_h = torch.linspace(-1, 1, h, **type_dev)[:, None]
        ones_w = x.new_ones((1, w))
        lin_w = torch.linspace(-1, 1, w, **type_dev)[None, :]
        new_maps_2d = torch.stack((lin_h * ones_w, lin_w * ones_h), dim=0)
        new_maps_4d = new_maps_2d[None]
        assert new_maps_4d.shape == (1, 2, h, w), (x.shape, new_maps_4d.shape)
        batch_size, type = x.shape[:2]
        new_maps_4d_batch = new_maps_4d.repeat(batch_size, 1, 1, 1)
        result = torch.cat((x, new_maps_4d_batch), dim=1)
        return result

class Encoder(nn.Module):
    def __init__(self, feature_extractor, out_dim=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.downsample = nn.MaxPool2d(2, 2)
        self.coord_conv = CoordConv()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if out_dim is not None:
            self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.coord_conv(x)
        x = self.feature_extractor(x)
        assert len(x.values()) == 1
        x = list(x.values())[0]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.fc is not None:
            x = self.fc(x)
        return x

class ResnetClassificationModel(nn.Module):
    def __init__(self, args, visual_types=3, tactile_types=1) -> None:
        super().__init__()
        self.visual_types = visual_types
        self.tactile_types = tactile_types
        self.encoder = make_encoder(args)
        self.fused_feature_dim = args.feature_dim * (self.visual_types + self.tactile_types)
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_feature_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, args.num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, visuals, tactiles):
        # visuals: B x T x C x H x W
        # tactiles: B x T_ x C x H x W

        # visuals
        B, T, C, H, W = visuals.shape
        visuals = visuals.view(B*T, C, H, W)
        visual_feats = self.encoder(visuals).reshape(B, T, -1)

        # tactiles
        B, T, C, H, W = tactiles.shape
        tactiles = tactiles.view(B*T, C, H, W)
        tactiles_feats = self.encoder(tactiles).reshape(B, T, -1)

        # fusion and prediction
        feats = torch.cat([visual_feats, tactiles_feats], dim=1).reshape(B, -1)
        preds = self.classifier(feats)
        return preds



if __name__ == "__main__":
    cfg = OmegaConf.load('/home/wutong/visual-tactile/configs/resnet/resnet.yaml')
    resnet_classifier = ResnetClassificationModel(cfg, visual_types=4, tactile_types=2)
    visuals = torch.zeros((32,4,3,128,128))
    tactiles = torch.ones((32,2,3,128,128))
    predict = resnet_classifier(visuals, tactiles)
    print(predict.shape)