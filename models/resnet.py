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
            self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.coord_conv(x)
        x = self.feature_extractor(x)
        assert len(x.values()) == 1
        x = list(x.values())[0]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.fc is not None:
            x = self.fc(x)
            x = self.norm(x)
        return x

class ResnetClassificationModel(nn.Module):
    def __init__(self, args, types=1) -> None:
        super().__init__()
        self.types = types
        self.encoder = make_encoder(args)
        self.fused_feature_dim = args.feature_dim * self.types
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_feature_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, args.num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, visual, tactile):
        # x: B x T_ x C x H x W
        if visual.ndim == 5 and tactile.ndim == 5:
            x = torch.concat([visual, tactile], dim=1)
        elif visual.ndim == 5 and tactile.ndim !=5:
            x = visual
        elif visual.ndim !=5 and tactile.ndim == 5:
            x = tactile
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x_feats = self.encoder(x).reshape(B, T, -1)

        # fusion and prediction
        feats = x_feats.reshape(B, -1)
        preds = self.classifier(feats)
        return None, preds



if __name__ == "__main__":
    """ test """
    cfg = OmegaConf.load('/home/wutong/visual-tactile/configs/resnet/resnet.yaml')
    resnet_classifier = ResnetClassificationModel(cfg, types=4)
    x = torch.ones((32,2,3,128,128))
    y = torch.ones((32,2))
    _, predict = resnet_classifier(x, x)
    print(predict.shape)