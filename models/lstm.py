"""
    Modified from https://github.com/MaxDu17/PlayItByEar_Code
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
from omegaconf import OmegaConf

def make_encoder(args):
    resnet = resnet18(pretrained=False)
    resnet.conv1 =  nn.Conv2d(args.in_channels, args.out_channels, kernel_size=args.kernel_size, stride=args.stride, padding=args.padding, bias=args.bias)
    resnet.fc = nn.Linear(512, args.feature_dim)
    return resnet

class LSTMClassificationModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.feature_dim = args.feature_dim
        self.patch_size = args.patch_size

        self.proj = make_encoder(args)
        self.norm = nn.LayerNorm(self.feature_dim)
        self.lstm = nn.LSTM(self.feature_dim, self.feature_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, args.num_classes),
            nn.Softmax(dim=-1)
        )
    def forward(self, visual, tactile):
        # x_shape: B x T x C x H x W
        if visual.ndim == 5 and tactile.ndim == 5:
            x = torch.concat([visual, tactile], dim=1)
        elif visual.ndim == 5 and tactile.ndim !=5:
            x = visual
        elif visual.ndim !=5 and tactile.ndim == 5:
            x = tactile
        # patch feature extraction
        B,T,C,H,W = x.shape
        x = x.view(B, T, C, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size)
        x = x.permute(0,1,3,5,2,4,6).reshape(-1, C, self.patch_size, self.patch_size)
        x_feats = self.norm(self.proj(x)).reshape(B, -1, self.feature_dim)

        _, (feats, c) = self.lstm(x_feats)
        feats = feats.squeeze(0)
        preds = self.classifier(feats)
        return None, preds

if __name__ == "__main__":
    """ test """
    cfg = OmegaConf.load('/home/wutong/visual-tactile/configs/lstm/lstm.yaml')
    lstm_classifier = LSTMClassificationModel(args=cfg)
    x = torch.zeros((32,2,3,128,128))
    y = torch.zeros((32,2))
    _, preds = lstm_classifier(x, x)
    print(preds.shape)
