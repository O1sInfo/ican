import torch.nn as nn
import torch.nn.functional as F

from ican import ICAN
from resnet import resnet50

class HOIDetector(nn.Module):
    def __init__(self):
        self.backbone = nn.Sequential(*list(resnet50().modules())[:121])
        self.interaction_pattern = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
        )
        self.hcan = ICAN()
        self.ocan = ICAN()
        self.cls_hc = nn.Sequential(
            nn.Linear(in_features=3072, out_features=2048, bias=True),
            F.dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            F.dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=29, bias=True),
        )
        self.cls_oc = nn.Sequential(
            nn.Linear(in_features=3072, out_features=2048, bias=True),
            F.dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            F.dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=29, bias=True),
        )
        self.cls_sp = nn.Sequential(
            nn.Linear(in_features=7456, out_features=2048, bias=True),
            F.dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            F.dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=29, bias=True),
        )
    
    def forward(image, h_bbox, o_bbox, ho_pattern):
        base_feat = self.backbone(image)
        sp_feat = self.interaction_pattern(ho_pattern)
        hc_feat = self.hcan(base_feat, h_bbox)
        oc_feat = self.ocan(base_feat, o_bbox)
        sp_feat = sp_feat.view(sp_feat.size(0), -1, 1, 1)
        sp_feat = torch.cat((sp_feat, hc_feat[:, :2048, :, :]), dim=1)
        hc_score = self.cls_hc(hc_feat)
        oc_score = self.cls_hc(oc_feat)
        sp_score = self.cls_hc(sp_feat)
        return hc_score, oc_score, sp_score










