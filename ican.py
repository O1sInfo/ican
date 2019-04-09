import torch.nn as nn
import torch.nn.functional as F

from roi_pool import RoiPool
from resnet import resnet50

class ICAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.roi_pooling = RoiPool(output_size=7, spatial_scale=1.0/8.0)
        self.res5 = resnet50().layer4
        self.gap1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(in_features=2048, out_features=512, bias=True)
        self.conv_k = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.gap2 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc2 = nn.Linear(in_features=512, out_features=1024, bias=True)
    
    def forward(self, base_feature, rois):
        h, w = base_feature.size(2), base_feature.size(3)
        feature_map_k = self.conv_k(base_feature)
        feature_map_v = self.conv_v(base_feature)
        instance_feature = self.roi_pooling(base_feature, rois)
        # (num_rois, ch, ph, pw)
        num_rois = instance_feature.size(0)
        instance_feature = self.res5(instance_feature)
        instance_feature = self.gap1(instance_feature)
        instance_q = self.fc1(instance_feature)
        instance_q = instance_q.expand(-1, -1, h, w)
        attention_weight = torch.sum(instance_q.mul(feature_map_k), 1)
        attention_weight = torch.softmax(attention_weight.view(num_rois, -1)).view(num_rois, h, w)
        attention_weight = attention_weight.expand(-1, 512, h, w)
        attention_map = feature_map_v.mul(attention_weight)
        context_vector = self.gap2(attention_map)
        context_vector = self.fc2(context_vector)
        assert(context_vector.size() == instance_feature.size())
        feature_vector = torch.cat((instance_feature, context_vector), dim=1)
        return feature_vector









