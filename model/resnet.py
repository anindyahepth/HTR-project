import torch
import torch.nn as nn
import timm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

#------Final Feature Extractor ----------

#The input image shape should be (3, 64, 1024)

class ResNet50_custom_feat_ex(nn.Module):
    def __init__(self, embed_dim=1024, pretrained = True):
        super().__init__()
        if pretrained:
          self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True)
          for param in self.backbone.parameters():
            param.requires_grad = False
        else:
          self.backbone = timm.create_model('resnet50', pretrained=False, features_only=True)


        self.maxpool = nn.MaxPool2d(kernel_size= 3, stride= (2,1), padding=1)
      



    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.layer1(x)
        x = self.maxpool(x)
        x = self.backbone.layer2(x)
        x = self.maxpool(x)
        x = self.backbone.layer3(x)
        
        output = self.maxpool(x)

        return output

#------ResNet18----------

class ResNet18_feat_ex(nn.Module):
    def __init__(self, embed_dim=256, pretrained = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.pretrained = pretrained
        if pretrained:
          self.backbone = timm.create_model('resnet18.a1_in1k', pretrained=True, features_only=True)
          for param in self.backbone.parameters():
            param.requires_grad = False
        else:
          self.backbone = timm.create_model('resnet18.a1_in1k', pretrained=False, features_only=True)

        # Input channels: 128 (from ResNet18 Layer 2)
        # Output channels: embed_dim (e.g., 768)
        self.feature_modifier_conv = nn.Sequential(
            nn.Conv2d(128, embed_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=(8,1), stride=1, padding=0)
        



    def forward(self, x):
        feature_maps = self.backbone(x)
        features_from_layer_2 = feature_maps[2] # Shape: (Batch, 128, H, W)
        modified_features = self.feature_modifier_conv(features_from_layer_2) # Shape: (Batch, new_feature_dim, H, W)
        

        output = self.maxpool(modified_features)

        return output

#------ResNet50----------

#The input image shape should be (3, 64, 1024)

class ResNet50_feat_ex(nn.Module):
    def __init__(self, embed_dim=512, pretrained = True):
        super().__init__()
        if pretrained:
          self.backbone = timm.create_model('resnet50.a1_in1k', pretrained=True, features_only=True)
          # for param in self.backbone.parameters():
          #   param.requires_grad = False
        else:
          self.backbone = timm.create_model('resnet50.a1_in1k', pretrained=False, features_only=True)

        #Changing feature_dim to desired value
        self.feature_modifier_conv = nn.Sequential(
            nn.Conv2d(512, embed_dim, kernel_size= 1, stride=1, padding=0),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=(8,1), stride=1, padding=0)
        
        

    def forward(self, x):
        feature_maps = self.backbone(x)
        features_from_layer_2 = feature_maps[2] # Shape: (Batch, 512, H, W)
        modified_features = self.feature_modifier_conv(features_from_layer_2) # Shape: (Batch, new_feature_dim, h, w)

        
        output = self.maxpool(modified_features)

        return output


#--------Older ResNet18-------------------------

def conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self, nb_feat = 384):
        
        self.inplanes = nb_feat // 4
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, nb_feat // 4, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nb_feat // 4, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)
        self.layer1 = self._make_layer(BasicBlock, nb_feat // 4, 2, stride=(2, 1))
        self.layer2 = self._make_layer(BasicBlock, nb_feat // 2, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, nb_feat, 2, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        
        return x
