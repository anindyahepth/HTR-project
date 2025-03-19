import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18_2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet18_2d, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)

        # Modify the first convolutional layer
        self.resnet18.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Adjusted kernel and stride

        # Remove max pooling
        self.resnet18.maxpool = nn.Identity()

        # Modify the stride of the first layer of each residual block
        for layer in [self.resnet18.layer1, self.resnet18.layer2, self.resnet18.layer3, self.resnet18.layer4]:
            for module in layer.modules():
                if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                    module.stride = (1, 1)

        # Remove the final fully connected layer
        self.resnet18.fc = nn.Identity()

        # Add a convolutional layer to map to the desired output channels
        self.final_conv = nn.Conv2d(512, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.final_conv(x)
        return x

# if __name__ == '__main__':
#     # Example usage:
#     in_channels = 768
#     out_channels = 768
#     batch_size = 50
#     height = 1
#     width = 128

#     model = ResNet18TwoDimensional(in_channels, out_channels)
#     input_tensor = torch.randn(batch_size, in_channels, height, width)
#     output_tensor = model(input_tensor)

#     print("Input shape:", input_tensor.shape)
#     print("Output shape:", output_tensor.shape)
