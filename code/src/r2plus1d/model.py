
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class r2plus1d_18(nn.Module):
    def __init__(self, pretrained=True, num_classes=500):
        super(r2plus1d_18, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        model = torchvision.models.video.r2plus1d_18(pretrained=self.pretrained)
        # delete the last fc layer
        modules = list(model.children())[:-1]
        # print(modules)
        self.r2plus1d_18 = nn.Sequential(*modules)
        self.fc1 = nn.Linear(model.fc.in_features, self.num_classes)

    def forward(self, x, dummy=None):
        out = self.r2plus1d_18(x)
        # print(out.shape)
        # Flatten the layer to fc
        out = out.flatten(1)
        out = self.fc1(out)

        return out


class r2plus1d_18_internal(nn.Module):
    def __init__(self, pretrained=True, num_classes=500):
        super(r2plus1d_18_internal, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        model = torchvision.models.video.r2plus1d_18(pretrained=self.pretrained)
        # delete the last fc layer
        modules = list(model.children())[:-1]
        # print(modules)
        self.r2plus1d_18 = nn.Sequential(*modules)
        self.fc1 = nn.Linear(model.fc.in_features, self.num_classes)

    def forward(self, x, dummy=None):
        outputs = []
        for name, layer in self.r2plus1d_18.named_children():
            x = layer(x)
            if name in ['0', '1', '2', '3', '4']:
                outputs.append(x)

        #out = self.r2plus1d_18(x)
        # print(out.shape)
        # Flatten the layer to fc
        out = x.flatten(1)
        out = self.fc1(out)
        outputs.append(out)

        return outputs