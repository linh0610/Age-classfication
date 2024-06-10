import torch
import torch.nn as nn
import torchvision.models as models

def set_device():
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    return torch.device(dev)

device =  set_device()
class AgeModel(nn.Module):
    def __init__(self, num_classes):
        super(AgeModel, self).__init__()
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.model.fc = nn.Sequential(nn.Dropout(p=0.5, inplace=True),
                                          nn.Linear(in_features=2048, out_features=256, bias=True),
                                          nn.Linear(in_features=256, out_features=14, bias=True))
        device = set_device()
        self.model = self.model.to(device)
    def forward(self, x):
        x = self.model(x)
        return x
