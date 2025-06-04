import torch
import learn2learn as l2l

from scipy.stats import truncnorm

class CifarCNN(torch.nn.Module):

    def __init__(self, output_size=5, hidden_size=64, layers=4):

        super(CifarCNN, self).__init__()
        self.hidden_size = hidden_size
        self.base = l2l.vision.models.ConvBase(output_size=hidden_size,
                             hidden=hidden_size,
                             channels=3,
                             max_pool=True,
                             layers=layers)

        self.features = torch.nn.Sequential(
            l2l.nn.Lambda(lambda x: x.view(-1, 3, 32, 32)),
            self.base,
            l2l.nn.Lambda(lambda x: x.view(-1, 256))
        )
        self.classifier = torch.nn.Linear(256, output_size, bias=True)
        self.classifier.weight.data.normal_()
        self.classifier.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x