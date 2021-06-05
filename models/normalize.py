import torch
import torch.nn as nn
import torch.nn.functional as F


class RAIN(nn.Module):
    def __init__(self, dims_in, eps=1e-5):
        '''Compute the instance normalization within only the background region, in which
            the mean and standard variance are measured from the features in background region.
        '''
        super(RAIN, self).__init__()
        self.foreground_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.eps = eps

    def forward(self, x, mask):
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')
        mean_back, std_back = self.get_foreground_mean_std(x * (1-mask), 1 - mask) # the background features
        normalized = (x - mean_back) / std_back
        normalized_background = (normalized * (1 + self.background_gamma[None, :, None, None]) +
                                 self.background_beta[None, :, None, None]) * (1 - mask)

        mean_fore, std_fore = self.get_foreground_mean_std(x * mask, mask) # the background features
        normalized = (x - mean_fore) / std_fore * std_back + mean_back
        normalized_foreground = (normalized * (1 + self.foreground_gamma[None, :, None, None]) +
                                self.foreground_beta[None, :, None, None]) * mask

        return normalized_foreground + normalized_background

    def get_foreground_mean_std(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])     # (B, C)
        num = torch.sum(mask, dim=[2, 3])       # (B, C)
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]
        var = torch.sum((region + (1 - mask)*mean - mean) ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]
        return mean, torch.sqrt(var+self.eps)
