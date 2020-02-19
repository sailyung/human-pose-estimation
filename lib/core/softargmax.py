import torch
import torch.nn as nn
from torch.nn import functional as F


class SoftArgmax2D(nn.Module):
    """
    Creates a module that computes Soft-Argmax 2D of a given input heatmap.
    Returns the index of the maximum 2d coordinates of the give map.
    :param beta: The smoothing parameter.
    :param return_xy: The output order is [x, y].
    """

    def __init__(self, beta: int = 100, return_xy: bool = False):
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta: {beta}")
        super().__init__()
        self.beta = beta
        self.return_xy = return_xy

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        :param heatmap: The input heatmap is of size B x N x H x W.
        :return: The index of the maximum 2d coordinates is of size B x N x 2.
        """
        heatmap = heatmap.mul(self.beta)
        batch_size, num_channel, height, width = heatmap.size()
        device: str = heatmap.device

        softmax: torch.Tensor = F.softmax(
            heatmap.view(batch_size, num_channel, height * width), dim=2
        ).view(batch_size, num_channel, height, width)

        xx, yy = torch.meshgrid(list(map(torch.arange, [height, width])))

        approx_x = (
            softmax.mul(xx.float().to(device))
                .view(batch_size, num_channel, height * width)
                .sum(2)
                .unsqueeze(2)
        )
        approx_y = (
            softmax.mul(yy.float().to(device))
                .view(batch_size, num_channel, height * width)
                .sum(2)
                .unsqueeze(2)
        )

        output = [approx_x, approx_y] if self.return_xy else [approx_y, approx_x]
        output = torch.cat(output, 2)
        return output
