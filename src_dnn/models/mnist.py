from torch import Tensor, argmax, flatten, nn


class mnist_rtg_cl_v0(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_shape = (1, 28, 28)

        self.model = nn.Sequential(
            nn.Linear(784, 10)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)
