from torch import Tensor, nn, argmax, unsqueeze


class mnist_rtg_v0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_shape = (1, 28, 28)
        self._model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784, out_features=10)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self._model(x)
        return x, argmax(x, dim=1)


class mnist_rtg_v1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_shape = (1, 28, 28)
        self._model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784, out_features=10),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self._model(x)
        return x, argmax(x, dim=1)


class mnist_rtg_v2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_shape = (1, 28, 28)
        self._model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=10),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self._model(x)
        return x, argmax(x, dim=1)


class mnist_rtg_v3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_shape = (1, 28, 28)
        self._model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=96),
            nn.LeakyReLU(),
            nn.Linear(in_features=96, out_features=10),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self._model(x)
        return x, argmax(x, dim=1)


class mnist_rtg_v4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_shape = (1, 28, 28)
        self._model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=96),
            nn.LeakyReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(in_features=96, out_features=10),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self._model(x)
        return x, argmax(x, dim=1)


class mnist_rtg_v5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_shape = (1, 28, 28)
        self._model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=96),
            nn.LeakyReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(in_features=96, out_features=10),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self._model(x)
        return x, argmax(x, dim=1)


class mnist_rtg_v6(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_shape = (1, 28, 28)
        self._model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(in_features=256, out_features=96),
            nn.BatchNorm1d(num_features=96),
            nn.LeakyReLU(),
            nn.Linear(in_features=96, out_features=10),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self._model(x)
        return x, argmax(x, dim=1)


class mnist_rtg_v7(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_shape = (1, 28, 28)
        self._cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        self._cl = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=10)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = unsqueeze(x, dim=1)
        x = self._cnn(x)
        x = self._cl(x)
        return x, argmax(x, dim=1)


class mnist_rtg_v8(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_shape = (1, 28, 28)
        self._cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self._cl = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=10)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = unsqueeze(x, dim=1)
        x = self._cnn(x)
        x = self._cl(x)
        return x, argmax(x, dim=1)


class mnist_rtg_v9(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_shape = (1, 28, 28)
        self._cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self._cl = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=10)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = unsqueeze(x, dim=1)
        x = self._cnn(x)
        x = self._cl(x)
        return x, argmax(x, dim=1)


class mnist_rtg_v10(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_shape = (1, 28, 28)
        self._cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self._cl = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=10)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = unsqueeze(x, dim=1)
        x = self._cnn(x)
        x = self._cl(x)
        return x, argmax(x, dim=1)


class mnist_rtg_v11(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_shape = (1, 28, 28)
        self._cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d(p=0.15),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self._cl = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=10)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = unsqueeze(x, dim=1)
        x = self._cnn(x)
        x = self._cl(x)
        return x, argmax(x, dim=1)


class mnist_rtg_v12(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_shape = (1, 28, 28)
        self._cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d(p=0.15),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self._cl = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=10),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = unsqueeze(x, dim=1)
        x = self._cnn(x)
        x = self._cl(x)
        return x, argmax(x, dim=1)
