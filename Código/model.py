import torch.nn as nn
import torch

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
class DQN(nn.Module):
    """Deep Q Network"""

    def __init__(self, n_actions: int) -> None:
        super().__init__()

        # Capas convolucionales de la red.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Capas lineales de la red.
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normaliza la entrada escalándola a [0, 1].
        x = x.float() / 255

        # Pasa la entrada a través de las capas convolucionales.
        x = self.conv_layers(x)

        # Aplanar la salida de las capas convolucionales.
        x = x.view(-1, 64 * 7 * 7)

        # Pasar la salida aplanada a través de las capas lineales.
        x = self.linear_layers(x)
        return x
