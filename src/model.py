import torch.nn as nn
import torch


# Inicialización de pesos para las capas lineales y convolucionales
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        # Inicializa los pesos de la capa
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# Definición de la arquitectura de la red neuronal DQN
class DQN(nn.Module):
    # Constructor de la clase DQN
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
        # Aplica la inicialización de pesos a todas las capas.
        self.apply(init_weights)

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

class PPO(nn.Module):
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
            nn.Linear(64 * 7 * 7, 512), nn.ReLU()
        )
        # Capa para la política (acciones)
        self.policy_head = nn.Linear(512, n_actions)
        # Capa para el valor (critic)
        self.value_head = nn.Linear(512, 1)
        # Aplica la inicialización de pesos a todas las capas.
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # Normaliza la entrada escalándola a [0, 1].
        x = x.float() / 255
        # Pasa la entrada a través de las capas convolucionales.
        x = self.conv_layers(x)
        # Aplanar la salida de las capas convolucionales.
        x = x.view(-1, 64 * 7 * 7)
        # Pasar la salida aplanada a través de las capas lineales.
        x = self.linear_layers(x)
        # Obtener la política y el valor
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value