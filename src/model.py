import torch.nn as nn
import torch


# Inicialización de pesos para las capas lineales y convolucionales.
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        # Inicializa los pesos de la capa
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# Definición de la arquitectura de la red neuronal DQN.
class DQN(nn.Module):
    # Constructor de la clase DQN
    def __init__(self, input_shape: tuple, n_actions: int) -> None:
        super().__init__()
        # Desempaquetamos la forma de entrada.
        c, h, w = input_shape
        # Capas convolucionales de la red.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Calculamos el tamaño de salida dinamicamente.
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out_size = self.conv_layers(dummy_input).view(1, -1).size(1)
        # Capas lineales de la red.
        self.linear_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(), 
            nn.Linear(512, n_actions)
        )
        # Aplica la inicialización de pesos a todas las capas.
        self.apply(init_weights)


    # Método forward que define el paso hacia adelante de la red.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalizamos la entrada dividiendo por 255.0 (float32 a uint8).
        x = x / 255.0
        # Pasamos la entrada por las capas convolucionales.
        x = self.conv_layers(x)
        # Aplanamos la salida de las capas convolucionales.
        x = x.view(x.size(0), -1) 
        # Pasamos la salida a través de las capas lineales.
        x = self.linear_layers(x)
        # Retornamos la salida de la red.
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