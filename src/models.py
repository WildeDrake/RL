import torch.nn as nn
import torch
import torch.nn.functional as F
import math



'''------------------------------------------------ LÓGICA NOISY NET----------------------------------------'''
# Definicion de la capa lineal con ruido Noisy Nets.
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        # Parámetros aprendibles: Mu (Media) y Sigma (Ruido).
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        # Inicialización especial.
        self.reset_parameters()
        self.reset_noise()


    # Inicialización especial.
    def reset_parameters(self):
        # Inicialización especial para Noisy Nets (Factorized Gaussian Noise).
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))


    # Establece el ruido para la capa.
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())


    # Resetea el ruido para la capa.
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)


    # Metodo forward que define el paso hacia adelante de la capa.
    def forward(self, input):
        if self.training:
            # Entrenamiento: w = µ + σ * ε.
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            # Evaluación: w = µ.
            return F.linear(input, self.weight_mu, self.bias_mu)
'''------------------------------------------------ LÓGICA NOISY NET----------------------------------------'''



# Inicializacion de pesos para las capas lineales y convolucionales.
def init_weights(m):
    # Si es NoisyLinear, NO tocamos los pesos.
    if isinstance(m, NoisyLinear):
        return 
    # Solo inicializamos si es Linear normal o Conv2d.
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)



# Definicion de la arquitectura de la red neuronal DQN.
class DQN(nn.Module):
    # Constructor de la clase DQN.
    def __init__(self, input_shape: tuple, n_actions: int, use_dueling: bool=False, use_noisy: bool=False, use_distributional: bool=False) -> None:
        super().__init__()
        self.use_dueling = use_dueling
        self.use_noisy = use_noisy
        self.use_distributional = use_distributional
        # Configuración Distributional RL.
        if self.use_distributional:
            self.n_atoms = 51
        else:
            self.n_atoms = 1
        self.n_actions = n_actions
        # Desempaquetamos la forma de entrada.
        c, h, w = input_shape
        # Capas convolucionales de la red.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        # Calculamos el tamaño de salida dinamicamente.
        '''------------------------------------------------ LÓGICA NOISY NET----------------------------------------'''
        if self.use_noisy:
            LinearLayer = NoisyLinear 
        else:
            LinearLayer = nn.Linear
        '''------------------------------------------------ LÓGICA NOISY NET----------------------------------------'''
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out_size = self.conv_layers(dummy_input).view(1, -1).size(1)
        # Ajustamos el tamaño de salida si usamos Distributional RL.
        output_dim = n_actions * self.n_atoms
        # Definimos las capas lineales dependiendo de si usamos Dueling DQN o DQN simple.
        if self.use_dueling == False:
            # Capas lineales de la red.
            self.linear_layers = nn.Sequential(
                LinearLayer(conv_out_size, 512), 
                nn.ReLU(), 
                LinearLayer(512, output_dim)
            )
        else:
            '''---------------------------------------- LÓGICA DUELING DQN + C51 ----------------------------------------'''
            # Value predice 51 átomos, Advantage predice Actions * 51 átomos.
            if self.use_distributional:
                val_out = self.n_atoms
            else:
                val_out = 1
            adv_out = output_dim
            # Dos cabezas separadas, Value (V) y Advantage (A).
            self.fc_value = nn.Sequential(
                LinearLayer(conv_out_size, 512), nn.ReLU(),
                LinearLayer(512, val_out) # Predice el valor del estado.
            )
            self.fc_advantage = nn.Sequential(
                LinearLayer(conv_out_size, 512), nn.ReLU(),
                LinearLayer(512, adv_out) # Predice N valores (ventaja de cada accion).
            )
            '''---------------------------------------- LÓGICA DUELING DQN + C51 ----------------------------------------'''
        # Aplica la inicializacion de pesos a todas las capas.
        self.apply(init_weights)


    # Este método será llamado por el Agente antes de entrenar.
    def reset_noise(self):
        if not self.use_noisy: return
        # Busca todas las capas NoisyLinear hijas y resetea su ruido.
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


    # Metodo forward que define el paso hacia adelante de la red.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalizamos la entrada dividiendo por 255.0 (float32 a uint8).
        x = x / 255.0
        # Pasamos la entrada por las capas convolucionales.
        x = self.conv_layers(x)
        # Aplanamos la salida de las capas convolucionales.
        x = x.view(x.size(0), -1) 
        # Dependiendo de si usamos Dueling DQN o no, seguimos diferentes caminos.
        if self.use_dueling == False:
            # Pasamos la salida a traves de las capas lineales.
            x = self.linear_layers(x)
            # Retornamos la salida de la red.
        else:
            '''---------------------------------------- LÓGICA DUELING DQN + C51----------------------------------------'''
            V = self.fc_value(x)
            A = self.fc_advantage(x)
            if self.use_distributional == False:
                # Dueling DQN normal.
                # Fórmula de Dueling: Q = V + (A - mean(A)).
                x = V + (A - A.mean(dim=1, keepdim=True))
            else:
                # Dueling con C51 requiere manipular dimensiones.
                # V shape: (batch, 1, atoms).
                # A shape: (batch, actions, atoms).
                V = V.view(-1, 1, self.n_atoms)
                A = A.view(-1, self.n_actions, self.n_atoms)
                x = V + (A - A.mean(dim=1, keepdim=True))
        # Si usamos Distributional RL, aplicamos Softmax.
        if self.use_distributional:
            # Si es C51, reestructuramos a (Batch, Actions, Atoms) y aplicamos Softmax para obtener probabilidades.
            x = x.view(-1, self.n_actions, self.n_atoms)
            x = F.softmax(x, dim=2) # Probabilidades suman 1 en el eje de los átomos.
        # Retornamos la salida de la red.
        return x
    '''------------------------------------------------ LÓGICA DUELING DQN + C51----------------------------------------'''



# Definicion de la arquitectura de la red neuronal PPO.
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
        # Capa para la politica (acciones).
        self.policy_head = nn.Linear(512, n_actions)
        # Capa para el valor (critic).
        self.value_head = nn.Linear(512, 1)
        # Aplica la inicializacion de pesos a todas las capas.
        self.apply(init_weights)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normaliza la entrada escalandola a [0, 1].
        x = x.float() / 255
        # Pasa la entrada a traves de las capas convolucionales.
        x = self.conv_layers(x)
        # Aplanar la salida de las capas convolucionales.
        x = x.view(-1, 64 * 7 * 7)
        # Pasar la salida aplanada a traves de las capas lineales.
        x = self.linear_layers(x)
        # Obtener la politica y el valor.
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value