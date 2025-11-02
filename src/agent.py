from torch.optim import Adam
from typing import Optional
from replay_memory import ReplayMemory, Transition
from model import DQN
import torch
import random
import os

# Agente de DQN para entornos Atari.
class DQNAgent:
    # Inicialización del agente DQN.
    def __init__(
        self,
        device: torch.device, # El dispositivo (CPU o GPU) en el que se ejecutará el agente.
        n_actions: int, # El número de acciones posibles que el agente puede tomar.
        lr: float,  # La tasa de aprendizaje para el optimizador de la red neuronal del agente.
        epsilon_start: float,   # La tasa de exploración inicial para la política epsilon-greedy.
        epsilon_end: float,  # La tasa de exploración final para la política epsilon-greedy.
        epsilon_decay: float,  # La tasa a la que decae el épsilon con el tiempo.
        total_memory: int,  # La capacidad máxima de la memoria de repetición.
        initial_memory: int,  # El número mínimo de transiciones requeridas en la memoria de repetición antes de que comience el aprendizaje.
        gamma: float,  # El factor de descuento para las recompensas futuras en la actualización de Q-learning.
        target_update: int,  # La frecuencia con la que se actualiza la red objetivo.
        network_file=None,  # Un archivo para cargar los pesos de la red pre-entrenada.
    ) -> None:
        self.n_actions = n_actions
        self.device = device
        # Inicializa las redes (policy y target)
        self.policy_net = DQN(n_actions).to(device)
        self.target_net = DQN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # Optimizador para la red de política.
        self.optimiser = Adam(self.policy_net.parameters(), lr=lr)
        self.steps_done = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.target_update = target_update
        # Inicializa la memoria de repetición.
        self.memory = ReplayMemory(capacity=total_memory)
        self.initial_memory = initial_memory
        # Si hay archivo de pesos existente, cargarlo correctamente
        if network_file and os.path.exists(network_file):
            state_dict = torch.load(network_file, map_location=device)
            try:
                self.policy_net.load_state_dict(state_dict)
                self.target_net.load_state_dict(state_dict)
                print(f"Modelo cargado desde {network_file}")
            except RuntimeError as e:
                print(f"Error al cargar pesos: {e}. Se entrenará desde cero.")
    # Almacena una nueva transición en la memoria de repetición.
    def new_transition(
        self,
        observation: torch.Tensor,  # La observación/estado actual.
        action: torch.Tensor,  # La acción tomada.
        reward: torch.Tensor,  # La recompensa recibida.
        next_observation: torch.Tensor,  # La siguiente observación/estado.
        done: bool,  # Una bandera que indica si el episodio ha terminado.
    ):
        self.memory.push(
            observation.to(self.device),
            action.to(self.device),
            reward.to(self.device),
            next_observation.to(self.device) if not done else None,
            done
        )
    # Calcula el valor épsilon actual para la política épsilon-greedy.
    def epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - min(self.steps_done / self.epsilon_decay, 1))
    # Calcula la siguiente acción a tomar utilizando una política épsilon-greedy.
    def next_action(
        self,
        observation: torch.Tensor,  # La observación/estado actual.
        epsilon: Optional[float] = None  # El valor épsilon a utilizar para la política épsilon-greedy.
    ) -> torch.Tensor:
        # Si no se proporciona un valor épsilon, calcula el valor actual.
        if epsilon is None:
            epsilon = self.epsilon()
        # Incrementa el contador de pasos.
        self.steps_done += 1
        # Decide si tomar una acción aleatoria o la mejor acción según la política actual.
        if random.random() > epsilon:
            with torch.no_grad():
                return (
                    self.policy_net(observation.to(self.device))
                    .max(1)[1]
                    .view(1, 1)
                )
        else:  # Toma una acción aleatoria.
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=self.device,
                dtype=torch.long,
            )
    # Realiza un paso de optimización de la red Q.
    def optimise(self, batch_size: int):  # batch_size: El tamaño del lote para el entrenamiento.
        # Solo comienza a optimizar una vez que haya suficientes transiciones en la memoria.
        if (len(self.memory) < self.initial_memory or len(self.memory) < batch_size):
            return 
        # Muestrea un lote de transiciones de la memoria de repetición (ya en device).
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, non_final_mask = self.memory.sample(batch_size, device=self.device)
        # Asegurar shapes esperadas para gather
        if action_batch.dim() == 1:
            action_batch = action_batch.unsqueeze(1)
        # Computa los valores Q predichos para los pares estado-acción en el lote.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # Inicializa los valores del siguiente estado como ceros.
        next_state_values = torch.zeros(batch_size, device=self.device)
        if non_final_mask.any():
            with torch.no_grad():
                # next_state_batch contiene solo los next_states no finales (según la implementación de sample)
                next_vals = self.target_net(next_state_batch).max(1)[0]
                next_state_values[non_final_mask] = next_vals
        # Computa los valores esperados de estado-acción utilizando la ecuación de Bellman.
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Computa la pérdida utilizando la pérdida de Huber (pérdida L1 suave).
        loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # Cero los gradientes de los parámetros de la red de política.
        self.optimiser.zero_grad()
        # Computa los gradientes y realiza el recorte de gradientes.
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        # Actualiza los parámetros de la red de política.
        self.optimiser.step()
        # Actualiza los parámetros de la red objetivo si se alcanza el intervalo de actualización del objetivo.
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())





# Agente de Double DQN para entornos Atari.
class DDQNAgent(DQNAgent):
    # Realiza un paso de optimización de la red Q.
    def optimise(self, batch_size: int):  # batch_size: El tamaño del lote para el entrenamiento.
        # Solo comienza a optimizar una vez que haya suficientes transiciones en la memoria.
        if (len(self.memory) < self.initial_memory or len(self.memory) < batch_size):
            return 
        # Muestrea un lote de transiciones de la memoria de repetición (ya en device).
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, non_final_mask = self.memory.sample(batch_size, device=self.device)
        # Asegurar shapes esperadas para gather
        if action_batch.dim() == 1:
            action_batch = action_batch.unsqueeze(1)
        # Computa los valores Q predichos para los pares estado-acción en el lote.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # Inicializa los valores del siguiente estado como ceros.
        next_state_values = torch.zeros(batch_size, device=self.device)
        if non_final_mask.any():
            with torch.no_grad():
                # Acciones seleccionadas por la policy_net (argmax) sobre los next states no finales
                best_next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
                # Evaluación de esas acciones con la target_net
                next_state_values[non_final_mask] = (
                    self.target_net(next_state_batch).gather(1, best_next_actions).squeeze(1)
                )
        # Computa los valores esperados de estado-acción utilizando la ecuación de Bellman.
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Computa la pérdida utilizando la pérdida de Huber (pérdida L1 suave).
        loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # Cero los gradientes de los parámetros de la red de política.
        self.optimiser.zero_grad()
        # Computa los gradientes y realiza el recorte de gradientes.
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        # Actualiza los parámetros de la red de política.
        self.optimiser.step()
        # Actualiza los parámetros de la red objetivo si se alcanza el intervalo de actualización del objetivo.
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())