import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import random
import os
import numpy as np

from replayBuffer import ReplayBuffer
from model import DQN, PPO


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
        initial_memory: int,  # La número mínimo de transiciones requeridas en la memoria de repetición antes de que comience el aprendizaje.
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
        # (Conservamos el nombre original 'optimiser' para compatibilidad)
        self.optimiser = Adam(self.policy_net.parameters(), lr=lr)
        # AMP
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        # Variables para la política epsilon-greedy.
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.steps_done = 0
        self.target_update = target_update
        # Inicializa la memoria de repetición.
        self.memory = ReplayBuffer(capacity=total_memory, device=self.device)
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
        action: int,                # La acción tomada como int
        reward: torch.Tensor,       # La recompensa recibida como tensor (o float)
        next_observation: torch.Tensor,  # La siguiente observación/estado
        done: bool,                 # Indica si el episodio terminó
    ):
        # Convertimos action a tensor dentro del CPU (guardamos en RAM para el replay)
        action_t = torch.as_tensor([action], dtype=torch.long)
        # Aceptamos reward como float o tensor — unificamos dtype y guardamos en CPU
        if not torch.is_tensor(reward):
            reward_t = torch.as_tensor([reward], dtype=torch.float32)
        else:
            reward_t = reward.to(dtype=torch.float32).cpu()
        # Convertir observaciones a float32 y mover a CPU (para que el replay buffer gestione la transferencia)
        obs_t = observation.cpu().to(dtype=torch.float32) if torch.is_tensor(observation) else torch.as_tensor(observation, dtype=torch.float32)
        next_obs_t = None
        if not done:
            if torch.is_tensor(next_observation):
                next_obs_t = next_observation.cpu().to(dtype=torch.float32)
            else:
                next_obs_t = torch.as_tensor(next_observation, dtype=torch.float32)
        self.memory.push(
            obs_t,
            action_t,
            reward_t,
            next_obs_t,
            done
        )

    # Calcula el valor épsilon actual para la política épsilon-greedy.
    def epsilon(self, episode: int):
        ratio = min(episode / self.epsilon_decay, 1.0)
        decay = np.exp(-5 * ratio)  
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * decay

    # Calcula la siguiente acción a tomar utilizando una política epsilon-greedy
    def next_action(
        self,
        observation: torch.Tensor,  # La observación/estado actual
        epsilon: float               # epsilon como float
    ) -> int:
        self.steps_done += 1
        if random.random() > epsilon:
            # Acción greedy
            with torch.no_grad():
                # asegurar shape (si la obs viene sin batch dim)
                obs = observation.to(self.device, dtype=torch.float32)
                # Si la red espera un batch y recibimos (C,H,W), añadimos dim 0
                if obs.dim() == 3:
                    obs = obs.unsqueeze(0)  # (1, C, H, W)
                # Forward pass (GPU si corresponde)
                qvals = self.policy_net(obs)
                action = qvals.max(1)[1].item()
        else:
            # Acción aleatoria
            action = random.randrange(self.n_actions)
        return action

    # Realiza un paso de optimización de la red Q.
    def optimize(self, batch_size: int):  # batch_size: El tamaño del lote para el entrenamiento.
        # Solo comienza a optimizar una vez que haya suficientes transiciones en la memoria.
        if (len(self.memory) < self.initial_memory or len(self.memory) < batch_size):
            return
        # Muestrea un lote de transiciones de la memoria de repetición (ya en device).
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, non_final_mask = self.memory.sample(batch_size, device=self.device)
        # Asegurar shapes esperadas para gather
        if action_batch.dim() == 1:
            action_batch = action_batch.unsqueeze(1)
        # Computa los valores Q predichos para los pares estado-acción en el lote.
        # state_batch ya está en device (según tu sample)
        # Guardamos el loss para backward
        # Calculamos next_state_values con la target_net
        next_state_values = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        if non_final_mask.any():
            with torch.no_grad():
                next_vals = self.target_net(next_state_batch).max(1)[0]
                next_state_values[non_final_mask] = next_vals
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Forward para la policy_net (posible AMP)
        if self.use_amp:
            from torch.amp import autocast
            with autocast("cuda"):
                state_action_values = self.policy_net(state_batch).gather(1, action_batch)
                loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            # Cero los gradientes y usamos el scaler para backward/step
            self.optimiser.zero_grad(set_to_none=True)
            # backward escalado
            self.scaler.scale(loss).backward()
            # clip grad norm en lugar de clamp por parámetro
            self.scaler.unscale_(self.optimiser)
            clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.scaler.step(self.optimiser)
            self.scaler.update()
        else:
            # Sin AMP 
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)
            loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            # Cero los gradientes de los parámetros de la red de política.
            self.optimiser.zero_grad(set_to_none=True)
            # Computa los gradientes y realiza el recorte de gradientes.
            loss.backward()
            # OPTIMIZACIÓN: clip grad norm en vez de clamp manual por parámetro
            clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            # Actualiza los parámetros de la red de política.
            self.optimiser.step()
        # Actualiza los parámetros de la red objetivo si se alcanza el intervalo de actualización del objetivo.
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())



# Agente de Double DQN para entornos Atari.
class DDQNAgent(DQNAgent):
    # Realiza un paso de optimización de la red Q.
    def optimize(self, batch_size: int):  # batch_size: El tamaño del lote para el entrenamiento.
        # Solo comienza a optimizar una vez que haya suficientes transiciones en la memoria.
        if (len(self.memory) < self.initial_memory or len(self.memory) < batch_size):
            return
        # Muestrea un lote de transiciones de la memoria de repetición (ya en device).
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, non_final_mask = self.memory.sample(batch_size, device=self.device)
        # Asegurar shapes esperadas para gather
        if action_batch.dim() == 1:
            action_batch = action_batch.unsqueeze(1)
        # Computa los valores Q predichos para los pares estado-acción en el lote.
        # state_action_values con policy_net
        # Inicializa los valores del siguiente estado como ceros.
        next_state_values = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        if non_final_mask.any():
            with torch.no_grad():
                # Acciones seleccionadas por la policy_net (argmax) sobre los next states no finales
                best_next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
                # Evaluación de esas acciones con la target_net
                next_state_values[non_final_mask] = (
                    self.target_net(next_state_batch).gather(1, best_next_actions).squeeze(1)
                )
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Compute loss
        if self.use_amp:
            from torch.amp import autocast
            with autocast("cuda"):
                state_action_values = self.policy_net(state_batch).gather(1, action_batch)
                loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            self.optimiser.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimiser)
            clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.scaler.step(self.optimiser)
            self.scaler.update()
        else:
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)
            loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            self.optimiser.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimiser.step()
        # Actualiza los parámetros de la red objetivo si se alcanza el intervalo de actualización del objetivo.
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

class agentPPO :
    def __init__(
        self,
        device: torch.device, # El dispositivo (CPU o GPU) en el que se ejecutará el agente.
        n_actions: int, # El número de acciones posibles que el agente puede tomar.
        lr: float,  # La tasa de aprendizaje para el optimizador de la red neuronal del agente.
        clippping_epsilon: float,  # El valor de epsilon para el recorte en PPO.
        total_memory: int,  # La capacidad máxima de la memoria de repetición.
        initial_memory: int,  # La número mínimo de transiciones requeridas en la memoria de repetición antes de que comience el aprendizaje.
        gamma: float,  # El factor de descuento para las recompensas futuras en la actualización de Q-learning.
        c1: float,  # Coeficiente para el término de pérdida del valor.
        target_update: int,  # La frecuencia con la que se actualiza la red objetivo.
        network_file=None,  # Un archivo para cargar los pesos de la red pre-entrenada.
    ) -> None:
        self.n_actions = n_actions
        self.device = device
        # Inicializa las redes (policy y target)
        self.policy_net = PPO(n_actions).to(device)
        self.target_net = PPO(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # Optimizador para la red de política.
        # (Conservamos el nombre original 'optimiser' para compatibilidad)
        self.optimiser = Adam(self.policy_net.parameters(), lr=lr)
        self.use_amp = False
        self.scaler = None
        # Variables para el calculo de la pérdida PPO.
        self.steps_done = 0
        self.clippping_epsilon = clippping_epsilon
        self.gamma = gamma
        self.c1 = c1
        self.target_update = target_update
        # Inicializa la memoria de repetición.
        self.memory = ReplayMemory(capacity=total_memory, device=self.device)
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

    def next_action(
        self,
        observation: torch.Tensor,  # La observación/estado actual
        epsilon: float               # epsilon como float
    ) -> int:
        self.steps_done += 1
        if random.random() > epsilon:
            # Acción greedy
            with torch.no_grad():
                # asegurar shape (si la obs viene sin batch dim)
                obs = observation.to(self.device, dtype=torch.float32)
                # Si la red espera un batch y recibimos (C,H,W), añadimos dim 0
                if obs.dim() == 3:
                    obs = obs.unsqueeze(0)  # (1, C, H, W)
                # Forward pass (GPU si corresponde)
                logits = self.policy_net(obs)
                action_probs = torch.softmax(logits, dim=1)
                action = action_probs.multinomial(num_samples=1).item()
        else:
            # Acción aleatoria
            action = random.randrange(self.n_actions)
        return action