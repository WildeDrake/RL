import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import random
import os
import numpy as np
from collections import deque

from Buffers import PPOBuffer
from models import PPO



# Agente PPO para entornos Atari.
class PPOAgent:
    def __init__(
        self,
        device: torch.device, # El dispositivo (CPU o GPU) en el que se ejecutara el agente.
        n_actions: int, # El numero de acciones posibles que el agente puede tomar.
        lr: float,  # La tasa de aprendizaje para el optimizador de la red neuronal del agente.
        clipping_epsilon: float,  # El valor de epsilon para el recorte en PPO.
        total_memory: int,  # La capacidad maxima de la memoria de repeticion.
        initial_memory: int,  # La numero minimo de transiciones requeridas en la memoria de repeticion antes de que comience el aprendizaje.
        gamma: float,  # El factor de descuento para las recompensas futuras en la actualizacion de Q-learning.
        network_file=None,  # Un archivo para cargar los pesos de la red pre-entrenada.
        input_shape=None  # La forma de la entrada para la red neuronal.
    ) -> None:
        self.n_actions = n_actions
        self.device = device
        # Inicializa las redes (policy y target)
        self.policy_net = PPO(n_actions).to(device)
        # Optimizador para la red de politica.
        # (Conservamos el nombre original 'optimizer' para compatibilidad)
        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.use_amp = torch.cuda.is_available() and device.type == 'cuda'
        self.scaler = None
        # Variables para el calculo de la perdida PPO.
        self.steps_done = 0
        self.clipping_epsilon = clipping_epsilon
        self.gamma = gamma
        # Inicializa la memoria de repeticion.
        self.memory = PPOBuffer(capacity=total_memory, device=self.device)
        self.initial_memory = initial_memory
        # Si hay archivo de pesos existente, cargarlo correctamente
        if network_file and os.path.exists(network_file):
            state_dict = torch.load(network_file, map_location=device)
            try:
                self.policy_net.load_state_dict(state_dict)
                print(f"Modelo cargado desde {network_file}")
            except RuntimeError as e:
                print(f"Error al cargar pesos: {e}. Se entrenara desde cero.")
        self.input_shape = input_shape

    def GAE(self, rewards, masks, values, next_value, gamma=0.99, tau=0.9):
        # Asegurar que son listas para poder indexar
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().tolist()
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().tolist()
        if isinstance(values, torch.Tensor):
            values_list = values.cpu().tolist()
        else:
            values_list = list(values)
        
        # Inicializar listas para GAE
        gae = 0
        gae_advantages = []
        
        # Iterar hacia atrás sobre los pasos
        for step in reversed(range(len(rewards))):
            # Valor del siguiente estado
            next_value_step = values_list[step + 1] if step + 1 < len(values_list) else next_value
            
            # Temporal difference error (delta)
            delta = rewards[step] + gamma * next_value_step * masks[step] - values_list[step]
            
            # GAE recursivo: g_t = δ_t + (γλ) * mask_t * g_{t+1}
            gae = delta + gamma * tau * masks[step] * gae
            
            # Insertar al inicio (porque iteramos hacia atrás)
            gae_advantages.insert(0, gae)
        
        # Convertir a tensores
        advantages = torch.tensor(gae_advantages, device=self.device, dtype=torch.float32)
        values_tensor = torch.tensor(values_list, device=self.device, dtype=torch.float32)
        
        # Retornos = advantages + valores originales
        returns = advantages + values_tensor
        
        return advantages, returns
    
    def new_transition(self, observation, action, reward, next_observation, done, value, log_prob):
        # Convierte las observaciones a numpy arrays si es necesario.
        obs = np.array(observation, copy=False)
        # Almacena la transicion en la memoria.
        self.memory.push(obs, action, reward, value, log_prob, done)

    def next_action(
        self,
        observation: torch.Tensor,  # La observacion/estado actual
        epsilon: float               # epsilon como float
    ) -> int:
        self.steps_done += 1
        if random.random() > epsilon:
            # Accion greedy
            with torch.no_grad():
                # asegurar shape (si la obs viene sin batch dim)
                obs = observation.to(self.device, dtype=torch.float32)
                # Si la red espera un batch y recibimos (C,H,W), añadimos dim 0
                if obs.dim() == 3:
                    obs = obs.unsqueeze(0)  # (1, C, H, W)
                # Forward pass (GPU si corresponde)
                logits, value = self.policy_net(obs)
                action_probs = torch.softmax(logits, dim=1)
                action = action_probs.multinomial(num_samples=1).item()
        else:
            # Accion aleatoria
            action = random.randrange(self.n_actions)
        return action
    
    def optimize(self, batch_size: int, n_epochs_ppo: int):
        
        # Solo comienza a optimizar una vez que haya suficientes transiciones en la memoria de repeticion.
        if len(self.memory) < self.initial_memory or len(self.memory) < batch_size:
            return
        
        # Muestrea un lote de transiciones de la memoria de repeticion (ya en device).
        state_batch, action_batch, reward_batch, values_batch, log_probs_batch, dones_batch, non_final_mask = self.memory.sample(batch_size)
        
        # Calcular el valor del próximo estado para bootstrapping en GAE
        with torch.no_grad():
            # Usar el último estado no-terminal para obtener next_value
            last_state = state_batch[-1].unsqueeze(0) if state_batch.dim() == 3 else state_batch[-1:].unsqueeze(0)
            _, next_value = self.policy_net(last_state)
            next_value = next_value.item() if dones_batch[-1].item() == 0 else 0.0
        
        # Calcular ventajas y retornos usando GAE
        advantages, returns = self.GAE(
            rewards=reward_batch,
            masks=1.0 - dones_batch,  # Pasar como tensores, la función los convertirá
            values=values_batch,
            next_value=next_value,
            gamma=self.gamma,
            tau=0.9
        )
        
        # Tamaño total del dataset
        dataset_size = len(reward_batch)

        for epoch in range(n_epochs_ppo):
            # Generar indices aleatorios para mini-batches
            indices = torch.randperm(dataset_size, device=self.device)
            
            # Iterar sobre mini-batches
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                mb_indices = indices[start_idx:end_idx]
                
                # Extraer mini-batch (vars de mb)
                mb_states = state_batch[mb_indices]
                mb_actions = action_batch[mb_indices]
                mb_old_log_probs = log_probs_batch[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Normalizar ventajas para estabilidad numérica
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Forward pass con la politica actual
                logits, values_pred = self.policy_net(mb_states)
                
                # Asegurar shape correcto: values_pred debe ser (mb_size,)
                values_pred = values_pred.squeeze(-1)
                
                # Crear distribucion de probabilidad sobre las acciones
                dist = torch.distributions.Categorical(logits=logits)
                
                # Calcular nuevas log probabilidades para las acciones tomadas
                mb_new_log_probs = dist.log_prob(mb_actions)
                
                # Calcular entropia (para fomentar exploracion)
                entropy = dist.entropy().mean()
                
                # Calcular policy loss con clipping PPO
                # Ratio: π_new(a|s) / π_old(a|s)
                ratio = torch.exp(mb_new_log_probs - mb_old_log_probs)
                
                # Surrogate sin clip: ratio * A
                surr1 = ratio * mb_advantages
                
                # Surrogate con clip: clip(ratio, 1-ε, 1+ε) * A
                surr2 = torch.clamp(
                    ratio, 
                    1.0 - self.clipping_epsilon,
                    1.0 + self.clipping_epsilon
                ) * mb_advantages
                
                # Policy loss: negativo del mínimo (paso conservador en PPO)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss: MSE entre valores predichos y returns objetivo
                value_loss = torch.nn.functional.mse_loss(values_pred, mb_returns)
                
                # Coeficientes de pérdida
                c1 = 1.0  # Peso del value loss
                c2 = 0.01  # Peso del entropy bonus
                # Loss total = PPO loss
                total_loss = policy_loss + c1 * value_loss - c2 * entropy
                
                # Optimizadores
                
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                
                # Gradient clipping (evita actualizaciones demasiado grandes)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
                
                self.optimizer.step()

        # Limpiar la memoria PPO despues de la optimizacion
        self.memory.clear()