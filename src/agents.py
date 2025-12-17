import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import random
import os
import numpy as np
from collections import deque

from replayBuffer import ReplayBuffer, PrioritizedReplayBuffer, PPOBuffer
from models import DQN, PPO



# Agente de DQN para entornos Atari.
class DQNAgent:
    # Inicializacion del agente DQN.
    def __init__(
        self,
        device: torch.device,   # El dispositivo (CPU o GPU) en el que se ejecutara el agente.
        n_actions: int,         # El numero de acciones posibles que el agente puede tomar.
        lr: float,              # La tasa de aprendizaje para el optimizador de la red neuronal del agente.
        epsilon_start: float,   # La tasa de exploracion inicial para la politica epsilon-greedy.
        epsilon_end: float,     # La tasa de exploracion final para la politica epsilon-greedy.
        epsilon_decay: float,   # La tasa a la que decae el epsilon con el tiempo.
        total_memory: int,      # La capacidad maxima de la memoria de repeticion.
        initial_memory: int,    # La numero minimo de transiciones requeridas en la memoria de repeticion antes de que comience el aprendizaje.
        gamma: float,           # El factor de descuento para las recompensas futuras en la actualizacion de Q-learning.
        target_update: int,     # La frecuencia con la que se actualiza la red objetivo.
        network_file=None,      # Un archivo para cargar los pesos de la red pre-entrenada.
        input_shape: tuple=None,# La forma de la entrada para la red neuronal.
        #----------------------------------------- Parámetros Rainbow DQN -----------------------------------------#
        use_double: bool=False,         # Habilitar Double DQN
        use_dueling: bool=False,        # Habilitar Dueling DQN
        use_per: bool=False,            # Habilitar Prioritized Experience Replay
        use_multi_step: bool=False,     # Habilitar N-step learning
        use_noisy: bool=False,          # Habilitar Noisy Nets
        use_distributional: bool=False  # Habilitar Distributional RL (C51)
        #----------------------------------------- Parámetros Rainbow DQN -----------------------------------------#
    ) -> None:
        '''----------------------------------------- Parámetros Rainbow DQN -----------------------------------------'''
        # Double
        self.use_double = use_double
        # Dueling
        self.use_dueling = use_dueling
        # PER
        self.use_per = use_per
        self.beta = 0.4  # Valor inicial de beta para PER
        # N-step
        self.use_multi_step = use_multi_step
        self.n_steps = 3 # Número de pasos para N-step learning
        self.n_step_buffer = deque(maxlen=self.n_steps) # Buffer para N-step
        # Noisy Nets
        self.use_noisy = use_noisy
        # Distributional RL
        self.use_distributional = use_distributional
        '''----------------------------------------- Parámetros Rainbow DQN -----------------------------------------'''
        self.n_actions = n_actions
        self.device = device
        # Inicializa las redes (policy y target).
        self.policy_net = DQN(input_shape, n_actions, use_dueling=use_dueling, use_noisy=use_noisy).to(device)
        self.target_net = DQN(input_shape, n_actions, use_dueling=use_dueling, use_noisy=use_noisy).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # Optimizador para la red de politica.
        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        # AMP.
        self.use_amp = torch.cuda.is_available() and device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        # Variables para la politica epsilon-greedy.
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        # Factor de descuento para las recompensas futuras.
        self.gamma = gamma
        # Contador de pasos realizados.
        self.steps_done = 0
        # Frecuencia de actualizacion de la red objetivo.
        self.target_update = target_update
        # Inicializa la memoria de repeticion.
        if use_per == False:
            self.memory = ReplayBuffer(capacity=total_memory, input_shape=input_shape, device=self.device)
        else:
            self.memory = PrioritizedReplayBuffer(capacity=total_memory)
        self.initial_memory = initial_memory
        # Si hay archivo de pesos existente, cargarlo correctamente.
        if network_file and os.path.exists(network_file):
            try:
                state_dict = torch.load(network_file, map_location=device)
                self.policy_net.load_state_dict(state_dict)
                self.target_net.load_state_dict(state_dict)
                print(f"Modelo cargado desde {network_file}")
            except Exception as e:
                print(f"Error al cargar pesos: {e}. Se entrenara desde cero.")
        

    # Almacena una nueva transicion en la memoria de repeticion.
    def new_transition(self, observation, action, reward, next_observation, done):
        # Convierte las observaciones a numpy arrays si es necesario.
        obs = np.array(observation, copy=False)
        # Convierte la siguiente observacion a numpy array si no es None.
        next_obs = np.array(next_observation, copy=False) if next_observation is not None else None
        # Almacena la transicion en la memoria.
        if self.use_multi_step == False:
            self.memory.push(obs, action, reward, next_obs, done)
        else:
            '''---------------------------------------- LÓGICA DE N-STEP ----------------------------------------'''
            # Guardar en buffer temporal
            transition = (obs, action, reward, next_obs, done)
            self.n_step_buffer.append(transition)
            # Si el buffer está lleno, calculamos la recompensa acumulada
            if len(self.n_step_buffer) == self.n_steps:
                reward_n, next_obs_n, done_n = self._get_n_step_info()
                obs_t, action_t = self.n_step_buffer[0][:2]
                self.memory.push(obs_t, action_t, reward_n, next_obs_n, done_n)
            # Si terminamos, vaciamos el buffer restante
            if done:
                while len(self.n_step_buffer) > 0:
                    reward_n, next_obs_n, done_n = self._get_n_step_info()
                    obs_t, action_t = self.n_step_buffer[0][:2]
                    self.memory.push(obs_t, action_t, reward_n, next_obs_n, done_n)
                    self.n_step_buffer.popleft()
            '''-------------------------------------------- LÓGICA DE N-STEP ----------------------------------------'''

    '''-------------------------------------------- LÓGICA DE N-STEP ----------------------------------------'''
    # Función auxiliar para calcular recompensas N-step
    def _get_n_step_info(self):
        reward_n, next_obs_n, done_n = 0, None, False
        # Calcula la recompensa acumulada y el siguiente estado N-step
        for i, transition in enumerate(self.n_step_buffer):
            _, _, r, next_obs, d = transition
            reward_n += r * (self.gamma ** i)
            if d: # Si se llega a un estado terminal
                done_n = True
                next_obs_n = next_obs
                break
            else: # No es terminal
                next_obs_n = next_obs
        return reward_n, next_obs_n, done_n
    '''-------------------------------------------- LÓGICA DE N-STEP ----------------------------------------'''
    

    # Calcula el valor de epsilon para la politica epsilon-greedy en un episodio dado.
    def epsilon(self, episode: int):
        ratio = min(episode / self.epsilon_decay, 1.0)
        decay = np.exp(-5 * ratio)  
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * decay
    

    # Calcula la siguiente accion a tomar utilizando una politica epsilon-greedy.
    def next_action(self, observation, epsilon: float) -> int:
        # Actualiza el contador de pasos.
        self.steps_done += 1
        # Si usamos Noisy Nets, ignoramos epsilon
        if self.use_noisy:
            should_explore = False
        else: # Si no usamos Noisy, usamos Epsilon-Greedy
            should_explore = random.random() < epsilon
        # Seleccion de accion.
        if not should_explore:
            '''-------------------------------------------- LÓGICA DE NOISY NET ----------------------------------------'''
            # Accion greedy (o Noisy)
            with torch.no_grad():
                # Asegurar shape
                obs_tensor = torch.as_tensor(np.array(observation), device=self.device, dtype=torch.float32)
                # Si la red espera un batch y recibimos (C,H,W), añadimos dim 0
                if obs_tensor.dim() == 3:
                    obs_tensor = obs_tensor.unsqueeze(0)
                # Forward pass (GPU si corresponde)
                qvals = self.policy_net(obs_tensor)
                action = qvals.argmax(dim=1).item()
            '''-------------------------------------------- LÓGICA DE NOISY NET ----------------------------------------'''
        else:
            # Accion aleatoria
            action = random.randrange(self.n_actions)
        return action


    # Realiza un paso de optimizacion de la red Q.
    def optimize(self, batch_size: int):
        # Solo comienza a optimizar una vez que haya suficientes transiciones en la memoria.
        if len(self.memory) < self.initial_memory or len(self.memory) < batch_size:
            return
        '''-------------------------------------------- LÓGICA DE NOISY NET ----------------------------------------'''
        # Resetear Ruido para entrenamiento estable 
        if self.use_noisy:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
        '''-------------------------------------------- LÓGICA DE NOISY NET ----------------------------------------'''
        # Muestrea un lote de transiciones de la memoria de repeticion.
        if self.use_per == False:
            # Muestreo normal 
            state, action, reward, next_state, done, _ = self.memory.sample(batch_size)
            indices, weights_tensor = None, None
        else:
            '''---------------------------------------- LÓGICA DE PER ----------------------------------------'''
            # Actualizamos Beta
            self.beta = min(1.0, self.beta + 1e-5)
            # Muestrea.
            state, action, reward, next_state, done, indices, weights = self.memory.sample(batch_size, self.beta)
            # Convertimos los pesos a tensor para multiplicar el loss
            weights_tensor = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
            '''---------------------------------------- LÓGICA DE PER ----------------------------------------'''
        # Convertimos los arrays de numpy a tensores.
        state_batch = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action_batch = torch.as_tensor(action, dtype=torch.int64, device=self.device)
        reward_batch = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        next_state_batch = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        done_batch = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        # Creamos la mascara de no finalizados.
        non_final_mask = (done_batch == 0).squeeze()
        # Calculo del Target (Red Objetivo).
        next_state_values = torch.zeros(batch_size, device=self.device)
        # Solo calculamos Q para los estados no finales.
        if non_final_mask.any():
            with torch.no_grad():
                non_final_next_states = next_state_batch[non_final_mask]
                if self.use_double == False: # DQN clasico
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
                else:   # Double DQN
                    '''---------------------------------------- LÓGICA DOUBLE DQN ----------------------------------------'''
                    best_actions = self.policy_net(non_final_next_states).argmax(dim=1).unsqueeze(1)
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, best_actions).squeeze(1)
                    '''---------------------------------------- LÓGICA DOUBLE DQN ----------------------------------------'''
        '''------------------------------------------------ LÓGICA N-STEP----------------------------------------'''
        # Si usamos N-step, gamma debe elevarse a la potencia de n_steps
        if self.use_multi_step:
            current_gamma = self.gamma ** self.n_steps
        else:
            current_gamma = self.gamma
        '''------------------------------------------------ LÓGICA N-STEP----------------------------------------'''
        expected_state_action_values = (next_state_values * current_gamma) + reward_batch.squeeze()
        # Definimos la funcion de calculo para no repetir codigo en el if/else del AMP.
        def compute_loss():
            q_values = self.policy_net(state_batch)
            state_action_values = q_values.gather(1, action_batch).squeeze()
            if self.use_per == False:
                loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)
            else:
                '''---------------------------------------- LÓGICA PER LOSS ----------------------------------------'''
                # Error individual por muestra.
                elementwise_loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')
                # Loss ponderado por los pesos.
                loss = (elementwise_loss * weights_tensor.squeeze()).mean()
                # Calculamos TD Error para actualizar el árbo.
                self.td_errors = torch.abs(state_action_values - expected_state_action_values).detach().cpu().numpy()
                '''---------------------------------------- LÓGICA PER LOSS ----------------------------------------'''
            return loss
        # Optimizacion del modelo.
        self.optimizer.zero_grad(set_to_none=True)
        # Uso de AMP si corresponde.
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                loss = compute_loss()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = compute_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
        '''---------------------------------------- LÓGICA PER LOSS ----------------------------------------'''
        if self.use_per:
            self.memory.update_priorities(indices, self.td_errors)
        '''---------------------------------------- LÓGICA PER LOSS ----------------------------------------'''
        # Actualizacion Soft/Hard de la Target Network..
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())




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
        gae_lambda: float,  # El factor de descuento para la ventaja generalizada (GAE).
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
        self.gae_lambda = gae_lambda
        # Inicializa la memoria de repeticion.
        self.memory = PPOBuffer(device=device)
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
        # Convertir a numpy arrays para procesamiento
        rewards_np = rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else np.array(rewards)
        masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else np.array(masks)
        values_np = values.cpu().numpy() if isinstance(values, torch.Tensor) else np.array(values)
        
        # Inicializar listas para GAE
        gae = 0.0
        gae_advantages = []
        
        # Iterar hacia atrás sobre los pasos
        for step in reversed(range(len(rewards_np))):
            # Valor del siguiente estado
            next_value_step = values_np[step + 1] if step + 1 < len(values_np) else next_value
            
            # Temporal difference error (delta)
            delta = float(rewards_np[step]) + gamma * float(next_value_step) * float(masks_np[step]) - float(values_np[step])
            
            # GAE recursivo: g_t = δ_t + (γλ) * mask_t * g_{t+1}
            gae = delta + gamma * tau * float(masks_np[step]) * gae
            
            # Insertar al inicio (porque iteramos hacia atrás)
            gae_advantages.insert(0, gae)
        
        # Convertir a tensores
        advantages = torch.tensor(gae_advantages, device=self.device, dtype=torch.float32)
        
        # Retornos = advantages + valores originales
        returns = advantages + values
        
        return advantages, returns
    
    def new_transition(self, observation, action, reward, next_observation, done, value, log_prob):
        # Convierte las observaciones a numpy arrays si es necesario.
        obs = np.array(observation, copy=False)
        # Almacena la transicion en la memoria.
        self.memory.push(obs, action, reward, value, log_prob, done)

    def next_action(self, observation: np.ndarray) -> tuple:
        # Actualiza el contador de pasos.
        self.steps_done += 1
        
        with torch.no_grad():
            # Convertir observación a tensor
            obs = torch.as_tensor(observation, device=self.device, dtype=torch.float32)
            
            # Asegurar que tiene dimensión batch (batch, C, H, W)
            # Si viene como (C, H, W), agregar dimensión batch
            if obs.dim() == 3:
                obs = obs.unsqueeze(0)  # (1, C, H, W)
            
            # Forward pass para obtener logits (política) y valor
            logits, value = self.policy_net(obs)
            
            # Crear distribución de probabilidad sobre las acciones
            dist = torch.distributions.Categorical(logits=logits)
            
            # Muestrear acción de la distribución
            action = dist.sample()
            
            # Calcular log de la probabilidad de la acción muestreada
            log_prob = dist.log_prob(action)
            
            # Extraer valores escalares
            action_item = action.item()
            log_prob_item = log_prob.item()
            value_item = value.squeeze().item()
        
        return action_item, log_prob_item, value_item
    
    def optimize(self, batch_size: int, n_epochs_ppo: int):
        
        # Solo comienza a optimizar una vez que haya suficientes transiciones en la memoria de repeticion.
        if len(self.memory) < self.initial_memory or len(self.memory) < batch_size:
            return
        
        # Muestrea un lote de transiciones de la memoria de repeticion (ya en device).
        state_batch, action_batch, reward_batch, values_batch, log_probs_batch, dones_batch, non_final_mask = self.memory.sample()
        
        # Calcular el valor del próximo estado para bootstrapping en GAE
        with torch.no_grad():
            # Usar el último estado para obtener next_value
            # state_batch ya tiene forma [batch_size, 4, 84, 84], así que usamos el último directamente
            if state_batch.dim() == 4:
                # Si tiene 4 dimensiones [batch, C, H, W], tomar el último y agregar batch dim
                last_state = state_batch[-1].unsqueeze(0).to(self.device)  # [1, 4, 84, 84]
            else:
                # Si tiene 3 dimensiones [C, H, W], agregar batch dim
                last_state = state_batch[-1].unsqueeze(0).to(self.device)  # [1, C, H, W]
            
            _, next_value = self.policy_net(last_state)
            next_value = next_value.item() if dones_batch[-1].item() == 0 else 0.0
        
        # Calcular ventajas y retornos usando GAE
        advantages, returns = self.GAE(
            rewards=reward_batch,
            masks=(~dones_batch).float(),  # Invertir dones (1.0 donde no hay terminal, 0.0 donde sí)
            values=values_batch,
            next_value=next_value,
            gamma=self.gamma,
            tau=self.gae_lambda,
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