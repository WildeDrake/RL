import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import random
import os
import numpy as np

from replayBuffer import ReplayBuffer
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
        use_double: bool=False,   # Habilitar Double DQN
        use_dueling: bool=False,  # Habilitar Dueling DQN
        use_per: bool=False       # Habilitar Prioritized Experience Replay
        #----------------------------------------- Parámetros Rainbow DQN -----------------------------------------#
    ) -> None:
        '''----------------------------------------- Parámetros Rainbow DQN -----------------------------------------'''
        self.use_double = use_double
        self.use_dueling = use_dueling
        self.use_per = use_per
        '''----------------------------------------- Parámetros Rainbow DQN -----------------------------------------'''

        self.n_actions = n_actions
        self.device = device
        # Inicializa las redes (policy y target).
        self.policy_net = DQN(input_shape, n_actions, use_dueling=use_dueling).to(device)
        self.target_net = DQN(input_shape, n_actions, use_dueling=use_dueling).to(device)
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
        self.memory = ReplayBuffer(capacity=total_memory, input_shape=input_shape, device=self.device)
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
        next_obs = np.array(next_observation, copy=False) if next_observation is not None else None
        # Almacena la transicion en la memoria.
        self.memory.push(obs, action, reward, next_obs, done)


    # Calcula el valor epsilon actual para la politica epsilon-greedy.
    def epsilon(self, episode: int):
        # Decaimiento exponencial del epsilon.
        ratio = min(episode / self.epsilon_decay, 1.0)
        decay = np.exp(-5 * ratio)  
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * decay


    # Calcula la siguiente accion a tomar utilizando una politica epsilon-greedy.
    def next_action(self, observation, epsilon: float) -> int:
        # Actualiza el contador de pasos.
        self.steps_done += 1
        # Exploracion epsilon-greedy.
        if random.random() > epsilon:
            # Accion greedy.
            with torch.no_grad():
                # Convertimos la observacion a tensor en el device correcto.
                obs_tensor = torch.as_tensor(np.array(observation), device=self.device, dtype=torch.float32)
                # Agregamos dimension de batch si falta (4, 84, 84) -> (1, 4, 84, 84).
                if obs_tensor.dim() == 3:
                    obs_tensor = obs_tensor.unsqueeze(0)
                # Forward pass (GPU si corresponde).
                qvals = self.policy_net(obs_tensor)
                action = qvals.argmax(dim=1).item()
        else:
            # Accion aleatoria.
            action = random.randrange(self.n_actions)
        return action


    # Realiza un paso de optimizacion de la red Q.
    def optimize(self, batch_size: int):  # batch_size: El tamaño del lote para el entrenamiento.
        # Solo comienza a optimizar una vez que haya suficientes transiciones en la memoria.
        if len(self.memory) < self.initial_memory or len(self.memory) < batch_size:
            return
        # Muestrea un lote de transiciones de la memoria de repeticion (ya en device).
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, non_final_mask = self.memory.sample(batch_size)
        # Calculo del Target (Red Objetivo).
        next_state_values = torch.zeros(batch_size, device=self.device)
        # Solo calculamos Q para los estados no finales.
        if non_final_mask.any():
                with torch.no_grad():
                    # Obtener los valores maximos de Q para los siguientes estados desde la red objetivo.
                    non_final_next_states = next_state_batch[non_final_mask]
                    if self.use_double == False: # DQN clasico
                        # Calculo de los valores maximos de Q para los siguientes estados.
                        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
                    else:   # Double DQN
                        '''---------------------------------------- LÓGICA DOUBLE DQN ----------------------------------------'''
                        # Selección de acción: La Red de Política (policy_net) elige la mejor acción para los siguientes estados.
                        best_actions = self.policy_net(non_final_next_states).argmax(dim=1).unsqueeze(1)
                        # Evaluación de acción: La Red Objetivo (target_net) evalúa el valor Q de esas acciones seleccionadas.
                        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, best_actions).squeeze(1)
                        '''---------------------------------------- LÓGICA DOUBLE DQN ----------------------------------------'''
        # Calculo del valor esperado de la accion.
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Definimos la funcion de calculo para no repetir codigo en el if/else del AMP.
        def compute_loss():
            # Calcula los valores Q actuales para las acciones tomadas.
            q_values = self.policy_net(state_batch)
            state_action_values = q_values.gather(1, action_batch)
            # Huber Loss (SmoothL1).
            loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            return loss
        # Optimizacion del modelo.
        self.optimizer.zero_grad(set_to_none=True) # set_to_none es un poco mas rapido.
        # Usar AMP si esta habilitado.
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                loss = compute_loss()
            # Backward y step con escalado.
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0) # Evita explosion de gradientes.
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:   # Sin AMP.
            loss = compute_loss()
            loss.backward()
            clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
        # Actualizacion Soft/Hard de la Target Network.
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())



# Agente de Double DQN para entornos Atari.
class RainbowDQNAgent(DQNAgent):
    pass

# Agente PPO para entornos Atari.
class PPOAgent:
    def __init__(
        self,
        device: torch.device, # El dispositivo (CPU o GPU) en el que se ejecutara el agente.
        n_actions: int, # El numero de acciones posibles que el agente puede tomar.
        lr: float,  # La tasa de aprendizaje para el optimizador de la red neuronal del agente.
        clippping_epsilon: float,  # El valor de epsilon para el recorte en PPO.
        total_memory: int,  # La capacidad maxima de la memoria de repeticion.
        initial_memory: int,  # La numero minimo de transiciones requeridas en la memoria de repeticion antes de que comience el aprendizaje.
        gamma: float,  # El factor de descuento para las recompensas futuras en la actualizacion de Q-learning.
        c1: float,  # Coeficiente para el termino de perdida del valor.
        target_update: int,  # La frecuencia con la que se actualiza la red objetivo.
        network_file=None,  # Un archivo para cargar los pesos de la red pre-entrenada.
        input_shape=None  # La forma de la entrada para la red neuronal.
    ) -> None:
        self.n_actions = n_actions
        self.device = device
        # Inicializa las redes (policy y target)
        self.policy_net = PPO(n_actions).to(device)
        self.target_net = PPO(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # Optimizador para la red de politica.
        # (Conservamos el nombre original 'optimizer' para compatibilidad)
        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.use_amp = torch.cuda.is_available() and device.type == 'cuda'
        self.scaler = None
        # Variables para el calculo de la perdida PPO.
        self.steps_done = 0
        self.clippping_epsilon = clippping_epsilon
        self.gamma = gamma
        self.c1 = c1
        self.target_update = target_update
        # Inicializa la memoria de repeticion.
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
                print(f"Error al cargar pesos: {e}. Se entrenara desde cero.")
        self.input_shape = input_shape

    def GAE(self, rewards, masks, values, next_value, gamma=0.99, tau=0.9): # Sirve para calcular las ventajas generalizadas del buffer
        values = values + [next_value]
        gae = 0
        ventajas = []
        for step in reversed(range(len(rewards))): # Recomiendan hacerlo para atrás
            # Conseguimos el error de Temporal Difference del step (delta)
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            # Calcular GAE: Una suma ponderada de los deltas futuros.
            gae = delta + gamma * tau * masks[step] * gae
            ventajas.insert(0, gae + values[step])
        # Convertir a tensores
        ventajas = torch.tensor(ventajas, device=self.device, dtype=torch.float32) #TODO: molestar con float32.
        tvalores = torch.tensor(values, device=self.device, dtype=torch.float32) # Tensor de valores
        returns = ventajas - tvalores

        #Normalización??
        ventajas = (ventajas - ventajas.mean()) / (ventajas.std() + 1e-8)

        return ventajas, returns
    
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
    
    def optimize(self, batch_size: int): #  AAAAAAAAAAAAAAA EL DIABLO
        # Solo comienza a optimizar una vez que haya suficientes transiciones en la memoria de repeticion.
        if len(self.memory) < self.initial_memory or len(self.memory) < batch_size:
            return
        # Muestrea un lote de transiciones de la memoria de repeticion (ya en device).
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, non_final_mask = self.memory.sample(batch_size)
        # Calculo del Target (Red Objetivo).
        next_state_values = torch.zeros(batch_size, device=self.device)
        # Solo calculamos Q para los estados no finales.
        if non_final_mask.any():
                with torch.no_grad():
                    # Obtener los valores maximos de Q para los siguientes estados desde la red objetivo.
                    non_final_next_states = next_state_batch[non_final_mask]
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]