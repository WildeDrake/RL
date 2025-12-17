import numpy as np
import torch

from utils import SumTree



# Definicion del buffer de repeticion.
class ReplayBuffer:
    # Constructor.
    def __init__(self, capacity: int, input_shape: tuple, device: str = "cuda"):
        self.device = device
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        # Preasignamos memoria contigua. 
        self.states = np.zeros((capacity, *input_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *input_shape), dtype=np.uint8)
        # Usamos tipos especificos para ahorrar memoria.
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    # Guarda una transicion en el buffer.
    def push(self, state, action, reward, next_state, done):
        # Si next_state es None, lo reemplazamos por el estado actual.
        if next_state is None:
            next_state = state
        # Guardamos en la posicion actual del puntero.
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        # Actualizamos puntero y tamaño.
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # Retorna un lote aleatorio de transiciones.
    def sample(self, batch_size: int):
        # Generamos indices aleatorios una sola vez.
        idxs = np.random.randint(0, self.size, size=batch_size)
        # Extraemos el lote usando los indices generados.
        batch_states = torch.as_tensor(self.states[idxs], device=self.device, dtype=torch.float32)
        batch_next_states = torch.as_tensor(self.next_states[idxs], device=self.device, dtype=torch.float32)
        batch_actions = torch.as_tensor(self.actions[idxs], device=self.device, dtype=torch.int64)
        batch_rewards = torch.as_tensor(self.rewards[idxs], device=self.device, dtype=torch.float32)
        batch_dones = torch.as_tensor(self.dones[idxs], device=self.device, dtype=torch.bool)
        # Creamos la mascara de no finalizados.
        non_final_mask = ~batch_dones 
        # retornamos el lote completo.
        return batch_states, batch_actions.unsqueeze(1), batch_rewards, batch_next_states, batch_dones, non_final_mask

    # Retorna el tamaño actual del buffer.
    def __len__(self):
        return self.size



# Definición de buffer de repeticion con Prioridades.
class PrioritizedReplayBuffer:
    # Constructor.
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Cuánto priorizamos (0 = nada, 1 = full)
        self.epsilon = 0.01 # Pequeño valor para que ninguna prioridad sea 0
    
    # Guarda una transicion en el buffer.
    def push(self, state, action, reward, next_state, done):
        # Guardamos la transicion.
        transition = (state, action, reward, next_state, done)
        # Al añadir una nueva experiencia, le damos la prioridad máxima actual para asegurar que se entrene al menos una vez.
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0
        # Añadimos la transicion con la prioridad maxima.
        self.tree.add(max_p, transition)

    # Muestra un lote de transiciones, junto con sus indices y pesos IS.
    def sample(self, batch_size, beta=0.4):
        batch_indices = []
        batch_priorities = []
        batch_transitions = []
        # Dividimos el rango total de prioridades en segmentos
        segment = self.tree.total() / batch_size
        # Para cada segmento, muestreamos una prioridad.
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            # Obtenemos la transicion correspondiente a esa prioridad.
            (idx, p, data) = self.tree.get(s)
            # Guardamos los datos del batch.
            batch_indices.append(idx)
            batch_priorities.append(p)
            batch_transitions.append(data)
        # Desempaquetar batch.
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch_transitions)
        # Convertir a numpy arrays eficientes.
        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch).reshape(-1, 1)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        done_batch = np.array(done_batch)
        # Calcular los pesos de importancia.
        sampling_probabilities = np.array(batch_priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max() # Normalizar para estabilidad
        # Retornamos los índices y los pesos
        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch, 
                batch_indices, np.array(is_weights, dtype=np.float32))

    # Actualiza las prioridades de las transiciones muestreadas.
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            p = (error + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

    # Retorna el tamaño actual del buffer.
    def __len__(self):
        return self.tree.n_entries
    


# En PPO difiere un poco el buffer: solo guarda la experiencia inmediata hecha por la politica reciente.
class PPOBuffer: 
    def __init__(self, device: str = "cpu"):
        self.states = []
        self.actions = []
        self.rewards = []
        self.size = 0
        self.values = [] # Valor estimado por el critico.
        self.log_probs = [] # Probabilidades logaritmicas.
        self.dones = [] # Indicador de finalizacion episodio.
        self.device = device
    
    def push(self, state,action,reward,value,log_prob,done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.size += 1
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.size = 0
    
    def __len__(self):
        return self.size
        
    def sample(self): # No se (?)
        batch_states = torch.as_tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        batch_actions = torch.as_tensor(np.array(self.actions), dtype=torch.int64, device=self.device).unsqueeze(1)
        batch_rewards = torch.as_tensor(np.array(self.rewards), dtype=torch.float32, device=self.device)
        batch_values = torch.as_tensor(np.array(self.values), dtype=torch.float32, device=self.device)
        batch_log_probs = torch.as_tensor(np.array(self.log_probs), dtype=torch.float32, device=self.device).unsqueeze(1)
        batch_dones = torch.as_tensor(np.array(self.dones), dtype=torch.bool, device=self.device)
        non_final_mask = ~batch_dones.bool()
        return batch_states, batch_actions, batch_rewards, batch_values, batch_log_probs, batch_dones, non_final_mask
