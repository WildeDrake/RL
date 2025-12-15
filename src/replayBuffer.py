import numpy as np
import torch



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


    #retorna un lote aleatorio de transiciones.
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



# En PPO difiere un poco el buffer: solo guarda la experiencia inmediata hecha por la politica reciente.
class PPOBuffer: 
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.size = 0
        self.values = [] # Valor estimado por el critico.
        self.log_probs = [] # Probabilidades logaritmicas.
        self.dones = [] # Indicador de finalizacion episodio.
    
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
        batch_states = torch.as_tensor(np.array(self.states), dtype=torch.float32)
        batch_actions = torch.as_tensor(np.array(self.actions), dtype=torch.int64).unsqueeze(1)
        batch_rewards = torch.as_tensor(np.array(self.rewards), dtype=torch.float32)
        batch_values = torch.as_tensor(np.array(self.values), dtype=torch.float32).unsqueeze(1)
        batch_log_probs = torch.as_tensor(np.array(self.log_probs), dtype=torch.float32).unsqueeze(1)
        batch_dones = torch.as_tensor(np.array(self.dones), dtype=torch.bool)
        non_final_mask = ~batch_dones
        return batch_states, batch_actions, batch_rewards, batch_values, batch_log_probs, batch_dones, non_final_mask
