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