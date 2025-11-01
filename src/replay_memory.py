from collections import namedtuple, deque
import torch
import random


# Definición de la estructura de datos
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


# Replay Memory para almacenar y muestrear experiencias pasadas (transiciones) 
# para entrenar agentes de aprendizaje por refuerzo.
class ReplayMemory:
    # Inicialización de la memoria de repetición. capacity define el tamaño máximo de la memoria.
    def __init__(self, capacity: int) -> None:
        self.memory = deque(maxlen=capacity)

    # Agrega una nueva transición a la memoria de repetición.
    def push(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done))

    # Muestrea un lote de transiciones aleatorias de la memoria de repetición. batch_size define el tamaño del lote.
    def sample(self, batch_size, device="cpu"):
        # Selección aleatoria de transiciones
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        # Conversión de listas de tensores a tensores individuales
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)
        done_batch = torch.cat(batch.done).to(device)
        # Devolvemos el lote de transiciones
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    # Retorna el número actual de transiciones almacenadas en la memoria de repetición.
    def __len__(self):
        return len(self.memory)