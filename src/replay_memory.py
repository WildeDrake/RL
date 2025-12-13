from collections import namedtuple, deque
import numpy as np
import torch

# Estructura para almacenar una transición (experiencia)
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

# Memoria de repetición para almacenar y muestrear transiciones pasadas.
class ReplayMemory:
    # Inicializa la memoria con una capacidad máxima y un dispositivo para los tensores.
    def __init__(self, capacity: int, device: str = "cuda") -> None:
        self.memory = deque(maxlen=capacity)
        self.device = device

    # Agrega una transición a la memoria de repetición.
    def push(self, state, action, reward, next_state, done):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32)
        if next_state is not None and not torch.is_tensor(next_state):
            next_state = torch.as_tensor(next_state, dtype=torch.float32)
        if not torch.is_tensor(action):
            action = torch.as_tensor([action], dtype=torch.long)
        if not torch.is_tensor(reward):
            reward = torch.as_tensor([reward], dtype=torch.float32)
        done = torch.as_tensor([done], dtype=torch.bool)
        # Guardar la transición
        self.memory.append(Transition(state, action, reward, next_state, done))

    # Muestrea un lote de transiciones aleatorias y las convierte a tensores en el dispositivo.
    def sample(self, batch_size: int, device: str = "cuda"):
        if len(self.memory) < batch_size:
            raise ValueError("La memoria no contiene suficientes transiciones para muestrear.")
        # Selección aleatoria mediante índices tensoriales
        idx = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in idx]
        batch = Transition(*zip(*batch))
        # Máscara de los next_states que no son None
        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            dtype=torch.bool,
            device=device
        )
        # Stack sólo los next_states válidos
        non_final_next_states = [s for s in batch.next_state if s is not None]
        if len(non_final_next_states) > 0:
            next_state_batch = torch.stack(non_final_next_states).to(device, non_blocking=True)
        else:
            next_state_batch = torch.empty((0,), dtype=torch.float32, device=device)
        # Devuelve los lotes y la máscara   
        return torch.stack(batch.state).to(device, non_blocking=True), torch.cat(batch.action).to(device, non_blocking=True), torch.cat(batch.reward).to(device, non_blocking=True), next_state_batch, torch.cat(batch.done).to(device, non_blocking=True), non_final_mask


    # Devuelve la cantidad actual de transiciones almacenadas.
    def __len__(self):
        """Devuelve la cantidad actual de transiciones almacenadas."""
        return len(self.memory)
