from collections import namedtuple, deque
import torch

# Estructura para almacenar una transición (experiencia)
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayMemory:
    # Memoria de repetición para almacenar y muestrear transiciones pasadas.

    def __init__(self, capacity: int) -> None:
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Agrega una transición a la memoria de repetición.
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32)
        if next_state is not None and not torch.is_tensor(next_state):
            next_state = torch.as_tensor(next_state, dtype=torch.float32)
        if not torch.is_tensor(action):
            action = torch.as_tensor([action], dtype=torch.long)
        if not torch.is_tensor(reward):
            reward = torch.as_tensor([reward], dtype=torch.float32)
        done = torch.as_tensor([done], dtype=torch.bool)
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: str = "cuda"):
        # Muestrea un lote de transiciones aleatorias.
        if len(self.memory) < batch_size:
            raise ValueError("La memoria no contiene suficientes transiciones para muestrear.")
        # Selección aleatoria mediante índices tensoriales
        idx = torch.randint(0, len(self.memory), (batch_size,))
        batch = [self.memory[i] for i in idx]
        batch = Transition(*zip(*batch))
        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state], dtype=torch.bool, device=device
        )

        # Convierte listas a tensores (solo pin_memory si están en CPU)
        state_batch = torch.stack(batch.state, dim=0)
        if state_batch.device.type == "cpu":
            state_batch = state_batch.pin_memory()
        state_batch = state_batch.to(device, non_blocking=True)

        action_batch = torch.cat(batch.action).to(device, non_blocking=True)
        reward_batch = torch.cat(batch.reward).to(device, non_blocking=True)

        next_state_batch = torch.stack([s for s in batch.next_state if s is not None], dim=0) if any(non_final_mask) else torch.empty(0)
        if next_state_batch.device.type == "cpu":
            next_state_batch = next_state_batch.pin_memory()
        next_state_batch = next_state_batch.to(device, non_blocking=True) if any(non_final_mask) else next_state_batch

        done_batch = torch.cat(batch.done).to(device, non_blocking=True)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, non_final_mask

    def __len__(self):
        # Devuelve la cantidad actual de transiciones almacenadas.
        return len(self.memory)
