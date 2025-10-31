from collections import namedtuple, deque
import random

# Defines a named tuple 'Transition' to represent a single transition in the replay memory.
# A transition contains a state, an action, a reward, and the next state.
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))

class ReplayMemory:
    """
    Replay Memory para almacenar y muestrear experiencias pasadas (transiciones) para entrenar agentes de aprendizaje por refuerzo.
    
    Args:
        capacity (int): La capacidad máxima de la memoria de repetición.
    """
    def __init__(self, capacity: int) -> None:
        """
        Inicializa la memoria de repetición con una capacidad especificada.
        
        Args:
            capacity (int): La capacidad máxima de la memoria de repetición.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """
        Agrega una nueva transición a la memoria de repetición.
        
        Args:
            *args: Una tupla que contiene un estado, una acción, una recompensa y el siguiente estado.
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Muestrea aleatoriamente un lote de transiciones de la memoria de repetición.
        
        Args:
            batch_size (int): El número de transiciones a muestrear en el lote.
        Returns:
            list: Una lista de transiciones muestreadas.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Retorna el número actual de transiciones almacenadas en la memoria de repetición.
        
        Returns:
            int: El número de transiciones almacenadas.
        """
        return len(self.memory)

