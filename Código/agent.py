from torch.optim import Adam

from typing import Optional
import model
from replay_memory import ReplayMemory, Transition
from model import DQN
import math
import torch
import random


class AtariAgent:
    def __init__(
        self,
        device: torch.device,
        n_actions: int,
        lr: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        total_memory: int,
        initial_memory: int,
        gamma: float,
        target_update: int,
        network_file=None,
    ) -> None:
        """
        Crea un nuevo agente Atari.

        Args:
            device (torch.device): El dispositivo (CPU o GPU) en el que se ejecutará el agente.
            n_actions (int): El número de acciones posibles que el agente puede tomar.
            lr (float): La tasa de aprendizaje para el optimizador de la red neuronal del agente.
            epsilon_start (float): La tasa de exploración inicial para la política epsilon-greedy.
            epsilon_end (float): La tasa de exploración final para la política epsilon-greedy.
            epsilon_decay (float): La tasa a la que decae el épsilon con el tiempo.
            total_memory (int): La capacidad máxima de la memoria de repetición.
            intial_memory (int): El número mínimo de transiciones requeridas en la memoria de repetición antes de que comience el aprendizaje.
            gamma (float): El factor de descuento para las recompensas futuras en la actualización de Q-learning.
            target_update (int): La frecuencia con la que se actualiza la red objetivo.
            network_file (str, opcional): Un archivo para cargar los pesos de la red pre-entrenada.
        """
        self.n_actions = n_actions
        self.device = device

        # Deep Q network para la política y el objetivo, con optimizador para la política.
        if network_file:
            self.policy_net = torch.load(network_file).to(self.device)
            self.target_net = torch.load(network_file).to(self.device)
        else:
            # Crea nuevas redes de política y objetivo si no se proporcionan pesos pre-entrenados.
            self.policy_net = DQN(self.n_actions).to(device)
            self.target_net = DQN(self.n_actions).to(device)

            self.policy_net.apply(model.init_weights)
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizador para la red de política.
        self.optimiser = Adam(self.policy_net.parameters(), lr=lr)

        self.steps_done = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.target_update = target_update

        # Inicializa la memoria de repetición.
        self.memory = ReplayMemory(capacity=total_memory)
        self.initial_memory = initial_memory

    def new_transition(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: bool,
    ):
        """
        Almacena una nueva transición en la memoria de repetición.

        Args:
            observation (torch.Tensor): La observación/estado actual.
            action (torch.Tensor): La acción tomada.
            reward (torch.Tensor): La recompensa recibida.
            next_observation (torch.Tensor): La siguiente observación/estado.
            done (bool): Una bandera que indica si el episodio ha terminado.
        """

        self.memory.push(
            observation.to('cpu'),
            action.to('cpu'),
            reward.to('cpu'),
            next_observation.to('cpu') if not done else None,
        )

    def epsilon(self):
        """
        Calcula el umbral épsilon actual para la política épsilon-greedy.
        
        Return:
            float: El valor épsilon actual.
        """
        return self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * (1 - min(self.steps_done / self.epsilon_decay, 1))

    def next_action(
        self, observation: torch.Tensor, epsilon: Optional[float] = None
    ) -> torch.Tensor:
        """
            Calcula la siguiente acción a tomar utilizando una política épsilon-greedy.

            Args:
                observation (torch.Tensor): La observación/estado actual.
                epsilon (float, opcional): El valor épsilon a utilizar para la política épsilon-greedy.

            Returns:
                torch.Tensor: La acción elegida.
        """
        if epsilon is None:
            epsilon = self.epsilon()

        self.steps_done += 1

        if random.random() > epsilon:
            with torch.no_grad():
                return (
                    self.policy_net(observation.to(self.device))
                    .max(1)[1]
                    .view(1, 1)
                )
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=self.device,
                dtype=torch.long,
            )

    def optimise(self, batch_size: int):
        """
            Realiza un paso de optimización de la red Q.

            Args:
                batch_size (int): El número de transiciones a muestrear y utilizar para la optimización.
        """

        # Solo comienza a optimizar una vez que haya suficientes transiciones en la memoria.
        if (
            len(self.memory) < self.initial_memory
            or len(self.memory) < batch_size
        ):
            return

        # Muestrea un lote de transiciones de la memoria de repetición.
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Crea una máscara para los estados no finales en el lote.
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )

        # Extrae los siguientes estados no finales y los convierte en un tensor.
        non_final_next_states = torch.stack(
            [s for s in batch.next_state if s is not None]
        ).to(self.device)

        # Convierte los estados, acciones y recompensas en el lote en tensores.
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # Computa los valores Q predichos para los pares estado-acción en el lote.
        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch
        )

        # Inicializa los valores del siguiente estado como ceros.
        next_state_values = torch.zeros(batch_size, device=self.device)
        # Actualiza los valores del siguiente estado con los valores de la red objetivo para los estados no finales.
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0].detach()
        )

        # Computa los valores esperados de estado-acción utilizando la ecuación de Bellman.
        expected_state_action_values = (
            next_state_values * self.gamma
        ) + reward_batch

        # Computa la pérdida utilizando la pérdida de Huber (pérdida L1 suave).
        loss = torch.nn.functional.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Cero los gradientes de los parámetros de la red de política.
        self.optimiser.zero_grad()

        # Computa los gradientes y realiza el recorte de gradientes.
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        # Actualiza los parámetros de la red de política.
        self.optimiser.step()

        # Actualiza los parámetros de la red objetivo si se alcanza el intervalo de actualización del objetivo.
        if self.steps_done % self.target_update == 1:
            self.target_net.load_state_dict(self.policy_net.state_dict())
