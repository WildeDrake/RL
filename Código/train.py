from collections import deque

import torch
import gymnasium as gym
from agent import AtariAgent
from torch.utils.tensorboard import SummaryWriter
from utils import convert_observation


def train(env: gym.Env, agent: AtariAgent, n_episodes: int, batch_size: int, max_episode_length: int):
    """
    Entrena al agente Atari utilizando el entorno especificado.

    Args:
        max_episode_length: El número máximo de pasos permitidos en un episodio.
        env (gym.Env): El entorno de Gym utilizado para el entrenamiento.
        agent (AtariAgent): El AtariAgent que será entrenado.
        n_episodes (int): El número de episodios para entrenar al agente.
        batch_size (int): El tamaño del lote utilizado para la optimización de la red Q.
    """

    rewards = deque(maxlen=100)
    writer = SummaryWriter()
    for episode in range(n_episodes):
        # Reinicia el entorno para un nuevo episodio y convierte la observación inicial.
        observation, _ = env.reset()
        observation = convert_observation(observation)
        total_reward = 0.0

        for _ in range(max_episode_length):
            # Elige la siguiente acción utilizando la política epsilon-greedy del agente.
            action = agent.next_action(observation, epsilon=0.02)

            # Tomar la acción elegida en el entorno y recibir la siguiente observación y recompensa.
            next_observation, reward, terminated, truncated, info = env.step(action)

            done = truncated or terminated
            next_observation = convert_observation(next_observation)

            total_reward += reward  # ignorar el tipo de dato.
            rewards.append(reward)
            reward = torch.tensor([reward])

            # Almacena la transición en la memoria de repetición.
            agent.new_transition(
                observation, action, reward, next_observation, done
            )

            # Actualiza la observación actual.
            observation = next_observation

            # Realiza un paso de optimización de la red Q.
            agent.optimise(batch_size)

            if done:
                break

        # Registra la relacion de episodios en TensorBoard.
        writer.add_scalar('Reward', total_reward, episode)
        writer.add_scalar('Epsilon', agent.epsilon(), episode)
        writer.flush()

        # Guardar el modelo cada 100 episodios.
        if episode % 100 == 0: 
            torch.save(agent.policy_net.to('cpu'), f'in_progress_model_{episode}')
            agent.policy_net.to(agent.device)
    writer.close()
    env.close()
