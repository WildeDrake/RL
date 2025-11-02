from collections import deque
import torch
import gymnasium as gym
from agent import DQNAgent, DDQNAgent
from torch.utils.tensorboard import SummaryWriter
from utils import convert_observation


# Función principal de entrenamiento del agente Atari.
def train(env: gym.Env, agent: DQNAgent, n_episodes: int, batch_size: int, max_episode_length: int):
    """
    Entrena un agente DQN/DDQN en un entorno de Atari.

    Args:
        env (gym.Env): Entorno de Gymnasium.
        agent (DQNAgent | DDQNAgent): Agente a entrenar.
        n_episodes (int): Número de episodios de entrenamiento.
        batch_size (int): Tamaño del lote para la actualización de la red.
        max_episode_length (int): Longitud máxima de cada episodio.
    """
    rewards = deque(maxlen=100)  # Mantiene las últimas 100 recompensas para el promedio.
    writer = SummaryWriter('runs/frogger')  # Para visualización en TensorBoard.

    for episode in range(n_episodes):
        # Reinicia el entorno para un nuevo episodio y convierte la observación inicial a tensor.
        observation, _ = env.reset()
        observation = convert_observation(observation, device=agent.device)

        # Inicializa la recompensa total del episodio.
        total_reward = 0.0

        for _ in range(max_episode_length):
            # Selecciona la siguiente acción utilizando la política epsilon-greedy del agente.
            with torch.no_grad():
                action = agent.next_action(observation)

            # Ejecuta la acción en el entorno y obtiene la siguiente observación y recompensa.
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated

            # Convierte la siguiente observación a tensor y mueve a GPU si aplica.
            next_observation = convert_observation(next_observation, device=agent.device)

            # Acumula la recompensa y actualiza la lista de recompensas recientes.
            total_reward += reward
            rewards.append(reward)

            # Convierte la recompensa a tensor y almacena la transición en la memoria de repetición.
            reward_t = torch.tensor([reward], dtype=torch.float32, device=agent.device)
            agent.new_transition(observation, action, reward_t, next_observation, done)

            # Optimiza la red Q usando un lote de la memoria de repetición.
            agent.optimise(batch_size)

            # Actualiza la observación actual.
            observation = next_observation

            # Si el episodio terminó, salir del bucle.
            if done:
                break

        # Calcula la recompensa promedio de los últimos 100 episodios y la loguea en TensorBoard.
        avg_reward = sum(rewards) / len(rewards)
        writer.add_scalar('Average Reward (100 episodes)', avg_reward, episode)

        # Guardar el modelo cada 100 episodios.
        if episode % 100 == 0 and episode > 0:
            torch.save(agent.policy_net.state_dict(), f'in_progress_model_{episode}.pth')

    # Cierra el escritor de TensorBoard y el entorno.
    writer.close()
    env.close()
