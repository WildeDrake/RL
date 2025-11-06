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
    writer = SummaryWriter('runs/'+ agent.__class__.__name__)  # Para visualización en TensorBoard.

    start_episode = 900
    for episode in range(start_episode, n_episodes):
        # Reinicia el entorno
        observation, _ = env.reset()
        observation = convert_observation(observation, device=agent.device)
        total_reward = 0.0           # recompensa original
        total_shaped_reward = 0.0    # recompensa con shaping
        eps = agent.epsilon(episode)

        progress = 0  # contador de avance para reward shaping
        steps = 0
        for _ in range(max_episode_length):
            # Acción epsilon-greedy
            action = agent.next_action(observation, epsilon=eps)

            # Ejecuta acción en el entorno
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated

            # Reward shaping: solo se aplica al avance
            shaped_reward = reward
            if reward == 1:  # avanzó un bloque
                progress += 1
                shaped_reward += 0.01 * progress  # bonus creciente
            elif reward == 0:
                progress = 0  # resetea si no avanzó

            # Actualiza totales
            total_reward += reward
            total_shaped_reward += shaped_reward

            # Convertir siguiente observación a tensor
            next_observation = convert_observation(next_observation, device=agent.device)

            # Almacena la transición usando **shaped_reward**
            reward_t = torch.tensor([shaped_reward], dtype=torch.float32, device=agent.device)
            agent.new_transition(observation, action, reward_t, next_observation, done)

            # Optimiza la red Q
            agent.optimize(batch_size)

            observation = next_observation
            steps += 1
            if done:
                break

        # Loguea en TensorBoard
        writer.add_scalar('Recompensa Total por Episodio', total_reward, episode)
        writer.add_scalar('Recompensa Shaped por Episodio', total_shaped_reward - total_reward, episode)
        writer.add_scalar('Progreso por Episodio', progress, episode)
        writer.add_scalar('Epsilon por Episodio', agent.epsilon(episode=episode), episode)
        writer.add_scalar('Pasos por Episodio', steps, episode)
        writer.add_scalar('GPU Memory Allocated por Episodio', torch.cuda.memory_allocated() / 1024**2, episode)
        writer.add_scalar('GPU Memory Reserved por Episodio', torch.cuda.memory_reserved() / 1024**2, episode)

        if episode % 250 == 0 and episode > 0:
            torch.save(agent.policy_net.state_dict(), f'in_progress_model_{episode}.pth')
   
   
    # Cierra el escritor de TensorBoard y el entorno.
    writer.close()
    env.close()
