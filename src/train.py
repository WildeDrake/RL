from collections import deque
import torch
import gymnasium as gym
from agent import DQNAgent, DDQNAgent
from torch.utils.tensorboard import SummaryWriter
from utils import convert_observation


# Función principal de entrenamiento del agente Atari.
def train(env: gym.Env, agent: DQNAgent, n_episodes: int, batch_size: int, max_episode_length: int):
    rewards = deque(maxlen=100)
    writer = SummaryWriter('runs/frogger')
    for episode in range(n_episodes):
        # Reinicia el entorno para un nuevo episodio y convierte la observación inicial.
        observation, _ = env.reset()
        observation = convert_observation(observation)
        # Inicializa la recompensa total del episodio.
        for _ in range(max_episode_length):
            # Elige la siguiente acción utilizando la política epsilon-greedy del agente.
            action = agent.next_action(observation)
            # Tomar la acción elegida en el entorno y recibir la siguiente observación y recompensa.
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated
            next_observation = convert_observation(next_observation)
            # Almacena la recompensa obtenida.
            reward_t = torch.tensor([reward], dtype=torch.float32)
            rewards.append(reward)
            # Almacena la transición en la memoria de repetición.
            agent.new_transition(observation, action, reward_t, next_observation, done)
            # Realiza un paso de optimización de la red Q.
            agent.optimise(batch_size)
            # Actualiza la observación actual.
            observation = next_observation
            # Verifica si el episodio ha terminado.
            if done:
                break
        # Calcula la recompensa promedio de los últimos 100 episodios.
        avg_reward = sum(rewards) / len(rewards)
        writer.add_scalar('Average Reward (100 episodes)', avg_reward, episode)
        writer.flush()
        # Guardar el modelo cada 100 episodios.
        if episode % 100 == 0: 
            torch.save(agent.policy_net.state_dict(), f'in_progress_model_{episode}.pth')
    # Cierra el escritor de TensorBoard y el entorno.
    writer.close()
    env.close()