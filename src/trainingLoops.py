import torch
import os
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from agents import DQNAgent, PPOAgent



# Función principal de entrenamiento del agente Atari.
def DQN_train_loop(env: gym.Env, agent: DQNAgent, n_episodes: int, batch_size: int, max_episode_length: int, save_model_interval: int):
    # Crea el escritor de TensorBoard.
    writer = SummaryWriter('runs/'+ agent.__class__.__name__) 
    # Crear directorio de modelos si no existe.
    makedirs = os.path.join('models', agent.__class__.__name__)
    if not os.path.exists(makedirs):
        os.makedirs(makedirs)
    # episodio inicial.
    start_episode = 0
    # Bucle principal de entrenamiento.
    for episode in range(start_episode, n_episodes):
        # Reinicia el entorno.
        observation, _ = env.reset()
        # Recompensa total del episodio.
        total_reward = 0.0
        # Calcula el valor de epsilon para este episodio.
        eps = agent.epsilon(episode)
        # Contador de pasos por episodio y de pasos totales.
        steps = 0
        # Bucle principal del episodio.
        for _ in range(max_episode_length):
            # Acción epsilon-greedy
            action = agent.next_action(observation, epsilon=eps)
            # Ejecuta acción en el entorno.
            next_observation, reward, terminated, truncated, info = env.step(action)
            # Revisar si el episodio ha terminado.
            done = truncated or terminated
            # Almacena la transición.
            agent.new_transition(observation, action, reward, next_observation, done)
            # Optimiza la red Q.
            agent.optimize(batch_size)
            # Actualiza recompensas y contadores.
            total_reward += reward
            observation = next_observation
            steps += 1
            # Si el episodio ha terminado, salir del bucle.
            if done:
                break
        # Loguea en TensorBoard.
        writer.add_scalar('Recompensa Total por Episodio', float(total_reward), episode)
        writer.add_scalar('Epsilon por Episodio', agent.epsilon(episode=episode), episode)
        writer.add_scalar('Pasos por Episodio', steps, episode)
        # Print de progreso en consola cada cierto tiempo.
        if episode % 20 == 0:
            print(f"Episodio: {episode} | Recompensa: {total_reward:.2f} | Epsilon: {eps:.3f}")
        # Guarda el modelo cada n episodios.
        if episode % save_model_interval == 0 and episode > 0:
            torch.save(agent.policy_net.state_dict(), os.path.join(makedirs, f'in_progress_model_{episode}.pth'))
    # Cierra el escritor de TensorBoard y el entorno.
    writer.close()
    env.close()
    print("Entrenamiento completado.")



# Función principal de entrenamiento del agente PPO.
def PPO_train_loop(env: gym.Env, agent: PPOAgent, n_episodes: int, batch_size: int, epochs: int, max_episode_length: int, save_model_interval: int):
    writer = SummaryWriter('runs/'+ agent.__class__.__name__) 
    start_episode = 0
    for episode in range(start_episode, n_episodes):
        # Reinicia el entorno.
        observation, _ = env.reset()
        # Recompensa total del episodio.
        total_reward = 0.0
        # Contador de pasos por episodio y de pasos totales.
        steps = 0
        # Bucle principal del episodio.
        for _ in range(max_episode_length):
            
            action, log_prob, value = agent.next_action(observation)
            # Ejecuta acción en el entorno.
            next_observation, reward, terminated, truncated, info = env.step(action)
            # Revisar si el episodio ha terminado.
            done = truncated or terminated
            # Almacena la transición con value y log_prob.
            agent.new_transition(observation, action, reward, next_observation, done, value, log_prob)
            # Optimiza la red PPO.
            agent.optimize(batch_size, epochs)
            # Actualiza recompensas y contadores.
            total_reward += reward
            observation = next_observation
            steps += 1
            # Si el episodio ha terminado, salir del bucle.
            if done:
                break
        # Loguea en TensorBoard.
        writer.add_scalar('Recompensa Total por Episodio', float(total_reward), episode)
        writer.add_scalar('Pasos por Episodio', steps, episode)
        # Print de progreso en consola cada cierto tiempo.
        if episode % 20 == 0:
            print(f"Episodio: {episode} | Recompensa: {total_reward:.2f}")
        # Guarda el modelo cada n episodios.
        if episode % save_model_interval == 0 and episode > 0:
            torch.save(agent.policy_net.state_dict(), f'PPO_model_{episode}.pth')
    # Cierra el escritor de TensorBoard y el entorno.
    writer.close()
    env.close()
    print("Entrenamiento completado.")

