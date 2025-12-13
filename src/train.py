import torch
import os
import gymnasium as gym
from agent import DQNAgent, DDQNAgent
from torch.utils.tensorboard import SummaryWriter
from utils import convert_observation, wrap_env, NoopStart
import ale_py


# Función principal de entrenamiento del agente Atari.
def train_loop(env: gym.Env, agent: DQNAgent, n_episodes: int, batch_size: int, max_episode_length: int):
    writer = SummaryWriter('runs/'+ agent.__class__.__name__) 
    start_episode = 0
    for episode in range(start_episode, n_episodes):
        # Reinicia el entorno
        observation, _ = env.reset()
        observation = convert_observation(observation, device=agent.device)
        total_reward = 0.0           # recompensa original
        eps = agent.epsilon(episode)

        
        steps = 0
        for _ in range(max_episode_length):
            # Acción epsilon-greedy
            action = agent.next_action(observation, epsilon=eps)

            # Ejecuta acción en el entorno
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated

            # Actualiza totales
            total_reward += reward

            # Convertir siguiente observación a tensor
            next_observation = convert_observation(next_observation, device=agent.device)

            # Almacena la transición
            reward_t = torch.tensor([reward], dtype=torch.float32, device=agent.device)
            agent.new_transition(observation, action, reward_t, next_observation, done)

            # Optimiza la red Q
            agent.optimize(batch_size)

            observation = next_observation
            steps += 1
            if done:
                break

        # Loguea en TensorBoard
        writer.add_scalar('Recompensa Total por Episodio', float(total_reward), episode)
        writer.add_scalar('Epsilon por Episodio', agent.epsilon(episode=episode), episode)
        writer.add_scalar('Pasos por Episodio', steps, episode)

        if episode % 250 == 0 and episode > 0:
            # tipo de agente
            makedirs = os.path.join('models', agent.__class__.__name__)
            if not os.path.exists(makedirs):
                os.makedirs(makedirs)
            torch.save(agent.policy_net.state_dict(), os.path.join(makedirs, f'in_progress_model_{episode}.pth'))
   
    # Cierra el escritor de TensorBoard y el entorno.
    writer.close()
    env.close()
   
   
    # Cierra el escritor de TensorBoard y el entorno.
    writer.close()
    env.close()


def training(config_data, agent_type):
    # Cargar parámetros de configuración
    env_name = config_data.get('env')  # Nombre del entorno
    actions = int(config_data.get('actions'))  # Número de acciones posibles
    eps_start = float(config_data.get('eps_start'))  # Valor inicial de epsilon
    eps_end = float(config_data.get('eps_end'))  # Valor final de epsilon
    eps_decay = float(config_data.get('eps_decay'))  # Tasa de decaimiento de epsilon
    memory_size = int(config_data.get('memory_size'))  # Tamaño de la memoria de experiencia
    learning_rate = float(config_data.get('learning_rate'))  # Tasa de aprendizaje
    initial_memory = int(config_data.get('initial_memory'))  # Memoria inicial antes de entrenar
    gamma = float(config_data.get('gamma'))  # Factor de descuento
    target_update = int(config_data.get('target_update'))  # Frecuencia de actualización de la red objetivo
    batch_size = int(config_data.get('batch_size'))  # Tamaño del lote para el entrenamiento
    model_path = config_data.get('model_path')  # Ruta para guardar el modelo entrenado
    episodes = int(config_data.get('episodes'))  # Número de episodios
    max_episode_length = int(config_data.get('max_episode_length'))
    # Cuda o MPS si está disponible, de lo contrario CPU
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if hasattr(torch, "has_mps") and torch.has_mps
        else torch.device("cpu")
    )
    print(f"Usando dispositivo: {device}")
    # Carga de entorno gymnasium 
    gym.register_envs(ale_py)
    env = gym.make(env_name, render_mode="rgb_array")
    env = NoopStart(env)
    env = wrap_env(env)
    # Selección dinámica del agente
    if agent_type == "DDQN":
        agent = DDQNAgent(
            device=device,
            n_actions=actions,
            lr=learning_rate,
            epsilon_start=eps_start,
            epsilon_end=eps_end,
            epsilon_decay=eps_decay,
            total_memory=memory_size,
            initial_memory=initial_memory,
            gamma=gamma,
            target_update=target_update,
            network_file=model_path
        )
        print("Entrenando agente: Double DQN")
    elif agent_type == "DQN":
        agent = DQNAgent(
            device=device,
            n_actions=actions,
            lr=learning_rate,
            epsilon_start=eps_start,
            epsilon_end=eps_end,
            epsilon_decay=eps_decay,
            total_memory=memory_size,
            initial_memory=initial_memory,
            gamma=gamma,
            target_update=target_update,
            network_file=model_path
        )
        print("Entrenando agente: DQN")
    else:
        raise ValueError(f"Tipo de agente no reconocido: {agent_type}")
    print("Iniciando entrenamiento...")
    train_loop(env, agent, episodes, batch_size, max_episode_length)
    print("Entrenamiento completado.")
    # Guardar el modelo entrenado
    print(f"Guardando modelo entrenado en '{model_path}'...")
    torch.save(agent.policy_net.state_dict(), model_path)
    # Liberar memoria del entorno al terminar
    env.close()