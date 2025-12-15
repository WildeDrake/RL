import torch
import gymnasium as gym
import ale_py

from agents import DQNAgent, PPOAgent
from utils import make_dqn_env, make_ppo_env
from trainingLoops import DQN_train_loop, PPO_train_loop



def train(config_data, agent_type):
    # Cargar parametros de configuracion
    env_name = config_data.get('env')                                   # Nombre del entorno.
    eps_start = float(config_data.get('eps_start'))                     # Valor inicial de epsilon.
    eps_end = float(config_data.get('eps_end'))                         # Valor final de epsilon.
    eps_decay = float(config_data.get('eps_decay'))                     # Tasa de decaimiento de epsilon.
    memory_size = int(config_data.get('memory_size'))                   # Tamaño de la memoria de experiencia.
    learning_rate = float(config_data.get('learning_rate'))             # Tasa de aprendizaje.
    initial_memory = int(config_data.get('initial_memory'))             # Memoria inicial antes de entrenar.
    gamma = float(config_data.get('gamma'))                             # Factor de descuento.
    target_update = int(config_data.get('target_update'))               # Frecuencia de actualizacion de la red objetivo.
    batch_size = int(config_data.get('batch_size'))                     # Tamaño del lote para el entrenamiento.
    model_path = config_data.get('model_path')                          # Ruta para guardar el modelo entrenado.
    episodes = int(config_data.get('episodes'))                         # Número de episodios.
    max_episode_length = int(config_data.get('max_episode_length'))     # Longitud maxima de un episodio.
    save_model_interval = int(config_data.get('save_model_interval'))   # Intervalo para guardar el modelo.
    # Cuda o MPS si esta disponible, de lo contrario CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Usando dispositivo: {device}")
    # Registro del entorno Atari.
    gym.register_envs(ale_py)
    # Seleccion dinamica del agente.
    if agent_type == "DQN":
        env = make_dqn_env(env_name)
        # Obtener la forma de la entrada del entorno.
        agent = DQNAgent(
            device=device,
            n_actions=env.action_space.n,
            lr=learning_rate,
            epsilon_start=eps_start,
            epsilon_end=eps_end,
            epsilon_decay=eps_decay,
            total_memory=memory_size,
            initial_memory=initial_memory,
            gamma=gamma,
            target_update=target_update,
            network_file=model_path,
            input_shape=env.observation_space.shape
        )
        DQN_train_loop(env, agent, episodes, batch_size, max_episode_length, save_model_interval)
        print("Entrenando agente DQN")
    elif agent_type == "PPO":
        env = make_ppo_env(env_name, seed=42)
        # Obtener la forma de la entrada del entorno.
        agent = PPOAgent(
            device=device,
            n_actions=env.action_space.n,
            lr=learning_rate,
            clippping_epsilon=0.2,
            total_memory=memory_size,
            initial_memory=initial_memory,
            gamma=gamma,
            target_update=target_update,
            network_file=model_path,
            input_shape=env.observation_space.shape
        )
        PPO_train_loop(env, agent, episodes, batch_size, max_episode_length)
        print("Entrenando agente PPO")
    else:
        raise ValueError(f"Tipo de agente no reconocido: {agent_type}")
    # Guardar el modelo entrenado
    print(f"Guardando modelo entrenado en '{model_path}'...")
    torch.save(agent.policy_net.state_dict(), model_path)
    # Liberar memoria del entorno al terminar
    env.close()