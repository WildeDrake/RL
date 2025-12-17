import torch
import gymnasium as gym
import ale_py

from DQNAgent import DQNAgent
from PPOAgent import PPOAgent
from utils import make_dqn_env, make_ppo_env
from trainingLoops import DQN_train_loop, PPO_train_loop



def train(config_data, agent_type):
    # Cargar parametros de configuracion
    '''---------------------------------------- PARAMETROS DE CONFIGURACION GENERALES----------------------------------------'''
    env_name = config_data.get('env')                                                # Nombre del entorno..
    model_path = config_data.get('model_path')                                       # Ruta para guardar/cargar el modelo.  
    run_name = config_data.get('run_name')                                           # Directorio para guardar graficos del train.
    save_model_interval = int(config_data.get('save_model_interval', fallback=500))  # Intervalo de guardado del modelo.
    '''---------------------------------------- PARAMETROS DE AGENTES----------------------------------------'''
    eps_start = float(config_data.get('eps_start', fallback=1))                     # DQN: Valor inicial de epsilon.
    eps_end = float(config_data.get('eps_end', fallback=0.01))                      # DQN: Valor final de epsilon.
    eps_decay = float(config_data.get('eps_decay', fallback=8000))                  # DQN: Tasa de decaimiento de epsilon.
    memory_size = int(config_data.get('memory_size', fallback=100000))              # DQN: Tamaño de la memoria de experiencia.
    learning_rate = float(config_data.get('learning_rate', fallback=0.0001))        # DQN: Tasa de aprendizaje.
    initial_memory = int(config_data.get('initial_memory', fallback=10000))         # DQN: Memoria inicial antes de entrenar.
    gamma = float(config_data.get('gamma', fallback=0.99))                          # DQN: Factor de descuento.
    target_update = int(config_data.get('target_update', fallback=2000))            # DQN: Frecuencia de actualizacion de la red objetivo.
    max_episode_length = int(config_data.get('max_episode_length', fallback=2000))  # DQN: Longitud maxima de un episodio.
    episodes = int(config_data.get('episodes', fallback=10000))                     # DQN: Número de episodios.
    batch_size = int(config_data.get('batch_size', fallback=128))                 # DQN Y PPO: Tamaño del lote para el entrenamiento
    gae_lambda = float(config_data.get('gae_lambda', fallback=0.95))            # PPO: Lambda para GAE (PPO).
    clip_ratio = float(config_data.get('clip_ratio', fallback=0.1))             # PPO: Factor de descuento.              
    epochs = int(config_data.get('epochs', fallback=4))                         # PPO: Número de épocas para PPO.
    '''---------------------------------------- PARAMETROS DE RAINBOW DQN----------------------------------------'''
    use_double = config_data.getboolean('use_double', fallback=False)                   # Habilitar Double DQN.
    use_dueling = config_data.getboolean('use_dueling', fallback=False)                 # Habilitar Dueling DQN.
    use_per = config_data.getboolean('use_per', fallback=False)                         # Habilitar Prioritized Experience Replay   
    use_multi_step = config_data.getboolean('use_multi_step', fallback=False)           # Habilitar N-step learning
    use_noisy = config_data.getboolean('use_noisy', fallback=False)                     # Habilitar Noisy Nets
    use_distributional = config_data.getboolean('use_distributional', fallback=False)   # Habilitar Distributional RL (C51)
    '''---------------------------------------- Rainbow DQN----------------------------------------'''
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
            input_shape=env.observation_space.shape,
            use_double=use_double,
            use_dueling=use_dueling,
            use_per=use_per,
            use_multi_step=use_multi_step,
            use_noisy=use_noisy,
            use_distributional=use_distributional
        )
        DQN_train_loop(env, agent, episodes, batch_size, max_episode_length, save_model_interval, run_name)
        print("Entrenando agente DQN")
    elif agent_type == "PPO":
        env = make_ppo_env(env_name, seed=42)
        # Obtener la forma de la entrada del entorno.
        agent = PPOAgent(
            device=device,
            n_actions=env.action_space.n,
            lr=learning_rate,
            clipping_epsilon=clip_ratio,
            total_memory=memory_size,
            initial_memory=initial_memory,
            gamma=gamma,
            gae_lambda=gae_lambda,
            network_file=model_path,
            input_shape=env.observation_space.shape
        )
        PPO_train_loop(env, agent, episodes, batch_size, epochs, max_episode_length, save_model_interval, run_name)
        print("Entrenando agente PPO")
    else:
        raise ValueError(f"Tipo de agente no reconocido: {agent_type}")
    # Guardar el modelo entrenado
    print(f"Guardando modelo entrenado en '{model_path}'...")
    torch.save(agent.policy_net.state_dict(), model_path)
    # Liberar memoria del entorno al terminar
    env.close()