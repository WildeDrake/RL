import os
import torch
import gymnasium as gym
import ale_py
import numpy as np

from models import DQN
from utils import make_dqn_env


# Ejecutar el modo de prueba del agente Atari
def test(config_data, agent_type):
    env_name = config_data.get('env')                                       # Nombre del entorno
    model_path = config_data.get('model_path')                              # Ruta del modelo entrenado
    episodes = int(config_data.get('episodes'))                             # Número de episodios de prueba
    video_folder = config_data.get('video_folder')                          # Carpeta para guardar los videos
    max_steps_per_episode = int(config_data.get('max_steps_per_episode'))   # Longitud máxima de un episodio de prueba
    # Cuda o MPS si está disponible, de lo contrario CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # Registro del entorno Atari
    gym.register_envs(ale_py)
    # Grabación de video opcional
    if video_folder:
        os.makedirs(video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda ep_id: True,
            disable_logger=True
        )
    # Selección del modelo según el tipo de agente
    if agent_type == "DQN":
        print("Probando agente: DQN")
        env = make_dqn_env(env_name)
        input_shape = env.observation_space.shape
        policy_net = DQN(env.action_space.n, input_shape).to(device)
    else:
        raise ValueError(f"Tipo de agente no reconocido: {agent_type}")
    # Carga del modelo entrenado si existe
    if os.path.exists(model_path):
        print(f"Cargando modelo desde '{model_path}'...")
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        policy_net.load_state_dict(new_state_dict)
    else:
        print(f"No se encontró el modelo '{model_path}', se ejecutará sin cargar pesos.")
    # Modo evaluación (sin gradientes)
    policy_net.eval()
    # Ejecutar los episodios de prueba
    for ep in range(episodes):
        total_reward = 0
        observation, _ = env.reset()
        # Bucle principal del episodio
        for step in range(max_steps_per_episode):
            # Convertir a tensor solo cuando se usa en la red
            obs_array = np.array(observation, copy=True)
            obs_t = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            # Agregar dimensión de batch (C, H, W) -> (1, C, H, W)
            if obs_t.dim() == 3:
                obs_t = obs_t.unsqueeze(0)
            # Selección de acción greedy
            with torch.no_grad():
                q_values = policy_net(obs_t)
                action = q_values.argmax(dim=1).item()
            # Paso en el entorno
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            # Actualizar 
            if terminated or truncated:
                break
        print(f"Episodio {ep+1}/{episodes} - Recompensa total: {total_reward}")
    # Cierra el entorno al final
    env.close()
    print("Test completado.")