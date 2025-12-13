import random
import os
import torch
import gymnasium as gym
from agents import DQN, PPOAgent
from utils import wrap_env, NoopStart
import ale_py


# Ejecutar el modo de prueba del agente Atari
def testing(config_data, agent_type, max_steps_per_episode=2000):
    env_name = config_data.get('env')  # Nombre del entorno
    model_path = config_data.get('model_path')  # Ruta del modelo entrenado
    episodes = int(config_data.get('episodes'))  # Número de episodios de prueba
    video_folder = config_data.get('video_folder')  # Carpeta para guardar los videos

    # Cuda o MPS si está disponible, de lo contrario CPU
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if hasattr(torch, "has_mps") and torch.has_mps
        else torch.device("cpu")
    )
    # Registro del entorno Atari
    gym.register_envs(ale_py)
    env = gym.make(env_name, render_mode='rgb_array')
    env = NoopStart(env)
    env = wrap_env(env)
    # Grabación de video opcional
    if video_folder:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda ep_id: True,
            disable_logger=True
        )
    # Selección del modelo según el tipo de agente
    if agent_type == "DDQN":
        print("Probando agente: Double DQN")
    elif agent_type == "DQN":
        print("Probando agente: DQN")
        policy_net = DQN(env.action_space.n)
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
    policy_net.to(device)
    # Modo evaluación (sin gradientes)
    policy_net.eval()
    # Ejecutar los episodios de prueba
    for ep in range(episodes):
        total_reward = 0
        observation, _ = env.reset()
        for step in range(max_steps_per_episode):
            # Convertir a tensor solo cuando se usa en la red
            obs_t = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            # Selección de acción greedy
            with torch.no_grad():
                action = policy_net(obs_t).max(1)[1].view(1, 1)
            # Paso en el entorno
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            # Actualizar 
            done = terminated or truncated
            if done:
                break
        print(f"Episodio {ep+1}/{episodes} - Recompensa total: {total_reward}")
    # Cierra el entorno al final
    env.close()