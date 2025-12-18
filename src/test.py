import os
import torch
import gymnasium as gym
import ale_py
import numpy as np

from models import DQN, PPO
from utils import make_dqn_env, make_ppo_env



# Funcion de test para el agente DQN.
def testLoopDQN(episodes, max_steps_per_episode, env, policy_net, device, use_distributional=False):
    # Configuración auxiliar para C51
    if use_distributional == False:
        support = None
    else:
        support = torch.linspace(0.0, 20.0, 51).to(device)

    # Ejecutar los episodios de prueba.
    for ep in range(episodes):
        total_reward = 0
        observation, _ = env.reset()
        # Bucle principal del episodio.
        for step in range(max_steps_per_episode):
            # Convertir la observación a tensor.
            obs_t = torch.tensor(observation, dtype=torch.float32, device=device)
            # Agregar dimension de batch (C, H, W) -> (1, C, H, W).
            if obs_t.dim() == 3:
                obs_t = obs_t.unsqueeze(0)
            # Seleccion de accion greedy.
            with torch.no_grad():
                if use_distributional == False:
                    # Se obtiene la acción con el mayor valor Q.
                    q_values = policy_net(obs_t)
                    action = q_values.argmax(dim=1).item()
                else:
                    # La red devuelve probabilidades (1, n_actions, 51)
                    dist = policy_net(obs_t)
                    # Calculamos valor esperado: sum(prob * valor)
                    expected_value = (dist * support).sum(dim=2)
                    action = expected_value.argmax(dim=1).item()
            # Paso en el entorno.
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            # Terminar el episodio si es necesario.
            if terminated or truncated:
                break
        print(f"Episodio {ep+1}/{episodes} - Recompensa total: {total_reward}")



# Funcion de test para el agente PPO.
def testLoopPPO(episodes, max_steps_per_episode, env, policy_net, device):
    pass



# Ejecutar el modo de prueba del agente Atari.
def test(config_data):

    # Cargar parametros de configuracion
    env_name = config_data.get('env')                                       # Nombre del entorno
    agent_type = config_data.get('agent')                                   # Tipo de agente (DQN o PPO)   
    model_path = config_data.get('model_path')                              # Ruta del modelo entrenado
    video_folder = config_data.get('video_folder')                          # Carpeta para guardar los videos
    episodes = int(config_data.get('episodes', fallback=5))                             # Numero de episodios de prueba
    max_steps_per_episode = int(config_data.get('max_steps_per_episode', fallback=2000))   # Longitud maxima de un episodio de prueba
    if agent_type == "DQN":
        use_dueling = config_data.getboolean('use_dueling', fallback=False)
        use_noisy = config_data.getboolean('use_noisy', fallback=False)
        use_distributional = config_data.getboolean('use_distributional', fallback=False)

    # Configuracion del dispositivo.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Registro del entorno Atari.
    gym.register_envs(ale_py)

    # Seleccion del modelo segun el tipo de agente.
    if agent_type == "DQN":
        print("Probando agente: DQN")
        env = make_dqn_env(env_name)
        input_shape = env.observation_space.shape
        n_actions = env.action_space.n
        policy_net = DQN(input_shape, n_actions, use_dueling=use_dueling, use_noisy=use_noisy, use_distributional=use_distributional).to(device)
    elif agent_type == "PPO":
        print("Probando agente: PPO")
        env = make_ppo_env(env_name)
        input_shape = env.observation_space.shape
        policy_net = PPO(env.action_space.n, input_shape).to(device)
    else:
        raise ValueError(f"Tipo de agente no reconocido: {agent_type}")
    
    # Grabacion de video opcional.
    if video_folder:
        os.makedirs(video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda ep_id: True,
            disable_logger=True
        )

    # Carga del modelo entrenado si existe.
    if os.path.exists(model_path):
        print(f"Cargando modelo desde '{model_path}'...")
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        policy_net.load_state_dict(new_state_dict)
    else:
        print(f"No se encontro el modelo '{model_path}', se ejecutara sin cargar pesos.")

    # Modo evaluacion (sin gradientes).
    policy_net.eval()

    # Realizar Test del agente segun el tipo.
    if agent_type == "DQN":
        testLoopDQN(episodes, max_steps_per_episode, env, policy_net, device, use_distributional=use_distributional)
    elif agent_type == "PPO":
        testLoopPPO(episodes, max_steps_per_episode, env, policy_net, device)
    else:
        raise ValueError(f"Tipo de agente no reconocido: {agent_type}")
    
    # Cierra el entorno al final.
    env.close()
    print("Test completado.")