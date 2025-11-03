import argparse
import os
import torch
import gymnasium as gym
import ale_py
from utils import wrap_env, load_parameters_from_config, NoopStart
from test import test
from train import train
from model import DQN
from agent import DQNAgent, DDQNAgent


# Ejecutar el modo de prueba del agente Atari
def testing(config_data, agent_type, max_steps_per_episode=1000):
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
    # Grabación de video opcional: solo si se especifica carpeta
    if video_folder:
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder)
    # Selección del modelo según el tipo de agente
    if agent_type == "DDQN":
        print("Probando agente: Double DQN")
        policy_net = DQN(env.action_space.n)
    elif agent_type == "DQN":
        print("Probando agente: DQN")
        policy_net = DQN(env.action_space.n)
    else:
        raise ValueError(f"Tipo de agente no reconocido: {agent_type}")
    # Carga del modelo entrenado si existe
    if os.path.exists(model_path):
        print(f"Cargando modelo desde '{model_path}'...")
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        policy_net.to(device)
    else:
        print(f"No se encontró el modelo '{model_path}', se ejecutará sin cargar pesos.")
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
            next_observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            # Actualizar observación
            observation = next_observation
            done = terminated or truncated
            if done:
                break
        print(f"Episodio {ep+1}/{episodes} - Recompensa total: {total_reward}")
    # Cierra el entorno al final
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
    env = gym.make(env_name)
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
    train(env, agent, episodes, batch_size, max_episode_length)
    print("Entrenamiento completado.")
    # Guardar el modelo entrenado
    print(f"Guardando modelo entrenado en '{model_path}'...")
    torch.save(agent.policy_net.state_dict(), model_path)
    # Liberar memoria del entorno al terminar (buena práctica)
    env.close()




def main():
    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['Training', 'Testing'], help='Training or Testing')
    parser.add_argument('--config', '-c', help='INI configuration file', required=True)
    parser.add_argument('--agent', '-a', help='Tipo de agente (DQN, DDQN, PPO, etc.)', default='DQN')
    args = parser.parse_args()
    # Cargar parámetros desde el archivo de configuración
    config_path = args.config
    config_data = load_parameters_from_config(config_path, args.mode)
    # Validar parametros minimos segun el modo
    required_keys = (
        ["env", "actions", "learning_rate", "episodes"]
        if args.mode == "Training"
        else ["env", "model_path", "episodes", "video_folder"]
    )
    for key in required_keys:
        if key not in config_data:
            raise ValueError(f"Missing required parameter '{key}' for mode '{args.mode}' in config file.")
    # Ejecutar el modo correspondiente
    if args.mode == 'Training':
        training(config_data, args.agent.upper())
    else:
        testing(config_data, args.agent.upper())


if __name__ == '__main__':
    main()
