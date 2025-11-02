import random
import torch
import gymnasium as gym
from agent import DQNAgent, DDQNAgent
from utils import convert_observation, wrap_env

# Funcion que decide la siguiente accion a tomar (greedy).
def next_action(observation, policy_net, device, env):
    """
    Decide la siguiente acción del agente usando greedy policy.

    Args:
        observation (torch.Tensor | np.array): Estado actual del entorno.
        policy_net (torch.nn.Module): Red neuronal que estima valores Q.
        device (torch.device): Dispositivo CPU/GPU.
        env (gym.Env): Entorno de Gym.

    Returns:
        torch.Tensor: Acción seleccionada.
    """
    with torch.no_grad():
        obs_t = observation.to(device) if torch.is_tensor(observation) else torch.tensor(observation, device=device, dtype=torch.float32)
        return policy_net(obs_t.unsqueeze(0)).max(1)[1].view(1, 1)


# Función principal de prueba del agente Atari.
def test(env, policy_net, num_episodes, video_folder=None, device="cpu", max_steps_per_episode=1000):
    """
    Ejecuta la prueba de un agente entrenado en un entorno Atari.

    Args:
        env (gym.Env): Entorno Atari.
        policy_net (torch.nn.Module): Red neuronal entrenada.
        num_episodes (int): Número de episodios a ejecutar.
        video_folder (str, optional): Carpeta para guardar videos. Si None, no graba.
        device (torch.device): Dispositivo para realizar inferencia.
        max_steps_per_episode (int): Máximo de pasos por episodio para evitar loops infinitos.
    """
    # Grabación de video solo si se especifica carpeta
    if video_folder:
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder)

    # Modo evaluación para desactivar gradientes
    policy_net.eval()

    for ep in range(num_episodes):
        total_reward = 0
        observation, _ = env.reset()
        observation = convert_observation(observation, device=device)

        for step in range(max_steps_per_episode):
            # Selección de acción
            action = next_action(observation, policy_net, device, env)

            # Tomar acción en el entorno
            next_observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Convertir siguiente estado a tensor
            observation = convert_observation(next_observation, device=device)

            # Terminar episodio si corresponde
            if terminated or truncated:
                break

        print(f"Episodio {ep+1}/{num_episodes} - Recompensa total: {total_reward}")

    # Cierra el entorno al final
    env.close()
