import random

import torch
import gymnasium as gym
from agent import AtariAgent
from utils import convert_observation, wrap_env


# Funcion que decide la siguiente accion a tomar.
def next_action(observation, policy_net, device):
    if random.random() > 0.00:
        with torch.no_grad():
            return (
                policy_net(observation.to(device))
                .max(1)[1]
                .view(1, 1)
            )
    else:
        return torch.tensor(
            [[random.randrange(4)]],
            device=device,
            dtype=torch.long,
        )


def test(env, policy_net, num_episodes, video_folder, device):
    env = gym.wrappers.RecordVideo(env, video_folder=video_folder)
    for _ in range(num_episodes):
        # Reinicia el entorno para un nuevo episodio y convierte la observación inicial.
        total = 0
        observation, _ = env.reset()
        observation = convert_observation(observation)

        while True:
            # Elige la siguiente acción utilizando la política epsilon-greedy del agente.
            action = next_action(observation, policy_net, device)

            # Tomar la acción elegida en el entorno y recibir la siguiente observación y recompensa.
            next_observation, reward, terminated, truncated, _ = env.step(
                action
            )
            total += reward

            next_observation = convert_observation(next_observation)

            # Actualiza la observación actual.
            observation = next_observation

            done = truncated or terminated
            if done:
                print(total)
                break
    env.close()
