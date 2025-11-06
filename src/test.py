import random
import torch
import gymnasium as gym
from agent import DQNAgent
from utils import convert_observation, wrap_env, NoopStart

# Epsilon-greedy (greedy en test)
def next_action(observation, agent: DQNAgent, epsilon: float = 0.0):
    agent.steps_done += 1
    if random.random() > epsilon:
        with torch.no_grad():
            obs_t = observation.to(agent.device) if torch.is_tensor(observation) else torch.tensor(
                observation, device=agent.device, dtype=torch.float32
            )
            action = agent.policy_net(obs_t.unsqueeze(0)).max(1)[1].item()
    else:
        action = random.randrange(agent.n_actions)
    return action


def test(env_name: str, agent: DQNAgent, num_episodes: int, video_folder=None,
         device="cpu", max_steps_per_episode=1000):
    """
    Ejecuta la prueba de un agente entrenado y graba video fluido del entorno real.
    """

    # --- Crear entorno real ---
    env = gym.make(env_name, render_mode="rgb_array")
    env = NoopStart(env, noop_max=30)  # 🔹 aplica aleatoriedad de inicio
    
    # --- Grabar video ---
    if video_folder:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda ep_id: True,
            disable_logger=True
        )

    # --- Política ---
    agent.policy_net.eval()
    agent.policy_net.to(device)

    for ep in range(num_episodes):
        total_reward = 0
        obs, _ = env.reset()

        for step in range(max_steps_per_episode):
            # Convierte la observación (sin wrappers visuales)
            obs_proc = convert_observation(obs, device=device)

            # Acción greedy
            action = next_action(obs_proc, agent, epsilon=0.0)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        print(f"Episodio {ep+1}/{num_episodes} - Recompensa total: {total_reward}")

    env.close()
