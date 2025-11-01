import gymnasium as gym
import ale_py

# Registrar todos los entornos de ALE
gym.register_envs(ale_py)

# Crear entorno de Frogger
env = gym.make("ALE/Frogger-v5", render_mode="rgb_array")
obs, info = env.reset()

# Hacer un paso random
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print("Reward:", reward)

env.close()
