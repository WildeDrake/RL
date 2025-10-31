import torch
import numpy as np
import gymnasium as gym
import configparser


# Funcion para convertir observaciones a tensores de PyTorch
def convert_observation(observation, device=None):
    # Convierte la observación a un tensor de PyTorch
    obs = torch.from_numpy(np.array(observation)).float()
    # Mueve el tensor al dispositivo especificado (CPU o GPU)
    return obs.to(device) if device else obs


# Funcion para envolver un entorno de Gym con acciones Noop al inicio
class NoopStart(gym.Wrapper):
    # Constructor de la clase NoopStart
    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0

    # Sobrescribe el método reset para incluir acciones Noop al inicio del episodio
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


# Funcion para envolver un entorno de Gym con varias transformaciones comunes
def wrap_env(env: gym.Env):
    # Loggeo de estadísticas del episodio
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # Convertir observaciones a escala de grises
    env = gym.wrappers.GrayScaleObservation(env)
    # Reescala las observaciones a una resolución más pequeña
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    # Normaliza las observaciones para tener media 0 y desviación estándar 1
    env = gym.wrappers.NormalizeObservation(env)
    # Pila de múltiples fotogramas consecutivos para proporcionar información temporal al agente
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env


# Cargar parámetros desde un archivo de configuración INI
def load_parameters_from_config(config_file, mode="Training"):
    config = configparser.ConfigParser()
    config.read(config_file)
    if mode not in config:
        print(f"Error: '{mode}' section not found in the configuration file '{config_file}'.")
        exit(1)
    return config[mode]