import numpy as np
import gymnasium as gym
import configparser



# Funcion para envolver un entorno de Gym con acciones Noop al inicio.
class NoopStart(gym.Wrapper):
    # Constructor de la clase NoopStart.
    def __init__(self, env: gym.Env, noop_max: int = 30, noop_action: int = 0):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = noop_action
    # Sobrescribe el metodo reset para incluir acciones Noop al inicio del episodio.
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info



# Funcion para envolver un entorno con varias transformaciones para DQN.
def make_dqn_env(env_name: str, render_mode="rgb_array") -> gym.Env: 
    # Crear el entorno base.
    env = gym.make(env_name, render_mode=render_mode)
    # Loggeo de estadísticas del episodio.
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # Acciones Noop al inicio del episodio.
    env = NoopStart(env, noop_max=30)
    # Convertir observaciones a escala de grises.
    env = gym.wrappers.GrayscaleObservation(env)
    # Reescala las observaciones a una resolucion mas pequeña.
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    # Pila de multiples fotogramas consecutivos.
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env



def make_ppo_env(env_name: str, seed: int, render_mode="rgb_array") -> gym.Env: 
    # Crear entorno base.
    env = gym.make(env_name, render_mode=render_mode)
    # Loggeo de estadísticas del episodio.
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # Acciones Noop al inicio del episodio.
    env = NoopStart(env, noop_max=30)
    # Convertir observaciones a escala de grises.
    env = gym.wrappers.GrayscaleObservation(env)
    # Reescala las observaciones a una resolucion mas pequeña.
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    # Pila de multiples fotogramas consecutivos.
    env = gym.wrappers.FrameStackObservation(env, 4)
    # Transformar recompensas pequeñas y estables [-1, 1].
    env = gym.wrappers.TransformReward(env, lambda r: np.sign(r))
    # PPO necesita seeds diferentes en cada entorno.
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env



# Cargar parametros desde un archivo de configuracion INI.
def load_parameters_from_config(config_file, mode="Training"):
    config = configparser.ConfigParser()
    config.read(config_file)
    if mode not in config:
        print(f"Error: '{mode}' section not found in the configuration file '{config_file}'.")
        exit(1)
    return config[mode]