import numpy as np
import gymnasium as gym
import configparser



# Clase para envolver un entorno de Gym y aplicar Max pooling y Skipframes.
class MaxAndSkipEnv(gym.Wrapper):

    # Constructor de la clase MaxAndSkipEnv.
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    # Sobrescribe el metodo step para aplicar Skipframes y Max pooling.
    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        info = {}
        # Buffer para guardar los ultimos 2 frames (para el Max pooling).
        obs_buffer = np.zeros((2, *self.env.observation_space.shape), dtype=np.uint8)
        # Ejecuta la accion durante skip frames.
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            # Guarda los ultimos 2 frames en el buffer.
            if i == self._skip - 2: obs_buffer[0] = obs
            if i == self._skip - 1: obs_buffer[1] = obs
            # Acumula la recompensa total.
            total_reward += reward
            if done:
                break
        # Toma el pixel mas brillante de los ultimos 2 frames.
        max_frame = obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info
    


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
    env = gym.make(env_name, render_mode=render_mode, frameskip=1, repeat_action_probability=0.0)
    # Loggeo de estadisticas del episodio.
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # Acciones Noop al inicio del episodio.
    env = NoopStart(env, noop_max=30)
    # Max pooling y Skipframes.
    env = MaxAndSkipEnv(env, skip=4)
    # Convertir observaciones a escala de grises.
    env = gym.wrappers.GrayscaleObservation(env)
    # Reescala las observaciones a una resolucion mas pequeña.
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    # Pila de multiples fotogramas consecutivos.
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env



def make_ppo_env(env_name: str, seed: int, render_mode="rgb_array") -> gym.Env: 
    # Crear entorno base.
    env = gym.make(env_name, render_mode=render_mode, frameskip=1, repeat_action_probability=0.0)
    # Loggeo de estadisticas del episodio.
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # Acciones Noop al inicio del episodio.
    env = NoopStart(env, noop_max=30)
    # Max pooling y Skipframes.
    env = MaxAndSkipEnv(env, skip=4)
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



# Estructura de datos SumTree para Prioritized Experience Replay.
class SumTree:
    # Constructor.
    def __init__(self, capacity):
        # La capacidad debe ser la potencia de 2 mas cercana para que el arbol sea balanceado.
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1) # Array para guardar las sumas de prioridades.
        self.data = np.zeros(capacity, dtype=object) # Array para guardar punteros.
        self.write = 0 # Puntero de escritura circular.
        self.n_entries = 0

    # Propagar cambios hacia la raiz del arbol.
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # Buscar una muestra aleatoria.
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        # Si llegamos a una hoja, devolvemos el indice.
        if left >= len(self.tree):
            return idx
        # Recorremos el arbol.
        if s <= self.tree[left]:    # Ir al hijo izquierdo.
            return self._retrieve(left, s)
        else:                       # Ir al hijo derecho.
            return self._retrieve(right, s - self.tree[left])
        
    # Obtener la suma total de prioridades.
    def total(self):
        return self.tree[0]

    # Añadir una nueva muestra con prioridad p.
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        # Circular buffer.
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    # Actualizar la prioridad de una muestra existente.
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # Obtener una muestra aleatoria basada en la prioridad.
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])