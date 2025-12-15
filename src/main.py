import argparse

from utils import load_parameters_from_config
from test import test
from train import train



def main():
    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['Training', 'Testing'], help='Training or Testing')
    parser.add_argument('--config', '-c', help='INI configuration file', required=True)
    parser.add_argument('--agent', '-a', help='Tipo de agente (DQN, DDQN, PPO, etc.)', default='DQN')
    args = parser.parse_args()
    # Cargar parámetros desde el archivo de configuración
    config_data = load_parameters_from_config(args.config, args.mode)
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
        train(config_data, args.agent.upper())
    else:
        test(config_data, args.agent.upper())


if __name__ == '__main__':
    main()
