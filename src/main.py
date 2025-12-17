import argparse

from utils import load_parameters_from_config
from test import test
from train import train



def main():
    # Parsear argumentos de línea de comandos.
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['training', 'testing'], help='Training or Testing')
    parser.add_argument('--config', '-c', help='INI configuration file', required=True)
    args = parser.parse_args()
    # Cargar parámetros desde el archivo de configuracion.
    config_data = load_parameters_from_config(args.config, args.mode)
    # Ejecutar el modo correspondiente.
    if args.mode == 'training':
        train(config_data)
    else:
        test(config_data)



if __name__ == '__main__':
    main()
