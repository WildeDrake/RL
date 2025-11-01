import pkg_resources

installed_packages = pkg_resources.working_set
for package in sorted(installed_packages, key=lambda x: x.key):
    print(f"{package.key}=={package.version}")

import ale_py
import gymnasium as gym

from ale_py.roms import get_game_path

print(get_game_path('frogger'))  # debería imprimir la ruta al .bin
