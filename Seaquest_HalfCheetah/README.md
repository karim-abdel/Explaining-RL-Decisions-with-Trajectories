## Overview
This folder is regarding experiments of 'Explaining RL Decisions with Trajectories': A Reproducibility Study. 
We focus on reproducing experiments for the **Seaquest** and **HalfCheetah** Environments.


## This part will focus on the requirements for Seaquest:

- Due to the versions of the imports a 3.8 python version is required.
  -- py -3.8 -m venv factvenv
- Clone the d4rl atari repository in the main folder:
  -- pip install git+https://github.com/takuseno/d4rl-atari
- Install gym with this atari version:
  -- pip install "gym[atari,accept-rom-license]"
- Install the Atari Dataset and put it in the data folder:
  -- http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html
  -- Add it to the ROMS: python -m atari_py.import_roms "data\Atari-2600-VCS-ROM-Collection\HC ROMS\BY ALPHABET\S-Z"
- Clone the pre trained decision transformer in the main folder:
  -- git clone https://huggingface.co/edbeeching/decision_transformer_atari
  -- Note: Do not forget to create an init.py file in the folder such that load_model.py can import the functions.

## This part will focus on the requirements for HalfCheetah:

- Install required dataset and follow their requirements:
  -- pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
  Download the package dependency mujoco:
  --https://github.com/google-deepmind/mujoco

For windows follow this tutorial: https://medium.com/@sayanmndl21/install-openai-gym-with-box2d-and-mujoco-in-windows-10-e25ee9b5c1d5
Some related github issues:
https://stackoverflow.com/questions/69307354/how-to-install-mujoco-py-on-windows
https://github.com/openai/mujoco-py/issues/253
https://github.com/openai/mujoco-py/issues/773
https://github.com/openai/mujoco-py/issues/298

Then Download the transformer:
pip install git+https://github.com/JannerM/doodad.git@janner

## Finally:

- Install all the requirements using
  ```
  pip install -r requirements.txt
  pip install codecarbon
  ```

## Trouble shooting:

- Try to pip install numpy seperatly before the requirements.txt file
- Try to pip install Cython seperatly before the requirements.txt file
