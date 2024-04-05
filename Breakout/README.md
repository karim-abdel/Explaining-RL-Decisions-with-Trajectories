## Overview
This folder is regarding experiments of 'Explaining RL Decisions with Trajectories': A Reproducibility Study. 
We focus on reproducing experiments for the **Seaquest** and **HalfCheetah** Environments.


## This part will focus on the requirements for Break out:

- Due to the versions of the imports a 3.9 python version is required.
- Install Eorl:
 -- pip install git+https://github.com/indrasweb/expert-offline-rl.git
- Clone the d4rl atari repository in the main folder:
  -- pip install git+https://github.com/takuseno/d4rl-atari
- Install gym with this atari version:
  -- pip install "gym[atari,accept-rom-license]"
- Clone the pre trained decision transformer in the main folder:
  -- git clone https://huggingface.co/edbeeching/decision_transformer_atari
  -- Note: Do not forget to create an init.py file in the folder such that load_model.py can import the functions.

## Finally:

- Install all the requirements using
  ```
  pip install -r requirements.txt
  pip install codecarbon
  ```

## Trouble shooting:

- Try to pip install numpy seperatly before the requirements.txt file
- Try to pip install Cython seperatly before the requirements.txt file
