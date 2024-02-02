This folder is regarding experiments of 'Explaining RL Decisions with Trajectories': A Reproducibility Study. 

## Files description:

The files are concerning the following parts of our study:

1. `table1_and_DTDAclaim` contains results reproducing table 1 of our paper. We also brefly mention how to obtain visually the DTDA claim.
2. `clustering_and_humanstudy_expts` is taking care of claim Clustering High-Level Behaviours. It also contains additional experiments regarding DBSCAN clustering and its comparison with XMeans. Additionally it also contains plots regarding the conducted Human Study.
3. `different_encoder_and_hyperparams_expts` is made of additional experiments to be further investigated on both chaning the encoding trajectory technique as well as tweaking some hyperparameters.
4. `Additional_Experiments_DTDA_RTISV` folder contains files regarding additional experiments for claim Distant Trajectories infulencing Decision of the Agent (DTDA) and Removing Trajectories induces a lower Initial State Value (RTISV). 


## Instructions for usage:

1. Before running the code-base, install the dependencies using:
    ```
        conda create -n xrl python=3.8 -y
        conda activate xrl
        pip install -r requirements.txt
        python -m ipykernel install --user --name xrl
    ```

2. Launch any of the `.ipynb` file using a jupyter server. Activate the `xrl` kernel and run the file to generate the results from the paper.

__Acknowledgements__: We use Dynamic Programming implementation from [andrecianflone/dynaq/](https://github.com/andrecianflone/dynaq/) and we are thankful to the authors for making it publicly available.


   
