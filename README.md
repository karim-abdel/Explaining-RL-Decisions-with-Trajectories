This is the code-base for 'Explaining RL Decisions with Trajectories': A Reproducibility Study.

[Other Potential Lik (to put the link)]( - ) |  [Full Paper (to put the link)]( - )

---
Here we provide the code for the gridworld experiments, which can be found in the Grid-World folder. 
In case of additional queries, feel free to reach out to one of the authors.
---

## Instructions for usage:

1. Before running the code-base, install the dependencies using:
    ```
        conda create -n xrl python=3.8 -y
        conda activate xrl
        pip install -r requirements.txt
        python -m ipykernel install --user --name xrl
    ```

2. Launch `gridworld_expts.ipynb` using a jupyter server. Activate the `xrl` kernel and run the file to generate the results from the paper.

__Acknowledgements__: We use Dynamic Programming implementation from [andrecianflone/dynaq/](https://github.com/andrecianflone/dynaq/) and we are thankful to the authors for making it publicly available.

### Citation

If you use this code for your research, please cite our paper:

```
@misc{deshmukh2023explaining,
      title={Explaining RL Decisions with Trajectories}, 
      author={Shripad Vilasrao Deshmukh and Arpan Dasgupta and Balaji Krishnamurthy and Nan Jiang and Chirag Agarwal and Georgios Theocharous and Jayakumar Subramanian},
      year={2023},
      eprint={2305.04073},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
