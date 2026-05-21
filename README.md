# Tensor Decomposition with Dual-Perspective for Temporal Knowledge Graph Completion

## Installation
Create a conda environment with pytorch and scikit-learn :
```
conda create --name complex python=3.7
source activate complex
conda install --file requirements.txt -c pytorch
```


## Datasets

python tkbc/process_icews.py
python tkbc/process_yago.py
python tkbc/process_gdelt.py


This will create the files required to compute the filtered metrics.

## Reproducing results

Run the following commands in tkbc folder to reproduce the results

```
python tkbc/learner.py --dataset ICEWS14 --model testmodel --rank 1594 --emb_reg 1e-1 --time_reg 1e-4

python tkbc/learner.py --dataset yago15k --model testmodel --rank 364 --emb_reg 1e-1 --time_reg 1e-4

python tkbc/learner.py --dataset gdelt --model testmodel --rank 1256 --learning_rate 3e-1 --emb_reg 1e-5 --time_reg 1e-2

```

## Acknowledgement
We refer to the code of [TPComplEx](https://github.com/Jinfa/TPComplEx). Thanks for their contributions.

