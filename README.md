# HonorsThesis_MJR
This repository is the workspace for my honors thesis which focuses on creating a hybrid path planning algorithm for AUVs.

Before running any code, you must install all of the dependencies using pip
```console
pip install -r requirements.txt
```

## d-star-lite
Features a base D* Lite algorithm.

## baseline_ppo
Features the baseline PPO model along with its results

To train: 
```console
python main.py
```
Optional command line arguments:
- --grid_size = 10 -> the size of the grid 
- --total_timesteps = 100000 -> the total number of timesteps that will be trained for.
- --save_freq = 10 -> How often a checkpoint of the model wil be saved along with a graph rendering.
- --train = None -> Set it to the directory of the model checkpoint that you want to continue training with. If nothing is entered, it will train a new model.
## d_star_lite_ppo
Features the hyrbid algorithm along with its results.

To train: 
```console
python main.py
```
Optional parameters:
--grid_size = 10 -> the size of the grid 
--total_timesteps = 100000 -> the total number of timesteps that will be trained for.
--save_freq = 10 -> How often a checkpoint of the model wil be saved along with a graph rendering.
--train = None -> Set it to the directory of the model checkpoint that you want to continue training with
