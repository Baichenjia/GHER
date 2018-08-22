## Guided goal generation for multi-goal reinforcement learning
The details of this paper shows [here](https://sites.google.com/view/gher-algorithm).

### Prerequisites
GHER requires python3.6, tensorflow-gpu 1.8.0, [mujoco](http://www.mujoco.org/)-engine with license,  [mujoco-py](https://github.com/openai/mujoco-py), openAI [baselines](https://github.com/openai/baselines).

### Installation

```
git clone https://github.com/Baichenjia/GHER.git
cd GHER
pip install -e .
```

### The conditional RNN model
Set the `envname` parameter in `GHER/gmm_model/CONDIG.py` file. This file describes the hyper-parameters in proposed conditional RNN model. Model architecture defined in file `GHER/gmm_model/gmm_model.py`. We have pre-trained the RNN model in all tasks and save the Tensorflow model parameters in folder `GHER/gmm_model/gmm_save/`. When G-HER start training, the pre-trained RNN model will be loaded according to `envname` setup.

### Train a G-HER agent
Training an agent is simple, `cd GHER/experiment/` and then
```
python train.py
```
This will train a GHER+DDPG agent on the `FetchPush` task. You can choose the other tasks by setting the choice in `train.py` file. Then set the same environment name in `GHER/gmm_model/CONDIG.py` to load the specific pre-trained RNN model.

### Results
We reproduce 3 baselines includes: [HER](https://arxiv.org/abs/1707.01495), typical [UVFAs](http://proceedings.mlr.press/v37/schaul15.pdf) in multi-goal setup. Both of them are under the sparse rewards setup. We also reproduce a vanilla DDPG with a shaped reward for comparison. We trained all the baselines with 1 cpu core and 1 NVIDIA 1080Ti GPU in a single machine. The policy file and learning curve are saved in folder `GHER/experiment/result/`. You are free to use these pre-trained baseline results. We illustrate the learning curve of baselines and proposed G-HER result shows in `cd GHER/experiment/result/pic`. You can plot the learning curve with 
```
python plot.py
```

### Play with G-HER
We save the pre-trained GHER agent in `GHER/experiment/result/Gher-result/`. To visualize the performance of the pre-trained best policy, `cd GHER/experiment/result` and
```
python play.py
```
This will use the pre-trained `FetchPickAndPlace` policy to rollout for 20 episodes. You are free to use other policies in other tasks by modifying the `policy_file` choice. The recorded video was shown [here](https://sites.google.com/view/gher-algorithm).
