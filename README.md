## Learning Representative Vessel Trajectories Using Reinforcement and Imitation Learning

### About
This repository includes all implementations and experiment setups for the master thesis of Jan
Löwenstrom. One part is the implementation of at least one popular RL algorithm (DDPG) from scratch
and its usage on classic RL problems (e.g. MauntainCarContinous). Secondly, multiple environments got designed and can be found in the [deeprl/envs](/deeprl/envs) folder (e.g. synthetic curve in multiple setups or an AIS environment fed with real AIS data).
Apart from that, this repo also settles into using Imitation Learning to mimic human captain behaviour, learn and generalize corresponding trajectories.

### Structure
- [deeprl/](/deeprl/)
    - [agents/](/deeprl/agents) - own implementations
    - [common/](/deeprl/common): - utility functions mainly used by own implementations
    - [envs/](/deeprl/envs): custom environments
    - [scripts/](/deeprl/scripts): preprocessing scripts and main learning routines for RL and IL approaches
- [data/](/data/): raw AIS data and extracted expert trajectories (pull from dvc remote)
- [experiments/](/experiments/): metrics recorded from training (sophisticated algorithms)
- [runs/](/runs/): output and metrics from training by own implementations


### Major dependencies
- `PyTorch` as machine learning framework
- `Stable-Baselines3` as primary library regarding all RL algorithms (built on top of PyTorc)
- `imitation` as library for Imitation Learning algorithms (built on top of StableBaselines3) 
- `MovingPandas` as library for preprocessing AIS data (filter and extract vessel trajectories)

### Installation
It is very recommended to use a virtual environment in which all dependencies can be loaded. The current workflow looks like this:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

#### Installing PROJ version 8 (requirement for MovingPandas)
with conda:
```bash
conda install -c conda-forge proj
```
```bash
 mkdir ~/GPS
 cd ~/GPS
 wget https://download.osgeo.org/proj/proj-8.0.0.tar.gz
 tar xzvf proj-8.0.0.tar.gz

 cd proj-8.0.0

 mkdir build
 cd build

 cmake .. -DCMAKE_INSTALL_PREFIX=/usr
 cmake --build .
 sudo cmake --build . --target install
 ```

#### Install GEOS (needs ubuntu 20.04+)
 ```bash
 sudo apt update
 sudo apt install libgeos++-dev libgeos-3.8.0 libgeos-c1v5 libgeos-dev libgeos-doc
 ```

Alternatevly install CMake and GEOS from source.


### Setup DVC remote and pull data
The remote storage is located on the misa server:
```bash
\\misa.intra.dlr.de\MI\AIS\imitation_learning_dvc
```
Define a custom dvc remote (mounted NAS):
```bash
dvc remote add nas /path/to/mnt/nas
```
Pull all data or pull specific data (e.g. only single months of AIS data).
```bash
dvc pull -r remote
dvc pull -r rmote data/raw_data/2020_01
```
It is also possible to just pull the expert trajectories and preprocessed AIS data to just start
training:
```bash
dvc pull -r remote data/processed/*
dvc pull -r remote data/expert_trajectory/*
```

### Usage
Modify `ais_imitation.sh`, e.g. changing algorithm from `bc` to `gail`, amount of neurons or training steps. Comment out commnds to sample expert trajectories (usually done just once)
or training, enable or disable rendering while testing.
Then, just run the script from the root folder:

```bash
./ais_imitation.sh
```
