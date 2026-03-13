# FeatureCloud Template

This repository is a minimal linear regression app based on a starter template 
for building a [FeatureCloud app](https://featurecloud.ai/) that can be run in two ways:

- as a regular FeatureCloud app using FeatureClouds dockerized testembed, running on just one machine (no publishing needed)
- as a native local simulation (without Docker) just on one machine (no publishing needed)

## How to navigate the code

The federated learning logic is implemented in:
- `logic.py` → function `fl_algorithm(...)`
The client and aggregation server code is seperated into two classes:
- `Client`
- `Aggregator`
Which code is called is set in if blocks:
`if fed_learning_class_instance.is_coordinator`.

The template is already wired so that any execution via featurecloud or native execution
call this same function.

Relevant files:

- `logic.py`: All relevant federated learning logic should be in the `fl_algorithm(...)`! 
You can easily call helper classes such as a client or coordinator class from here and 
differentiate between client and coordinator via a simple `fed_learning_class_instance.is_coordinator` call.
- `helper/protocolfedlearningclass.py`: This protocol describes all the federated learning helper methods such as sending and receiving data you can run.
- `run_simulation.py`: Runs a federeated learning simulation. See the file to change whether featurecloud is used or the simulation is native.
- `requirements.txt`: Please make sure to add any dependencies you add to `logic.py`. 
Otherwise the dockerized execution of your app will fail, as the Dockerfile to dockerize your app uses this file!

## Dependencies

### 1) For native simulation
For native simulation of your federated learning app, you need to have all relevant python
packages of your algorithm installed. If you correctly added the dependencies to the `requirements.txt`, 
you can simply install them e.g. via `venv`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) For dockerized simulation
Just docker (docker engine or docker desktop)

## How to run simulations

The default simulation entry script is:

- `run_simulation.py`

It already contains both simulation calls:

- `run_simulation_featurecloud(...)`
- `run_simulation_native(...)`

Change the main function accordingly to what you wanna do, here either of these functions is called.

## Project structure (important files)

- `logic.py`: your federated learning algorithm implementation. 
- `states.py`: minimal FeatureCloud state (`initial` → `terminal`). You should never need to touch this.
- `run_simulation.py`: Small caller script for simulations.
- `helper/run_app_simulation.py`: native + FeatureCloud simulation helpers. You should never need to touch this.
- `main.py`: FeatureCloud app server entrypoint. You should never need to touch this.
