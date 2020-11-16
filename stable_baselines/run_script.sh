#!/bin/bash

python run_exp.py --gpu 0 --env Walker2d-v3 --algo MDAL --expert-path expert_walker_with_actions --num-timesteps 3000000 --sgd-steps 1 --mdpo-update-steps 100 --seed 0
python run_exp.py --gpu 0 --env Walker2d-v3 --algo MDAL --expert-path expert_walker_with_actions --num-timesteps 3000000 --sgd-steps 1 --mdpo-update-steps 100 --seed 1
python run_exp.py --gpu 0 --env Walker2d-v3 --algo MDAL --expert-path expert_walker_with_actions --num-timesteps 3000000 --sgd-steps 1 --mdpo-update-steps 100 --seed 2
python run_exp.py --gpu 0 --env Walker2d-v3 --algo MDAL --expert-path expert_walker_with_actions --num-timesteps 3000000 --sgd-steps 1 --mdpo-update-steps 100 --seed 3
python run_exp.py --gpu 0 --env Walker2d-v3 --algo MDAL --expert-path expert_walker_with_actions --num-timesteps 3000000 --sgd-steps 1 --mdpo-update-steps 100 --seed 4