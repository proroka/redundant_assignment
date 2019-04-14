#!/bin/bash

python3 launch_experiments_nodes.py --output_results=node_results.bin
python3 launch_experiments.py --output_results=results.bin
sudo poweroff
