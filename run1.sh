#!/bin/bash

python GNN.py --online_split=False &
python GNN.py --file_option=_miss_graph --online_split=False &