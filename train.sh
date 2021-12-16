#!/usr/bin/env bash
#
# eval.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#


python src/train.py \
    --attn_model general \
    --batch_size 64 \
    --embedding_size 1000 \
    --hidden_size 1000 \
    --n_layers 4 \
    --dropout 0.1 \
    --teacher_forcing_ratio 0.8 \
    --clip 5.0 \
    --lr 0.0005 \
    --n_epochs 10000 \
    --plot_every 20 \
    --print_every 10 \
    --language spa \
    --device cuda \
    --seed 19 \

/
