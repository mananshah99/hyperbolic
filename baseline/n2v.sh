#!/usr/bin/env python
set -x

# node2vec

## email
python -m openne \
    --method node2vec \
    --label-file ../data/email/email-Eu-core-department-labels.txt \
    --input ../data/email/email-eu-core.txt \
    --graph-format edgelist \
    --output embeddings/node2vec_email.txt \
    --q 0.25 \
    --p 0.25

## chg-miner
python -m openne \
    --method node2vec \
    --label-file ../data/chg-miner/chg-miner-labels.txt \
    --input ../data/chg-miner/chg-miner-graph.txt \
    --graph-format edgelist \
    --output embeddings/node2vec_chg_miner.txt \
    --q 0.25 \
    --p 0.25
