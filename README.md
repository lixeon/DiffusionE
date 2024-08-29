# DiffusionE
Official code for the KDD 2024 paper "DiffusionE: Reasoning on Knowledge Graphs via Diffusion-based Graph Neural Networks"

And the another official repo version goes to https://github.com/Worker-AGI/DiffusionE/

## Dependencies

- torch == 1.12.1
- torch_scatter == 2.0.9
- numpy == 1.21.6
- scipy == 1.10.1

## Reproduction

### Transductive settings (in `\transductive`)

#### Reproduction with training scripts

##### Family dataset

```
python3 train.py --data_path ./data/family/ --train --topk 100 --layers 8 --fact_ratio 0.90 --gpu 0
```

##### UMLS dataset
```
python3 train.py --data_path ./data/umls/ --train --topk 100 --layers 5 --fact_ratio 0.90 --gpu 0
```

##### WN18RR dataset
```
python3 train.py --data_path ./data/WN18RR/ --train --topk 1000 --layers 8 --fact_ratio 0.96 --gpu 0
```

##### FB15k-237 dataset
```
python3 train.py --data_path ./data/fb15k-237/ --train --topk 2000 --layers 7 --fact_ratio 0.99 --remove_1hop_edges --gpu 0
```

##### NELL995 dataset
```
python3 train.py --data_path ./data/nell/ --train --topk 2000 --layers 6 --fact_ratio 0.95 --gpu 0
```

##### YAGO3-10 dataset
```
python3 train.py --data_path ./data/YAGO/ --train --topk 1000 --layers 8 --fact_ratio 0.995 --gpu 0
```

### Inductive settings (in `\inductive`)

#### Reproduction with training scripts

The full training scripts can be found in [inductive/reproduce.sh](https://github.com/LARS-research/DiffusionE/blob/main/inductive/reproduce.sh).

For example, training on `WN18RR v1` dataset:

```
python3 train.py --data_path ./data/WN18RR_v2 --gpu 1
python3 train.py --data_path ./data/fb237_v1 --gpu 1
python3 train.py --data_path ./data/nell_v1 --gpu 4
```



## Citation

If you find DiffusionE useful in your research or applications, please kindly cite:

```
@inproceedings{10.1145/3637528.3671997,
author = {Cao, Zongsheng and Li, Jing and Wang, Zigan and Li, Jinliang},
title = {DiffusionE: Reasoning on Knowledge Graphs via Diffusion-based Graph Neural Networks},
year = {2024},
isbn = {9798400704901},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3637528.3671997},
doi = {10.1145/3637528.3671997},
abstract = {Graph Neural Networks (GNNs) have demonstrated powerful capabilities in reasoning within Knowledge Graphs (KGs), gathering increasing attention. Our idea stems from the observation that the prior work typically employs hand-designed or sample-designed paradigms in the process of message propagation, engaging a set of adjacent entities at each step of propagation. As a result, such methods struggle with the increasing number of entities involved as propagation steps extend. Moreover, they neglect the message interactions between adjacent entities and propagation relations in KG reasoning, leading to semantic inconsistency during the message aggregation phase. To address these issues, we introduce a novel knowledge graph embedding method through a diffusion process, termed DiffusionE. Specifically, we reformulate the message propagation in knowledge reasoning as a diffusion process, regarding the message semantics as the diffusion signal. In this sense, guided by semantic information, messages can be transmitted between nodes effectively and adaptively. Furthermore, the theoretical analysis suggests our method can leverage an optimal diffusivity for message propagation in the semantic interactions of KGs. It shows that DiffusionE effectively leverages message interactions between entities and propagation relations, ensuring semantic consistency in KG reasoning. Comprehensive experiments reveal that our method attains state-of-the-art performance compared to prior work on several well-established benchmarks.},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {222–230},
numpages = {9},
keywords = {diffusion process, graph neural networks, knowledge graph},
location = {Barcelona, Spain},
series = {KDD '24}
}
```
