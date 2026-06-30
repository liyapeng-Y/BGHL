
Official PyTorch implementation of **“Navigating Hierarchy: Hyperbolic Learning on Brain Graphs for Disorder Diagnosis”**.

HLBG is designed for brain disorder diagnosis from functional brain networks. It models the natural hierarchy of brain graphs across **ROIs**, **functional communities**, and **whole-brain** levels by combining hyperbolic representation learning.

---

## Overview

Functional brain networks exhibit hierarchical organization: local ROI-level interactions support regional processing, community-level structures capture coordinated functional modules, and whole-brain patterns reflect global integration. Existing brain graph learning methods often underuse the relationships between ROIs, communities, and the full brain network.

To address this limitation, **Hyperbolic Learning on Brain Graphs (HLBG)** introduces a hierarchy-aware framework that:

- projects ROI-, community-, and whole-brain-level representations into Lorentzian hyperbolic space;
- imposes multi-level geometric entailment constraints to preserve brain-network hierarchy;
- uses a **Graph-aware Mamba (GaMamba)** module to capture long-range dependencies while retaining graph topology;
- learns discriminative representations for disorder diagnosis and biomarker discovery.

Experiments in the paper are conducted on **ABIDE-I** for Autism Spectrum Disorder diagnosis and **REST-MDD** for Major Depressive Disorder diagnosis.

---

## Framework

The framework figure is available here:

[View framework diagram](./Fig/framework.pdf)

---

## Repository Structure

```text
.
├── README.md
├── HLBG_README.md
├── requirements.txt
├── train.py
├── Fig/
│   └── framework.pdf
├── datasets/
│   ├── ABIDEDataset.py
│   ├── read_data.py
│   ├── comm_utils.py
│   ├── utils.py
│   ├── craddock200.rda
│   ├── node_clus_map.pickle
│   ├── Yeo7_coordinates.npy
│   └── DICE_CPAC200_&_Yeo-7-liberal_res-1x1x1.pkl
└── model/
    ├── HLBG.py
    ├── Gamamba.py
    ├── SubgraphAwareAttention.py
    ├── gnn.py
    ├── lorentz.py
    ├── position_encoding.py
    ├── utils.py
    └── ptdec/
```

---

## Requirements

The main dependencies include:

- Python 3.9+
- PyTorch
- PyTorch Geometric
- CUDA-enabled PyTorch environment, recommended for Mamba-related modules
- `mamba-ssm`
- `causal-conv1d`
- `geoopt`
- `deepdish`
- `pyreadr`
- `scikit-learn`
- `numpy`, `scipy`, `pandas`

Install the provided environment:

```bash
pip install -r requirements.txt
```


---

## Dataset Preparation

### Supported datasets

- **ABIDE-I**: Autism Spectrum Disorder diagnosis  
  Dataset website: `http://fcon_1000.projects.nitrc.org/indi/abide/`

- **REST-MDD**: Major Depressive Disorder diagnosis  
  Dataset website: `http://rfmri.org/REST-meta-MDD`


## Training

Run HLBG on ABIDE- data:

```bash
python train.py \
  --root_dir ./data/ABIDE_pcp/cpac/ \
  --epochs 70 \
  --batch-size 64 \
  --dropout 0.2
```
## Citation

<!-- If you find this repository useful, please cite our paper once the bibliographic information is available:

 ```bibtex
@article{hlbg,
  title   = {Navigating Hierarchy: Hyperbolic Learning on Brain Graphs for Disorder Diagnosis},
  author  = {To be added},
  journal = {To be added},
  year    = {To be added}
}
```  -->

This work also builds on community-aware brain graph learning. Please consider citing the following related work:

```bibtex
@inproceedings{ijcai2025p467,
  title     = {Community-Aware Graph Transformer for Brain Disorder Identification},
  author    = {Pei, Shengbing and Ma, Jiajun and Lv, Zhao and Zhang, Chao and Guan, Jihong},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {James Kwok},
  pages     = {4191--4199},
  year      = {2025},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2025/467},
  url       = {https://doi.org/10.24963/ijcai.2025/467}
}
```


