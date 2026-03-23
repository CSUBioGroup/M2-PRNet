# 🥥 M2-PRNet

Official PyTorch implementation of **M2-PRNet: Multi-Scale and Multi-Modal Learning for Protein–RNA Binding Affinity Prediction**

<img src="./assets/model.jpg" alt="Overview of M2-PRNet" width="800">

M2-PRNet is a multi-scale and multi-modal framework for protein–RNA binding affinity prediction, combining atomistic interactions, residue-level organization, and global structural representations. It achieves state-of-the-art performance and strong generalization on dynamic and cold-start scenarios.

---

## 🛠️ Installation

```bash
conda env create -f environment.yml
conda activate m2prnet_env
```

## 📖 Datasets and Model Weights

We release the datasets used in this work via 🤗 Hugging Face:  
👉 [xxxxx](https://huggingface.co/datasets/xxxx)

- **PRA310** (full dataset, including PRA201 subset): `PRA310.csv`  
- **MD150** (MD-derived dynamic dataset): `MD150.csv`

Dataset statistics are summarized below. PRA denotes protein–RNA binding affinity datasets:

| Dataset | Type | Size |
| :---: | :---: | :---: |
| PRA310 | PRA | 310 |
| PRA201 | PRA (pair-only) | 201 |
| MD150 | MD-derived dynamic complexes | 150 |

## 📦 Model Checkpoints

Five-fold model checkpoints are available at 🤗 Hugging Face:  
👉 [/NikoWz/M2-PRNet](https://huggingface.co/NikoWz/M2-PRNet)

- Place model weights in `./workdir/`  
- Place `.pkl` files in the project root (`./`)
- Place `data` Folder in the project root (`./`)


## 🚀 Training and Evaluation

**Note1:** It is normal that the first epoch for training on a new dataset is relatively slow, because we need to conduct the caching procedure.

### Run 5-fold inference on PRA310
```
python Train_PNA.py --sampler --share_weights --image_network --contrastive --data_set PNA_keys.csv --onlytest 
```


### Run finetune on PRA310
```
python Train_PNA.py --sampler --share_weights --image_network --contrastive --data_set PNA_keys.csv 
```


### Run finetune on PRA201
```
python Train_PNA.py --sampler --share_weights --image_network --contrastive --data_set PNA_keys_201.csv 
```


## 🚀 Zero-shot Blind-test on the MD150 datasets

```
python Train_PNA.py --sampler --share_weights --image_network --contrastive --data_set MD_keys.csv --onlytest --image_dir data/PRA310/PRA310/MD1200
```

