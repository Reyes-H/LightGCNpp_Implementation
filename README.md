# LightGCN++: Enhanced Graph Convolutional Network for Recommendation

[![Paper](https://img.shields.io/badge/Paper-RecSys%202024-blue)](https://dl.acm.org/doi/10.1145/3640457.3688176)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

Source code and datasets for **"Revisiting LightGCN: Unexpected Inflexibility, Inconsistency, and A Remedy Towards Improved Recommendation"** (RecSys 2024 Short Paper).

## 📖 Paper & Resources

- **Paper**: [ACM Digital Library](https://dl.acm.org/doi/10.1145/3640457.3688176)
- **Supplementary Document**: [PDF](supplementary_document.pdf)

## 🎯 Project Overview

This project implements and evaluates **LightGCN++**, an enhanced version of the LightGCN recommendation model. The implementation addresses the inflexibility and inconsistency issues identified in the original LightGCN approach through several key improvements:

### Key Features
- **Enhanced Graph Convolution**: Improved message passing mechanism with better parameter control
- **Multi-Dataset Support**: Comprehensive evaluation across 5 popular recommendation datasets
- **Flexible Architecture**: Support for both LightGCN++ and Matrix Factorization baselines
- **Extensive Evaluation**: Performance comparison across different datasets and model configurations
- **Visualization Tools**: Built-in tools for performance analysis and embedding visualization

### Supported Datasets
- **MovieLens-1M** (ml-1m): Movie recommendation dataset with 1M ratings
- **Amazon-Book**: Book recommendation dataset from Amazon
- **Yelp2018**: Business recommendation dataset from Yelp
- **LastFM**: Music recommendation dataset
- **Gowalla**: Location-based social network dataset

### Model Variants
- **LightGCN++ (lgn)**: Enhanced graph convolutional network with improved flexibility
- **Matrix Factorization (mf)**: Traditional collaborative filtering baseline
- **LightGCN**: Graph convolutional network, old model,from SIGIR 2020:https://github.com/gusye1234/LightGCN-PyTorch

## 🚀 Quick Start

### Environment Setup

```bash
# Create and activate conda environment
conda create -n lightgcnpp python=3.9
conda activate lightgcnpp

# Install dependencies
conda install git
pip install torch torchvision torchaudio numpy scipy tqdm scikit-learn pandas tensorboardX matplotlib
```

### Data Preparation

```bash
# Download MovieLens-1M dataset
python data_download.py

# Preprocess data for LightGCN++
python preprocess.py
```

### Training Models

#### LightGCN++ Training
```bash
# Train on different datasets
python code/main.py --model lgn --dataset gowalla
python code/main.py --model lgn --dataset lastfm
python code/main.py --model lgn --dataset yelp2018
python code/main.py --model lgn --dataset amazon-book
python code/main.py --model lgn --dataset ml-1m
```

#### Matrix Factorization Training
```bash
# Train MF baseline on different datasets
python code/main_mf.py --model mf --dataset gowalla
python code/main_mf.py --model mf --dataset lastfm
python code/main_mf.py --model mf --dataset yelp2018
python code/main_mf.py --model mf --dataset amazon-book
python code/main_mf.py --model mf --dataset ml-1m
```

### Key Parameters

The model uses the following default hyperparameters:
- **α (alpha)**: 0.6 - Controls the balance between ego and propagated embeddings
- **β (beta)**: -0.1 - Negative sampling parameter
- **γ (gamma)**: 0.2 - Weight for the final embedding combination
- **Learning Rate**: 0.001
- **Embedding Dimension**: 64
- **Number of Layers**: 2
 
### Simple experiment reproduction of LightGCN++ and ligtGCN

#### Training and Testing
```python SELFRec/main.py```
Run this python program every time, but manually set different parameters for the parser variable each time.
 
#### Core parameter description
- **epoch**: - The number of training rounds,always 20
 
- **α (alpha)**: - Controls the balance between ego and propagated embeddings
- **β (beta)**: - Negative sampling parameter
- **γ (gamma)**: - Weight for the final embedding combination
 
- **dataset**: - Select the dataset for this experiment
- **model_name**: - Select the model for this experiment

#### Experimental run setup
| No.experiment | dataset       | model_name   | α (alpha) | β (beta) | γ (gamma) |
|---------------|---------------|--------------|-----------|----------|-----------|
| 1             | 'amazon-book' | 'LightGCN'   | -         | -        | -         |
| 2             | 'amazon-book' | 'LightGCNpp' | 1.0       | 1.0      | 0.5       |
| 3             | 'amazon-book' | 'LightGCNpp' | 0.6       | -0.1     | 0.2       |
| 4             | 'gowalla'     | 'LightGCN'   | -         | -        | -         |
| 5             | 'gowalla'     | 'LightGCNpp' | 1.0       | 1.0      | 0.5       |
| 6             | 'gowalla'     | 'LightGCNpp' | 0.6       | -0.1     | 0.2       |
| 7             | 'lastfm'      | 'LightGCN'   | -         | -        | -         |
| 8             | 'lastfm'      | 'LightGCNpp' | 1.0       | 1.0      | 0.5       |
| 9             | 'lastfm'      | 'LightGCNpp' | 0.6       | -0.1     | 0.0       |
| 10            | 'yelp2018'    | 'LightGCN'   | -         | -        | -         |
| 11            | 'yelp2018'    | 'LightGCNpp' | 1.0       | 1.0      | 0.5       |
| 12            | 'yelp2018'    | 'LightGCNpp' | 0.6       | -0.1     | 0.0       |

## 📁 Project Structure

```
LightGCNpp/
├── code/                    # Main implementation
│   ├── main.py             # LightGCN++ training script
│   ├── main_mf.py          # Matrix Factorization training script
│   ├── model.py            # Model definitions (LightGCN++, MF)
│   ├── dataloader.py       # Data loading utilities
│   ├── Procedure.py        # Training and evaluation procedures
│   ├── utils.py            # Utility functions
│   ├── parse.py            # Command line argument parsing
│   ├── logs/               # Training logs and results
│   ├── embs/               # Saved embeddings
│   └── checkpoints/        # Model checkpoints
├── data/                   # Dataset files
│   ├── ml-1m/             # MovieLens-1M dataset
│   ├── amazon-book/        # Amazon-Book dataset
│   ├── yelp2018/          # Yelp2018 dataset
│   ├── lastfm/            # LastFM dataset
│   └── gowalla/           # Gowalla dataset
├── output/                # All generated visualizations and analysis results from visual.py
├── SELFRec/                # Simple experiment reproduction of LightGCN++ and ligtGCN
│   ├── logs/               # Output and plotting code for the experiment
│   ├── model/              # experiment models
│   │   └── graph/            # graph models
│   │       ├── LightGCN.py             # LightGCN, old model
│   │       └── LightGCNpp.py           # LightGCN++, new model, this paper
│   └── main.py             # LightGCN++ and LightGCN running script in Simple experiment reproduction
├── data_download.py        # Dataset download script
├── preprocess.py          # Data preprocessing script
├── visual.py              # Visualization and analysis tools
└── README_NEW.md          # This file
```

## 📊 Evaluation & Visualization

### Performance Metrics
The model evaluates performance using:
- **NDCG@20**: Normalized Discounted Cumulative Gain at top-20
- **Recall@20**: Recall at top-20 recommendations
- **Precision@20**: Precision at top-20 recommendations
 
In the Simple experiment reproduction stage, the following evaluation indicators are used:
- **NDCG@10**: Normalized Discounted Cumulative Gain at top-10
- **Recall@10**: Recall at top-10 recommendations
- **NDCG@20**: Normalized Discounted Cumulative Gain at top-20
- **Recall@20**: Recall at top-20 recommendations
- **NDCG@40**: Normalized Discounted Cumulative Gain at top-40
- **Recall@40**: Recall at top-40 recommendations

### Visualization Tools

Run the visualization script to generate performance comparisons:

```bash
python visual.py
```

**All generated images and visualizations will be saved in the `output/` folder.**
 
**In the Simple experiment reproduction stage, the test performance charts of different models on different data sets are obtained using the following python program and then need to be saved manually.**
```
python SELFR/logs/draw_amazon_book.py
python SELFR/logs/draw_gowalla.py
python SELFR/logs/draw_lastfm.py
python SELFR/logs/draw_yelp2018.py
```

#### The `output/` folder contains:
- **NDCG@20 Curves** (`ndcg_comparison_{dataset}.png`): For each dataset, a line plot comparing NDCG@20 across epochs for both `lgn` and `mf` models.
- **Bar Charts for Best Metrics** (`ndcg_bar_comparison.png`, `recall_bar_comparison.png`, `precision_bar_comparison.png`): For all datasets, grouped bar charts comparing the best NDCG@20, Recall@20, and Precision@20 between `lgn` and `mf`.
- **Embedding Visualizations** (`embedding_{dataset}_{model}.png`): t-SNE plots of the first 1000 user and item embeddings for each dataset and model, showing the distribution and clustering of learned representations.

Additionally, the script prints a summary table in the terminal, listing the best NDCG@20, Recall@20, and Precision@20 for each (dataset, model) pair. 

## 📈 Results

The implementation provides comprehensive evaluation results across all supported datasets. Training logs are saved in `code/logs/` with detailed performance metrics at regular intervals.

Key findings include:
- Improved flexibility compared to original LightGCN
- Consistent performance across different datasets
- Better parameter sensitivity and control
- Enhanced recommendation quality