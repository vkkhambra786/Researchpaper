# RDBMS to NoSQL Schema Transformation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Research Paper](https://img.shields.io/badge/Paper-Published-orange)](#)

## Overview

This repository contains the implementation of **automated schema transformation from RDBMS to NoSQL databases** using machine learning approaches. The research explores three different methodologies for optimizing NoSQL schema design based on query patterns and data relationships.

### Key Features
- **Hierarchical Clustering Approach**: Groups related tables based on query patterns and foreign key relationships
- **Three ML Approaches**: Supervised Learning, Reinforcement Learning, and Graph Neural Networks
- **TPC-H Dataset**: Industry-standard benchmark for testing and validation
- **Automated Decision Making**: Predicts "embed" vs "reference" relationships for optimal NoSQL schema design

---

## Table of Contents
- [Research Background](#research-background)
- [Approaches](#approaches)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Research Background

Traditional RDBMS schemas are optimized for normalization and ACID properties, while NoSQL databases prioritize denormalization for performance and scalability. Manual schema transformation is complex and error-prone. This research presents automated approaches using:

- **Query Pattern Analysis**: Analyzes co-access patterns from TPC-H queries
- **Relationship Modeling**: Leverages foreign key constraints and table sizes
- **ML-Based Optimization**: Three different machine learning paradigms for decision-making

**Published Paper**: *[Schema Transformation from RDBMS to NOSQL Using Hierarchical Clustering]*  

---

## Approaches

### 1. Supervised Learning (Random Forest)
- **Method**: Classification-based approach using labeled training data
- **Model**: Random Forest with 150 trees
- **Features**: Query overlap, size ratio, relationship cardinality
- **Accuracy**: ~83% cross-validation accuracy
- **Best For**: Static schemas with well-defined patterns

**Script**: [`supervised_learning2.py`](supervised_learning2.py)

### 2. Reinforcement Learning (Deep Q-Network)
- **Method**: Agent-based learning with reward signals
- **Model**: DQN with MLP policy
- **Reward System**: +3 for optimal embeds, penalties for large embeds
- **Training**: 30,000 timesteps with exploration decay
- **Best For**: Dynamic workloads requiring adaptive decisions

**Script**: [`reinforcement_learning.py`](reinforcement_learning.py)

### 3. Graph Neural Network (GCN)
- **Method**: Graph-aware relational learning
- **Model**: SchemaGNN with GCNConv layers
- **Features**: Node (table) and edge (FK) features with neighborhood aggregation
- **Loss**: Binary Cross-Entropy, converges to ~0.0000
- **Best For**: Relational data with complex graph structures (Recommended)

**Script**: [`graph_neural_network.py`](graph_neural_network.py)

### Hierarchical Clustering
- **Method**: Hierarchical clustering of tables based on query co-occurrence
- **Linkage**: Ward's method for optimal cluster compactness
- **Output**: Dendrogram visualization and cluster assignments

**Scripts**: 
- [`agglomerative_clustering2.py`]( agglomerative_clustering2.py)

---

## Dataset

### TPC-H Benchmark
The **TPC-H** dataset is an industry-standard decision support benchmark consisting of:

- **8 Tables**: Customer, Orders, Lineitem, Part, Supplier, Partsupp, Nation, Region
- **22 Queries**: Complex analytical queries with joins and aggregations
- **Relationships**: 7 foreign key constraints
- **Scale Factor**: SF-1 (1GB dataset)

**Data Source**: [TPC-H Benchmark Specification](http://www.tpc.org/tpch/)

---

## Installation

### Prerequisites
- Python 3.8+
- PostgreSQL (for TPC-H data)
- Virtual environment (recommended)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/vkkhambra786/Researchpaper.git
cd Researchpaper
```

2. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables** (for database credentials):
```bash
cp .env.example .env
# Edit .env with your database credentials
```

---

## Usage

### 1. Supervised Learning Approach
```bash
python supervised_learning2.py
```
**Output**: 
- `nosql_schema.json` - Generated NoSQL schema
- `schema_visualization.png` - Visual graph representation
- Cross-validation accuracy and feature importance

### 2. Reinforcement Learning Approach
```bash
python reinforcement_learning.py
```
**Output**:
- `nosql_schema.json` - Generated NoSQL schema with rewards
- Decision summary with reward scores
- Training progress and loss curves

### 3. Graph Neural Network Approach
```bash
python graph_neural_network.py
```
**Output**:
- `nosql_schema.json` - Generated NoSQL schema
- `nosql_schema_transformation.png` - Graph visualization
- Training loss convergence (0.0000)

### 4. Agglomerative Clustering
```bash
python agglomerative_clustering.py
```
**Output**:
- Dendrogram visualization
- Cluster assignments for tables
- Query co-occurrence matrix

---

## Project Structure

```
Researchpaper/
├── supervised_learning2.py       # Random Forest approach
├── reinforcement_learning.py     # DQN approach
├── graph_neural_network.py       # GNN approach
├── agglomerative_clustering.py   # Clustering analysis
├── agglomerative_clust3.py       # Alternative clustering
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── .env.example                  # Environment variables template
├── clueterinf.txt                # Model training documentation
├── nosql_schema.json             # Generated NoSQL schema
├── schema_visualization.png      # Schema graph visualization
├── nosql_schema_transformation.png  # GNN output graph
└── README.md                     # This file
```

---

## Results

### Comparison of Approaches

| Approach | Accuracy/Loss | Training Time | Adaptability | Best Use Case |
|----------|---------------|---------------|--------------|---------------|
| **Supervised Learning** | 83% accuracy | ~5 seconds | Low (static) | Labeled data, quick prototyping |
| **Reinforcement Learning** | Reward-based | ~2 minutes (30k steps) | High (dynamic) | Evolving schemas, exploration |
| **Graph Neural Network** | Loss: 0.0000 | ~30 seconds (300 epochs) | Medium | Relational data (Recommended) |

### Key Findings
- **GNN outperforms** other approaches for relational data due to neighborhood-aware learning
- **Agglomerative clustering** effectively identifies table groupings based on query patterns
- **TPC-H queries** show strong co-access patterns (e.g., Orders-Lineitem, Customer-Nation)

### Example Schema Transformations
- **Embed**: Lineitem in Orders (high query overlap)
- **Embed**: Nation in Customer (small size, frequent joins)
- **Reference**: Orders in Customer (large child table)

---
 

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## Contact

**Author**: [Vikas]  
**Email**: [vkkhambra786@gmail.com]  
**GitHub**: [@vkkhambra786](https://github.com/vkkhambra786)  

---

## Acknowledgments

- TPC-H Benchmark for providing the standard dataset
- Stable Baselines3 for RL implementation
- PyTorch Geometric for GNN framework
- scikit-learn for ML utilities

---


**Star ⭐ this repository if you find it useful!**