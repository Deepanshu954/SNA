# Community Detection & Information Diffusion — Interactive Showcase

A comprehensive, interactive showcase of research on **Community Detection in Social Networks using Information Diffusion Models**.

## 🔬 Overview

This project explores community structure in the **Facebook Social Circles** dataset (~4,039 nodes, ~88,000 edges) using:

- **Louvain algorithm** for community detection (12 communities, Q = 0.8318)
- **Independent Cascade (IC) model** for information diffusion simulation
- **DW-Louvain** — a novel diffusion-weighted community detection approach

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd community_detection_showcase
pip install -r requirements.txt
```

### 2. Download the Dataset

Download `facebook_combined.txt` from one of these sources:
- [SNAP Stanford](https://snap.stanford.edu/data/ego-Facebook.html)
- [Kaggle](https://www.kaggle.com/datasets/pypiahmad/social-circles)

Place it in the `data/` folder. (It may already be downloaded for you.)

### 3. Run the Streamlit App

```bash
streamlit run app/main.py
```

Open your browser to `http://localhost:8501`.

## 📱 Pages

| Page | Description |
|------|-------------|
| **Home** | Project overview and key metrics |
| **Network Explorer** | Interactive network visualization with Pyvis |
| **Community Detection** | Louvain algorithm with tunable resolution |
| **Diffusion Simulator** | IC model simulation and spread analysis |
| **DW-Louvain** | Diffusion-weighted community detection |
| **Results Dashboard** | Summary figures and downloadable results |

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Modularity Q | 0.8318 |
| Communities | 12 |
| Intra-community activations | 78% |
| Diffusion speed ratio | 2.3× |
| DW-Louvain projected Q | 0.89 |

## 🛠️ Tech Stack

- **Streamlit** — Web framework
- **NetworkX** — Graph analysis
- **python-louvain** — Community detection
- **Pyvis** — Interactive graph visualization
- **Plotly / Matplotlib** — Charts and figures
- **scikit-learn** — NMI evaluation

## 📚 Research Paper

*Community Detection in Social Networks using Information Diffusion Models*  
Harshit Singh Shakya, Manjeet Singh Jhakar, Deepanshu Chauhan  
Bennett University, Semester VI, 2024-25

## 📁 Project Structure

```
community_detection_showcase/
├── app/                          ← Streamlit app
│   ├── main.py
│   ├── pages/
│   │   ├── 1_Network_Explorer.py
│   │   ├── 2_Community_Detection.py
│   │   ├── 3_Diffusion_Simulator.py
│   │   ├── 4_DW_Louvain.py
│   │   └── 5_Results_Dashboard.py
│   └── assets/
│       └── style.css
├── src/                          ← Shared Python modules
│   ├── graph_utils.py
│   ├── community_detection.py
│   ├── diffusion.py
│   ├── dw_louvain.py
│   ├── visualize.py
│   └── evaluate.py
├── data/
│   └── facebook_combined.txt
├── notebooks/
│   └── community_detection_colab.ipynb
├── requirements.txt
└── README.md
```
