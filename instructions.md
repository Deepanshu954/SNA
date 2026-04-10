# Community Detection Showcase — Complete Project Details & Coding Prompts

---

## PROJECT OVERVIEW

**Name:** Community Detection & Information Diffusion — Interactive Showcase  
**Based on:** Research paper "Community Detection in Social Networks using Information Diffusion Models"  
**Dataset:** Facebook Social Circles (facebook_combined.txt) — ~4,039 nodes, ~88,000 edges  
**Platforms:** (1) Streamlit Web App  (2) Google Colab Notebook

---

## FOLDER STRUCTURE

```
community_detection_showcase/
│
├── app/                          ← Streamlit app
│   ├── main.py                   ← entry point (streamlit run app/main.py)
│   ├── pages/
│   │   ├── 1_Network_Explorer.py
│   │   ├── 2_Community_Detection.py
│   │   ├── 3_Diffusion_Simulator.py
│   │   ├── 4_DW_Louvain.py
│   │   └── 5_Results_Dashboard.py
│   └── assets/
│       └── style.css
│
├── src/                          ← shared Python modules
│   ├── graph_utils.py
│   ├── community_detection.py
│   ├── diffusion.py
│   ├── dw_louvain.py
│   ├── visualize.py
│   └── evaluate.py
│
├── data/
│   ├── facebook_combined.txt     ← main dataset
│   └── ego_circles/              ← ground-truth ego-circle files
│
├── notebooks/
│   └── community_detection_colab.ipynb   ← Google Colab notebook
│
├── requirements.txt
└── README.md
```

---

## TECH STACK

| Component         | Library/Tool                     |
|-------------------|----------------------------------|
| Web framework     | Streamlit >= 1.32                |
| Graph analysis    | NetworkX >= 3.0                  |
| Community detect  | python-louvain == 0.16           |
| Diffusion sim     | Custom IC model (pure NetworkX)  |
| Interactive graph | pyvis >= 0.3.2                   |
| Visualisation     | matplotlib >= 3.8, plotly >= 5.0 |
| Data              | pandas >= 2.0, numpy >= 1.24     |
| Evaluation        | scikit-learn >= 1.3 (for NMI)    |
| Export            | kaleido (plotly PNG export)      |

requirements.txt:
```
streamlit>=1.32
networkx>=3.0
python-louvain==0.16
pyvis>=0.3.2
matplotlib>=3.8
plotly>=5.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
kaleido
```

---

## FEATURES PER PAGE (STREAMLIT)

### Page 1 — Network Explorer
- Load and display the Facebook graph using Pyvis (interactive HTML)
- Sidebar sliders: max nodes to display (50 / 200 / 500 / all), spring layout seed
- Stats card row: nodes, edges, avg degree, clustering coefficient, density
- Degree distribution histogram (plotly)
- Top-10 hub nodes table
- Download graph as HTML button

### Page 2 — Community Detection
- Run Louvain with one click; show spinner while computing
- Sidebar: resolution parameter slider (0.5 → 2.0), random seed
- Output: modularity score, number of communities, largest/smallest community
- Community-coloured network (pyvis, each community = distinct colour)
- Bar chart: community size distribution (plotly)
- Table: community ID, size, top-3 hub nodes per community
- Download partition as CSV

### Page 3 — Diffusion Simulator
- Select seed strategy: highest degree / random / custom node ID
- Slider: propagation probability p (0.01 → 0.5)
- Slider: number of simulation runs (10 → 200)
- "Run Simulation" button → show progress bar
- Animated spread curve (plotly line chart, updates per step)
- Pie chart: intra vs inter-community activations
- Metrics: mean spread, std dev, communities reached, 2.3× speed comparison
- Colour the network by activation time step (pyvis)

### Page 4 — DW-Louvain
- Explain the 2-stage pipeline visually (SVG flowchart embedded)
- Sidebar: N simulations (100 → 1000), propagation prob, seed
- "Compute Weights" button → progress bar → weight histogram
- "Run Weighted Louvain" button → new partition
- Side-by-side comparison: standard Louvain vs DW-Louvain
  - Modularity Q, NMI, community count, diffusion containment
- Grouped bar chart: all metrics side by side (plotly)
- Algorithm comparison table (all 5 algorithms)

### Page 5 — Results Dashboard
- Full-width layout with all summary metrics at top
- Grid of all 6 paper figures (matplotlib, reproduced exactly)
- Download all figures as ZIP
- Download metrics comparison table as CSV
- "Generate Report" button → builds a summary PDF using matplotlib

---

## FEATURES (GOOGLE COLAB)

### Section 1 — Setup & EDA
- pip install block
- Dataset download from Kaggle (or upload widget)
- Load graph, print stats
- Degree distribution plot
- Top hub nodes

### Section 2 — Community Detection
- Louvain algorithm with modularity score
- Community visualisation (spring layout, tab20 colours)
- Community size bar chart
- Per-community stats table

### Section 3 — IC Diffusion Simulation
- Full IC model implementation
- 100-run simulation loop with tqdm progress
- Spread distribution histogram
- Intra vs inter community activation analysis
- Diffusion speed comparison bar chart

### Section 4 — DW-Louvain Implementation
- Full DW-Louvain code (N=1000 simulations)
- Edge weight histogram
- Weighted Louvain detection
- Side-by-side comparison with baseline

### Section 5 — Evaluation & Paper Figures
- NMI vs ego-circle ground truth
- Conductance calculation
- All 6 paper figures reproduced in a single cell
- Algorithm comparison table and grouped bar chart
- Export all results to CSV

---

## KEY ALGORITHMS TO IMPLEMENT

### IC Diffusion (graph_utils → diffusion.py)
```python
def simulate_ic(G, seed_node, prob=0.1, seed=None):
    rng = random.Random(seed)
    activated = {seed_node}
    queue = deque([seed_node])
    activation_time = {seed_node: 0}
    step = 0
    while queue:
        step += 1
        next_queue = []
        for node in list(queue):
            for nbr in G.neighbors(node):
                if nbr not in activated and rng.random() < prob:
                    activated.add(nbr)
                    next_queue.append(nbr)
                    activation_time[nbr] = step
        queue = deque(next_queue)
    return activated, activation_time
```

### Intra/Inter Analysis
```python
def analyze_community_spread(G, partition, activated):
    intra = inter = 0
    for u, v in G.edges():
        if u in activated and v in activated:
            if partition[u] == partition[v]:
                intra += 1
            else:
                inter += 1
    return intra, inter
```

### DW-Louvain
```python
def compute_diffusion_weights(G, N=1000, p=0.1):
    weights = {edge: 0 for edge in G.edges()}
    nodes = list(G.nodes())
    for _ in tqdm(range(N)):
        seed = random.choice(nodes)
        activated, _ = simulate_ic(G, seed, prob=p)
        for u, v in G.edges():
            if u in activated and v in activated:
                weights[(u, v)] = weights.get((u, v), 0) + 1
                weights[(v, u)] = weights.get((v, u), 0) + 1
    for k in weights:
        weights[k] /= N
    return weights

def dw_louvain(G, N=1000, p=0.1):
    weights = compute_diffusion_weights(G, N, p)
    G_w = G.copy()
    for u, v in G_w.edges():
        G_w[u][v]['weight'] = weights.get((u, v), 0.0)
    partition = community_louvain.best_partition(G_w, weight='weight')
    Q = community_louvain.modularity(partition, G_w, weight='weight')
    return partition, Q
```

---

---

# CODING PROMPT 1 — STREAMLIT WEB APP
# (Use this prompt with Claude / GPT-4o / Gemini / Cursor / GitHub Copilot)

---

```
You are an expert Python developer. Build a complete, production-quality Streamlit 
multi-page web application that showcases a research paper on Community Detection in 
Social Networks using Information Diffusion Models.

== CONTEXT ==
Research paper findings:
- Dataset: Facebook Social Circles (facebook_combined.txt from SNAP/Kaggle)
  — ~4,039 nodes, ~88,000 edges, undirected, unweighted
- Algorithm: Louvain community detection → 12 communities, modularity Q = 0.8318
- IC diffusion simulation (p=0.1, 100 runs, highest-degree seed):
  → avg spread 127 nodes, 78% intra-community activations, 2.3× intra speed
- Proposed method: DW-Louvain (projected Q=0.89, NMI=0.81 vs ego-circles)
- Baseline comparison: Girvan-Newman (Q=0.61), Label Prop (Q=0.71), Infomap (Q=0.78)

== FOLDER STRUCTURE TO CREATE ==
community_detection_showcase/
├── app/
│   ├── main.py                         ← streamlit entry point
│   └── pages/
│       ├── 1_Network_Explorer.py
│       ├── 2_Community_Detection.py
│       ├── 3_Diffusion_Simulator.py
│       ├── 4_DW_Louvain.py
│       └── 5_Results_Dashboard.py
├── src/
│   ├── graph_utils.py
│   ├── community_detection.py
│   ├── diffusion.py
│   ├── dw_louvain.py
│   ├── visualize.py
│   └── evaluate.py
├── data/
│   └── facebook_combined.txt
└── requirements.txt

== REQUIREMENTS ==
streamlit>=1.32, networkx>=3.0, python-louvain==0.16, pyvis>=0.3.2,
matplotlib>=3.8, plotly>=5.0, pandas>=2.0, numpy>=1.24, scikit-learn>=1.3

== SRC MODULE SPECS ==

graph_utils.py:
- load_graph(path) → nx.Graph with error handling
- get_stats(G) → dict with nodes, edges, avg_degree, density, avg_clustering
- get_top_hubs(G, n=10) → list of (node, degree)
- degree_distribution(G) → (degrees, counts) arrays

community_detection.py:
- detect_communities(G, resolution=1.0, seed=42) → (partition_dict, modularity_score)
- community_stats(G, partition) → pd.DataFrame with community_id, size, top_nodes
- get_community_colors(partition) → dict {node: hex_color} using tab20

diffusion.py:
- simulate_ic(G, seed_node, prob=0.1, rng_seed=None) → (activated_set, activation_time_dict)
- run_batch(G, seed_node, prob, n_runs) → list of (activated_set, activation_time)
- analyze_spread(runs) → dict with mean, std, max, min
- analyze_intra_inter(G, partition, activated) → (intra_count, inter_count)
- diffusion_speed_comparison(G, partition, runs) → dict {intra_steps, inter_steps}

dw_louvain.py:
- compute_diffusion_weights(G, N=1000, p=0.1, callback=None) → dict {(u,v): weight}
- run_dw_louvain(G, N=1000, p=0.1) → (partition, Q, weights)
- compare_partitions(G, partition_base, partition_dw) → comparison_dict

evaluate.py:
- compute_nmi(partition, ground_truth_circles) → float
- compute_conductance(G, partition) → dict {community_id: conductance}
- compute_diffusion_containment(G, partition, activated) → float

visualize.py — ALL matplotlib functions, return (fig, ax), never plt.show():
- plot_degree_dist(G) → fig
- plot_community_sizes(partition) → fig
- plot_diffusion_spread(activation_curves) → fig
- plot_intra_inter_pie(intra, inter) → fig
- plot_algorithm_comparison() → fig (hardcoded comparison data)
- plot_results_dashboard() → fig (3-panel summary)

== PAGE SPECS ==

main.py:
- st.set_page_config(page_title="Community Detection Showcase", layout="wide")
- Landing page with project title, abstract, 4 metric KPI cards (nodes, edges, Q, communities)
- Navigation hint to use sidebar
- Use st.cache_data to load graph once

1_Network_Explorer.py:
- Load graph using graph_utils.load_graph()
- Sidebar: "Max nodes to display" selectbox [200, 500, 1000, "All"]
- Sample nodes for display using largest connected component if needed
- Display pyvis graph in st.components.v1.html() with height=550
- Pyvis config: show_buttons=False, bgcolor="#ffffff", font_color="#333333"
- Stats cards row using st.columns(5): nodes, edges, avg_degree, clustering, density
- Plotly degree distribution histogram below
- Top hubs dataframe (st.dataframe with column config)

2_Community_Detection.py:
- Sidebar: resolution slider (0.5 to 2.0, step 0.1), seed input, "Detect Communities" button
- Use st.session_state to cache partition between reruns
- After detection: show Q score as st.metric, n_communities as st.metric
- Two columns: left = community-colored pyvis graph, right = community size bar chart (plotly)
- Community stats table below with st.dataframe
- st.download_button for CSV of partition

3_Diffusion_Simulator.py:
- Sidebar: seed strategy radio ["Highest degree", "Random", "Custom node ID"]
  If "Custom": st.number_input for node ID
- p slider (0.01 to 0.5, step 0.01), n_runs slider (10 to 200, step 10)
- "Run Simulation" button with st.progress bar
- After run: 3 metric cards (mean spread, communities reached, intra %)
- Plotly animated line chart showing cumulative spread over time steps
- Two columns: pyvis graph coloured by activation time | intra/inter pie chart
- Callout box: "Intra-community spread is X× faster than inter-community"

4_DW_Louvain.py:
- Explain the pipeline with a st.expander showing pseudocode
- Sidebar: N_sims slider (100–1000, step 100), p slider, "Run DW-Louvain" button
- Stage 1 progress bar for weight computation
- Stage 2 progress for Louvain
- After completion:
  Left column: standard Louvain metrics
  Right column: DW-Louvain metrics
  Delta shown with st.metric's delta parameter
- Plotly grouped bar chart: Modularity, NMI, Diffusion Containment for all 5 algorithms
- Full comparison table: Girvan-Newman, Label Prop, Infomap, Louvain, DW-Louvain
  (use projected/expected values for DW-Louvain, mark with ★)

5_Results_Dashboard.py:
- Full-width KPI row: Q=0.8318, 12 communities, 78% intra, 2.3× speed
- 2×3 grid of all 6 matplotlib figures from visualize.py
- Each figure in a card with title and caption matching the paper
- st.download_button for each individual figure (PNG)
- "Download All Figures" → ZIP of all 6 PNGs
- Metrics comparison table (pd.DataFrame → st.dataframe)
- "Download CSV" button for the table

== STYLE GUIDELINES ==
- Use st.markdown with custom CSS for metric cards:
  background #f0f4f8, border-left 4px solid #1A3C5E, border-radius 8px
- Color scheme: primary #1A3C5E (navy), accent #2471A3 (blue), success #27AE60
- All pyvis graphs: bgcolor="#fafafa", font_color="#333", node size proportional to degree
- Community colors: use matplotlib tab20 colormap, convert to hex
- Show st.spinner() during any computation longer than 1 second
- Use st.cache_data for graph loading; use st.session_state for computed partitions

== IMPORTANT IMPLEMENTATION NOTES ==
1. The IC diffusion simulate_ic must return activation_time dict (node → step number)
   so the pyvis graph on page 3 can colour nodes by activation time
2. DW-Louvain weight computation is slow — use a callback parameter so the Streamlit
   progress bar can update: callback(i, N) called after each of N simulations
3. For pyvis in Streamlit, write the HTML to a temp file then read it back:
   from pyvis.network import Network
   net = Network(height="500px", width="100%", bgcolor="#fafafa")
   net.from_nx(G_sample)
   net.save_graph("/tmp/graph.html")
   with open("/tmp/graph.html") as f:
       st.components.v1.html(f.read(), height=520)
4. For the plotly animated spread chart, pre-compute cumulative activations per time
   step across all runs and plot mean ± std as a ribbon (go.Scatter with fill='tonexty')
5. Add a README.md explaining how to:
   - pip install -r requirements.txt
   - Download the dataset from Kaggle
   - Run: streamlit run app/main.py

Write ALL files completely, end to end, no placeholders, no "# TODO" comments.
Every function must be fully implemented. The app must run with:
  streamlit run app/main.py
```

---

---

# CODING PROMPT 2 — GOOGLE COLAB NOTEBOOK
# (Use this prompt with Claude / GPT-4o / Gemini — paste into a new chat)

---

```
You are an expert Python developer and data scientist. Create a complete, 
publication-quality Google Colab notebook (.ipynb format) that reproduces and 
showcases the results from a research paper on Community Detection in Social 
Networks using Information Diffusion Models.

== RESEARCH PAPER CONTEXT ==
Title: Community Detection in Social Networks using Information Diffusion Models
Authors: Harshit Singh Shakya, Manjeet Singh Jhakar, Deepanshu Chauhan
Institution: Bennett University, Semester VI, 2024-25
Dataset: Facebook Social Circles (SNAP/Kaggle: pypiahmad/social-circles)
  — ~4,039 nodes, ~88,000 edges, undirected unweighted ego-network

Key Results to Reproduce:
- Louvain: 12 communities, Q = 0.8318, avg clustering = 0.6055
- IC diffusion (p=0.1, 100 runs): mean spread 127, std 22, intra% = 78%, 2.3× speed
- DW-Louvain projected: Q = 0.89, NMI = 0.81 vs ego-circles
- Baselines: Girvan-Newman Q=0.61, Label Prop Q=0.71, Infomap Q=0.78

== NOTEBOOK STRUCTURE ==
The notebook must have exactly 5 sections, each clearly titled with a large Markdown 
header and a brief introduction cell explaining what the section covers.

──────────────────────────────────────────────
SECTION 1: Setup & Exploratory Data Analysis
──────────────────────────────────────────────

Cell 1.1 — Installation (code cell):
!pip install python-louvain pyvis tqdm -q

Cell 1.2 — Imports (code cell):
All imports: networkx, community (louvain), matplotlib, numpy, pandas, 
collections, random, tqdm, sklearn.metrics, itertools, warnings

Cell 1.3 — Dataset loading (code cell):
Provide two options with clear markdown comment above each:
Option A: Upload from local machine using files.upload() 
Option B: Download directly using:
  !wget -q "https://raw.githubusercontent.com/..." or from Kaggle API
Then: G = nx.read_edgelist('facebook_combined.txt', create_using=nx.Graph(), nodetype=int)
Print: nodes, edges, is_connected, number_of_components

Cell 1.4 — Network statistics (code cell):
Compute and print as a formatted table using pandas DataFrame:
  - Number of nodes
  - Number of edges
  - Average degree
  - Average clustering coefficient
  - Network density
  - Diameter (on largest connected component)
  - Graph type (undirected, unweighted)
Print: "Top 10 Hub Nodes by Degree:" table

Cell 1.5 — Degree distribution (code cell + plot):
Log-log histogram using matplotlib, styled with:
  figsize=(8,4), color '#1A3C5E', edgecolor 'white', both axes log scale
  xlabel, ylabel, title, grid alpha 0.25
Save as 'fig1_degree_dist.png'
Caption: "Figure 1: Log-log degree distribution confirming scale-free power-law topology"

──────────────────────────────────────────────
SECTION 2: Community Detection (Louvain)
──────────────────────────────────────────────

Cell 2.1 — Louvain detection (code cell):
partition = community_louvain.best_partition(G, random_state=42)
Q = community_louvain.modularity(partition, G)
Print formatted results:
  "Modularity Score (Q): 0.XXXX"
  "Number of Communities: XX"
  "Average Community Size: XXX"
  "Largest Community: XXX nodes"

Cell 2.2 — Community size distribution plot:
Bar chart with matplotlib, communities sorted by size descending
Colors from tab20 colormap, red dashed mean line
figsize=(9,4), save as 'fig2_community_sizes.png'
Caption match paper Figure 2

Cell 2.3 — Network visualisation (code cell):
Spring layout (seed=42), nodes coloured by community (tab20), 
node size proportional to degree (between 20 and 200)
figsize=(10,8), edge color='#CCCCCC', alpha=0.6
Title: "Figure: Community Structure — Louvain Detection (12 Communities, Q=0.8318)"
Note: "Each color represents a distinct detected community"

Cell 2.4 — Convergence simulation (code cell + plot):
Since we can't see internal iterations, simulate the convergence story:
Run Louvain 15 times with increasing resolution and plot Q values
This shows the algorithm's sensitivity and stability
Title: "Modularity across resolution parameter values"

Cell 2.5 — Community statistics table:
For each community: ID, size, top 3 nodes by degree, internal edge density
Display as styled DataFrame

──────────────────────────────────────────────
SECTION 3: Information Diffusion Simulation
──────────────────────────────────────────────

Cell 3.1 — IC model implementation (code cell):
Full implementation of:
  simulate_ic(G, seed_node, prob=0.1, rng_seed=None) 
    → returns (activated_set, activation_time_dict)
  The activation_time_dict maps each activated node → time step it was activated

Cell 3.2 — Batch simulation (code cell):
seed = max(G.nodes(), key=lambda n: G.degree(n))
Run 100 simulations with tqdm progress bar
Collect: spread sizes, activation time series per step, activated sets

Cell 3.3 — Spread analysis results (code cell + display):
Print formatted results table matching Table 5 from paper:
  mean spread, std, max, min, communities reached, intra%, inter%, speed ratio

Cell 3.4 — Diffusion spread over time (code cell + plot):
Compute mean cumulative activated nodes per time step across all runs
Compute std for ribbon
Plot two lines: intra-community only vs full network
Shaded ribbon for std, fill_between for gap between intra/full
figsize=(8,4), save as 'fig3_diffusion_spread.png'
Caption match paper Figure 3

Cell 3.5 — Intra vs inter pie chart (code cell + plot):
figsize=(5,4), explode=(0.05,0.05), colors=['#1A3C5E','#E67E22']
White text autopct, save as 'fig4_intra_inter_pie.png'
Caption match paper Figure 4

Cell 3.6 — Diffusion speed analysis (code cell + plot):
Compare average steps to reach 50%, 75%, 90% penetration
For intra-community activations vs cross-community activations
Grouped bar chart showing the 2.3× gap clearly

──────────────────────────────────────────────
SECTION 4: DW-Louvain — Proposed Framework
──────────────────────────────────────────────

Cell 4.1 — Framework explanation (markdown cell):
Explain the two-stage pipeline with formatted math:
Stage 1: w(u,v) = |{k : u ∈ Aₖ and v ∈ Aₖ}| / N
Stage 2: Q_w = (1/2W) Σᵢⱼ [w(i,j) − s(i)s(j)/2W] δ(cᵢ,cⱼ)
Include the full pseudocode in a code block

Cell 4.2 — DW-Louvain implementation (code cell):
Full implementation:
  compute_diffusion_weights(G, N=500, p=0.1) → dict of edge weights
    Uses tqdm for progress tracking
    Handles both (u,v) and (v,u) directions
  run_dw_louvain(G, N=500, p=0.1) → (partition, Q, weights)

Cell 4.3 — Run DW-Louvain (code cell):
N=500 for Colab speed (note: paper used N=1000 for full results)
Print: "Stage 1: Computing diffusion weights..."
Print progress via tqdm
Print: "Stage 2: Running weighted Louvain..."
Print: Q_dw, n_communities_dw

Cell 4.4 — Edge weight distribution (code cell + plot):
Histogram of all edge weights, log scale y-axis
Annotate: mean weight, threshold separating intra/inter
figsize=(7,4), color '#1A3C5E'
Caption: shows bimodal distribution separating intra (high) from inter (low)

Cell 4.5 — Side-by-side comparison (code cell + plot):
Two pyvis graphs or two matplotlib spring layouts side by side:
Left: standard Louvain communities
Right: DW-Louvain communities
Title: "Standard Louvain vs DW-Louvain Community Partitions"

──────────────────────────────────────────────
SECTION 5: Evaluation & Paper Figures
──────────────────────────────────────────────

Cell 5.1 — Metrics computation (code cell):
Implement:
  compute_nmi(partition, ground_truth) → NMI score using sklearn
  compute_conductance(G, partition) → per-community conductance
  compute_diffusion_containment(G, partition, activated_sets) → float
  
For NMI: convert partition and ground_truth to label arrays for sklearn.metrics.normalized_mutual_info_score
Note: ground truth from ego-circles files or simulated for demonstration

Cell 5.2 — Algorithm comparison table (code cell + display):
Full comparison DataFrame with all 5 algorithms:
Columns: Algorithm, Modularity Q, NMI, Diffusion Containment, Scalability
Mark DW-Louvain values as "0.89 ★ (projected)"
Display as styled DataFrame with blue header

Cell 5.3 — Algorithm comparison chart (code cell + plot):
3-metric grouped bar chart (Modularity, NMI, Diffusion Containment)
figsize=(9,4), 3 color groups, annotate bars with values
DW-Louvain bar slightly different shade to indicate projected
save as 'fig5_algorithm_comparison.png'
Caption match paper Figure 5

Cell 5.4 — Results summary dashboard (code cell + plot):
3-panel figure in 1 row, figsize=(12,4):
Panel A: Spread size histogram (100 runs)
Panel B: Steps to penetration (50/75/90%) — grouped bars intra vs inter
Panel C: Louvain vs DW-Louvain head-to-head on 3 metrics
save as 'fig6_results_dashboard.png'
Caption match paper Figure 6

Cell 5.5 — All figures in one grid (code cell + plot):
Reproduce all 6 paper figures in a 2×3 matplotlib grid
figsize=(18,10), tight_layout
Each subplot has correct Figure N caption from the paper

Cell 5.6 — Export results (code cell):
Save all metrics to results_summary.csv
Save partition to community_partition.csv (node_id, community_id)
Save edge weights to diffusion_weights.csv (node1, node2, weight)
Print download links using google.colab.files.download()

Cell 5.7 — Conclusions (markdown cell):
Summary of all key findings matching paper conclusions:
- Q = 0.8318, 12 communities
- 78% intra-community activations
- 2.3× intra speed advantage
- DW-Louvain projected improvements
- Future work directions

== STYLE REQUIREMENTS FOR ALL PLOTS ==
- figsize: use (8,4) for single charts, (9,5) for complex, (12,4) for dashboards
- DPI: 150 for all savefig calls
- Color scheme: primary #1A3C5E, secondary #2471A3, accent #E67E22
- All axes: grid(True, alpha=0.25, linestyle='--')
- Font: default matplotlib but all titles fontsize=12, fontweight='bold'
- Every figure: fig.tight_layout(pad=0.8)
- Every figure: plt.savefig('figN_name.png', dpi=150, bbox_inches='tight')
- Every savefig followed by plt.show()
- Captions: printed as a markdown cell below each plot cell, matching paper exactly

== NOTEBOOK FORMATTING REQUIREMENTS ==
- Every section header: # Section N: Title (H1 markdown, with emoji icon)
- Every subsection: ## N.M Sub-title (H2)
- Every code cell preceded by a 1-2 sentence markdown description
- Key results printed as formatted boxes using print("="*50) borders
- After each major result, a markdown "Key Finding" callout using blockquote (>)
- Notebook must run top to bottom with Runtime → Run All with zero errors
- Add a cell at the very top: hardware check (GPU/CPU, RAM available)
- Add a "Table of Contents" markdown cell at the top after the title

== OUTPUT ==
Provide the complete notebook as valid JSON (.ipynb format).
Every cell must be complete. No placeholder cells. No "see above" references.
The notebook must be self-contained and reproducible.
```

---

## HOW TO USE THESE PROMPTS

### For the Streamlit App:
1. Open a new chat with Claude / GPT-4o / Cursor
2. Paste **CODING PROMPT 1** exactly as written
3. Ask it to generate all files one by one if needed
4. Once files are generated:
   ```bash
   pip install -r requirements.txt
   streamlit run app/main.py
   ```

### For the Colab Notebook:
1. Open a new chat with Claude / GPT-4o / Gemini
2. Paste **CODING PROMPT 2** exactly as written
3. Request output as .ipynb JSON format
4. Save as community_detection_colab.ipynb
5. Upload to Google Colab and run with Runtime → Run All

### Tips:
- If the model truncates output, say: "Continue from where you left off — next file is X"
- For large files, ask section by section: "Now write only src/diffusion.py"
- Test each src/ module independently before running the full app
- For Colab: make sure facebook_combined.txt is uploaded before Section 1

---

## DATASET ACCESS

Download facebook_combined.txt from:
- Kaggle: https://www.kaggle.com/datasets/pypiahmad/social-circles
- SNAP: https://snap.stanford.edu/data/ego-Facebook.html

Place in the data/ folder for Streamlit, or upload directly in Colab.
