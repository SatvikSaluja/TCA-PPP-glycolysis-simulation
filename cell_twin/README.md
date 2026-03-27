<img width="1901" height="990" alt="image" src="https://github.com/user-attachments/assets/68c6af40-4196-409d-82ef-6932f56146cb" /># 🧬 Cell Digital Twin — Metabolic Atlas v2
### Full-Stack Metabolic Simulation: FastAPI + PyTorch GNN + Escher-Style Pathway Map
Working link - https://tca-ppp-glycolysis-simulation.vercel.app/
---

## 🗂 Project Structure

```
cell_twin/
├── backend/
│   ├── main.py           # FastAPI REST API (35 endpoints)
│   ├── kegg_data.py      # Real KEGG metabolic data (35 reactions, 50+ metabolites)
│   ├── gnn_model.py      # PyTorch Geometric GNN (GATv2 + FiLM conditioning)
│   ├── requirements.txt  # Python dependencies
│   └── models/
│       └── gnn_weights.pt  # Saved GNN weights (auto-generated on first run)
└── frontend/
    └── index.html        # Complete self-contained frontend
```

---

## 🚀 Quick Start

### 1. Install Python dependencies
```bash
cd backend
pip install -r requirements.txt

# If PyTorch Geometric has issues:
pip install torch==2.3.1
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
```

### 2. Start the backend
```bash
cd backend
python main.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 3. Open the frontend
```bash
# Simply open frontend/index.html in your browser
# Or serve it:
cd frontend && python -m http.server 3000
# Open http://localhost:3000
```
---

## 🔬 What's Implemented

### 4 Complete Metabolic Pathways
| Pathway | Reactions | Direction | Key Output |
|---------|-----------|-----------|------------|
| **Glycolysis** | 10 steps | Glucose → Pyruvate | 2 ATP + 2 NADH |
| **Gluconeogenesis** | 11 steps | Pyruvate → Glucose | Requires 4 ATP + 2 GTP |
| **TCA Cycle** | 8 steps | Acetyl-CoA → CO₂ | 3 NADH + 1 FADH₂ + 1 GTP |
| **Pentose Phosphate** | 8 steps | G6P → R5P | 2 NADPH + R5P |

Plus: **Fermentation** (LDH) and **Oxidative Phosphorylation** (ETC + ATP synthase)

### PyTorch GNN Architecture
- **GATv2Conv** × 3 layers (Graph Attention Network v2)
- **FiLM conditioning** — environmental context [glucose, O₂, enzyme, temp] modulates all layers
- **Bipartite graph**: metabolites = nodes, reactions = virtual nodes
- **Edge features**: stoichiometry, ΔG°, reversibility
- **Training**: 1500 biologically-constrained synthetic samples, 100 epochs, cosine LR
- **Output**: non-negative flux per reaction via `softplus` activation

### KEGG Integration
- Live REST calls to `https://rest.kegg.jp/get/{pathway_id}`
- Embedded fallback data from:
  - hsa00010 (Glycolysis/Gluconeogenesis)
  - hsa00020 (TCA Cycle)
  - hsa00030 (Pentose Phosphate)

### Escher-Style Pathway Map
- Metabolites as **sized circles** (radius ∝ flux magnitude)
- Reactions as **rotated diamond nodes**
- Edges with **directional arrows** + width/opacity ∝ flux
- Animated **flux particles** flowing along active edges
- Compartment boundaries: **cytosol** + **mitochondria**
- D3.js zoom/pan with scroll-to-zoom

---

## 🧠 API Endpoints

```
GET  /api/health           → System status + model availability
GET  /api/pathways         → Full graph structure (nodes + edges)
GET  /api/reactions        → All 35 reactions with regulation info
GET  /api/metabolites      → All metabolite metadata (KEGG IDs, formulas)
POST /api/simulate         → FBA heuristic flux simulation
POST /api/simulate/gnn     → GNN-based flux prediction
GET  /api/kegg/{id}        → Live KEGG REST pathway data
GET  /api/atp_yield        → ATP accounting by condition
POST /api/train            → Retrain GNN (background task)
GET  /api/model_status     → GNN training status
```

Example request:
```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"glucose": 8.0, "oxygen": 0.21, "enzyme_activity": 1.0, "temperature": 37.0}'
```

---

## 📊 Biological Details

### Regulation Points Modeled
- **PFK-1**: Inhibited by ATP/citrate, activated by AMP/F-2,6-BP
- **PDH**: Inhibited by Acetyl-CoA/NADH; activated by Ca²⁺
- **IDH**: Activated by ADP/Ca²⁺; inhibited by ATP/NADH
- **Citrate Synthase**: Inhibited by ATP/NADH/succinyl-CoA
- **Pyruvate Carboxylase**: Activated by Acetyl-CoA
- **G6PDH**: Inhibited by NADPH (X-linked; G6PD deficiency)

### Emergent Behaviors
| Condition | Behavior |
|-----------|----------|
| Low O₂ | LDH/fermentation ↑, OxPhos ↓ (Pasteur Effect) |
| Low glucose | Gluconeogenesis ↑, PPP ↓ |
| High enzyme activity | All fluxes ↑ proportionally |
| T > 40°C | Enzyme denaturation (Arrhenius decay) |
| Very low O₂ + high flux | Warburg Effect detected |

---

## 🔧 Extending the System

### Add Real Training Data
Replace synthetic labels in `kegg_data.py:generate_training_samples()` with:
- BIGG Models flux data (JSON format)
- COBRA toolbox FBA solutions
- Experimental metabolomics datasets

### Add More Pathways
1. Add metabolites to `METABOLITES` dict in `kegg_data.py`
2. Add reactions to `REACTIONS` list
3. Add positions to `LAYOUT` dict
4. Add pathway color to `PATHWAY_COLORS`

### Upgrade GNN
- Add `torch_scatter` for proper message aggregation
- Implement `GraphSAGE` or `MPNN` for better inductive generalization
- Add uncertainty quantification (MC-Dropout or ensemble)
- Integrate real COBRA/FBA solutions as training targets

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `torch` | Neural network engine |
| `torch-geometric` | Graph neural network layers |
| `numpy/scipy` | Numerical computing |
| `httpx` | Async HTTP (KEGG API) |
| `d3.js` (CDN) | Interactive visualization |

---

## 🎯 Portfolio Value

This project demonstrates:
- **Systems biology** knowledge (real pathway biochemistry)
- **Graph Neural Networks** (GATv2, FiLM, bipartite graphs)
- **Scientific software** architecture (FBA + ML hybrid)
- **Full-stack** development (FastAPI + D3.js)
- **Data visualization** (Escher-style metabolic maps)
- **API design** (RESTful, async, well-documented)
