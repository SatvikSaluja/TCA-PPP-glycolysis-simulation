"""
gnn_model.py — PyTorch Geometric GNN for metabolic flux prediction.

Architecture:
  - Node features: metabolite properties (concentration proxy, charge, compartment)
  - Edge features: reaction stoichiometry, reversibility, ΔG
  - Message passing: 3-layer GATv2Conv (Graph Attention Network v2)
  - Output: flux per reaction edge (regression)
  - Environmental context injected via FiLM conditioning (Feature-wise Linear Modulation)

This implements a proper graph neural network where:
  - Metabolites ARE nodes
  - Reactions ARE hyperedges (split into substrate→rxn and rxn→product node pairs)
  - Global environmental state [glucose, O2, enzyme, temp] modulates all layers
"""

import math, os, json
import numpy as np

# ── Try importing torch; fall back to heuristic if unavailable ────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    try:
        from torch_geometric.nn import GATv2Conv, global_mean_pool
        from torch_geometric.data import Data
        TG_AVAILABLE = True
    except ImportError:
        TG_AVAILABLE = False
except ImportError:
    TORCH_AVAILABLE = False
    TG_AVAILABLE = False

from kegg_data import METABOLITES, REACTIONS, generate_training_samples

# ── Graph construction ─────────────────────────────────────────────────────────
PATHWAY_IDX = {
    "glycolysis":0,"tca":1,"ppp":2,"gluconeo":3,
    "fermentation":4,"oxphos":5,"energy":6,"redox":7
}
COMP_IDX = {"cytosol":0,"mitochondria":1}
MET_IDS  = list(METABOLITES.keys())
RXN_IDS  = [r["id"] for r in REACTIONS]
N_METS   = len(MET_IDS)
N_RXNS   = len(REACTIONS)

def met_idx(mid: str) -> int:
    return MET_IDS.index(mid) if mid in MET_IDS else 0

def build_graph_structure():
    """
    Build the bipartite metabolite–reaction graph.
    We add N_RXNS virtual reaction nodes after the metabolite nodes.
    Node indices: 0..N_METS-1 = metabolites, N_METS..N_METS+N_RXNS-1 = reactions
    """
    src, dst, edge_attr = [], [], []
    for i, rxn in enumerate(REACTIONS):
        rxn_node = N_METS + i
        # Substrate edges: metabolite → reaction
        for s in rxn["substrates"]:
            m = met_idx(s)
            src.append(m); dst.append(rxn_node)
            edge_attr.append([1.0, rxn["delta_g"]/100.0, float(rxn["reversible"]), 0.0])
        # Product edges: reaction → metabolite
        for p in rxn["products"]:
            m = met_idx(p)
            src.append(rxn_node); dst.append(m)
            edge_attr.append([-1.0, rxn["delta_g"]/100.0, float(rxn["reversible"]), 1.0])

    return src, dst, edge_attr

SRC, DST, EDGE_ATTR = build_graph_structure()
N_NODES = N_METS + N_RXNS

def make_node_features(glucose: float = 8.0, oxygen: float = 0.21,
                        enzyme: float = 1.0, temp: float = 37.0):
    """Per-node feature vector [5 dims]:
       pathway_onehot(8) + compartment(2) + charge_norm + is_rxn_node
    """
    feats = []
    for mid in MET_IDS:
        m = METABOLITES[mid]
        pw  = PATHWAY_IDX.get(m["pathway"], 0)
        cmp = COMP_IDX.get(m["comp"], 0)
        pw_oh = [0.0]*8; pw_oh[pw] = 1.0
        cm_oh = [0.0]*2; cm_oh[cmp] = 1.0
        ch = m["charge"] / 5.0
        feats.append(pw_oh + cm_oh + [ch, 0.0])  # 11 dims, is_rxn=0

    for rxn in REACTIONS:
        pw  = PATHWAY_IDX.get(rxn["pathway"], 0)
        pw_oh = [0.0]*8; pw_oh[pw] = 1.0
        cm_oh = [0.0, 0.0]  # reactions span compartments
        dg_norm = rxn["delta_g"] / 250.0
        feats.append(pw_oh + cm_oh + [dg_norm, 1.0])  # is_rxn=1

    return feats  # shape: [N_NODES, 11]


if TORCH_AVAILABLE and TG_AVAILABLE:
    class FiLMLayer(nn.Module):
        """Feature-wise Linear Modulation: scale/shift node features by env context."""
        def __init__(self, feat_dim, env_dim=4):
            super().__init__()
            self.gamma = nn.Linear(env_dim, feat_dim)
            self.beta  = nn.Linear(env_dim, feat_dim)

        def forward(self, x, env):
            g = self.gamma(env).unsqueeze(1)  # [B,1,F]
            b = self.beta(env).unsqueeze(1)
            return g * x + b

    class MetabolicGNN(nn.Module):
        """
        3-layer GATv2 with:
          - FiLM conditioning from environmental context
          - Skip connections
          - Separate readout MLP for reaction flux prediction
        """
        def __init__(self, hidden=64, heads=4, env_dim=4, n_rxns=N_RXNS):
            super().__init__()
            sample = make_node_features()
            in_dim = len(sample[0])
            self.embed = nn.Linear(in_dim, hidden)

            self.conv1 = GATv2Conv(hidden, hidden//heads, heads=heads,
                                   edge_dim=4, concat=True, dropout=0.1)
            self.conv2 = GATv2Conv(hidden, hidden//heads, heads=heads,
                                   edge_dim=4, concat=True, dropout=0.1)
            self.conv3 = GATv2Conv(hidden, hidden//heads, heads=heads,
                                   edge_dim=4, concat=True, dropout=0.1)

            self.film1 = FiLMLayer(hidden, env_dim)
            self.film2 = FiLMLayer(hidden, env_dim)
            self.film3 = FiLMLayer(hidden, env_dim)

            self.norm1 = nn.LayerNorm(hidden)
            self.norm2 = nn.LayerNorm(hidden)
            self.norm3 = nn.LayerNorm(hidden)

            # Env encoder
            self.env_enc = nn.Sequential(
                nn.Linear(env_dim, 32), nn.SiLU(),
                nn.Linear(32, env_dim)
            )

            # Reaction flux readout: takes reaction node embedding + env
            self.readout = nn.Sequential(
                nn.Linear(hidden + env_dim, 64), nn.SiLU(),
                nn.Linear(64, 32), nn.SiLU(),
                nn.Linear(32, 1)
            )
            self.n_rxns = n_rxns
            self.n_mets = N_METS

        def forward(self, data):
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            env = data.env  # [B, 4] or [4]
            if env.dim() == 1: env = env.unsqueeze(0)

            x = self.embed(x)  # [N, H]

            # Layer 1
            h = self.conv1(x, edge_index, edge_attr)
            h = self.film1(h.unsqueeze(0), env).squeeze(0)
            h = self.norm1(h + x)
            h = F.silu(h)

            # Layer 2
            h2 = self.conv2(h, edge_index, edge_attr)
            h2 = self.film2(h2.unsqueeze(0), env).squeeze(0)
            h2 = self.norm2(h2 + h)
            h2 = F.silu(h2)

            # Layer 3
            h3 = self.conv3(h2, edge_index, edge_attr)
            h3 = self.film3(h3.unsqueeze(0), env).squeeze(0)
            h3 = self.norm3(h3 + h2)

            # Extract reaction node embeddings [N_RXNS, H]
            rxn_emb = h3[self.n_mets: self.n_mets + self.n_rxns]

            # Concatenate env context to each reaction node
            env_exp = env.expand(self.n_rxns, -1)
            inp = torch.cat([rxn_emb, env_exp], dim=-1)

            flux = self.readout(inp).squeeze(-1)  # [N_RXNS]
            flux = F.softplus(flux)  # non-negative fluxes
            return flux


    def build_data_object(glucose: float, oxygen: float, enzyme: float, temp: float) -> "Data":
        """Build a PyG Data object for a given environmental condition."""
        node_feats = make_node_features(glucose, oxygen, enzyme, temp)
        x = torch.tensor(node_feats, dtype=torch.float)
        edge_index = torch.tensor([SRC, DST], dtype=torch.long)
        edge_attr  = torch.tensor(EDGE_ATTR, dtype=torch.float)
        env = torch.tensor(
            [glucose/20.0, oxygen, enzyme/3.0, (temp-20)/25.0],
            dtype=torch.float
        )
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, env=env)


# ── Heuristic fallback (no PyTorch) ───────────────────────────────────────────
def heuristic_predict(glucose: float, oxygen: float, enzyme: float, temp: float) -> dict:
    """Biologically-constrained analytic flux calculation (FBA-like)."""
    import math
    g, o, e, t = glucose, oxygen, enzyme, temp
    tf = math.exp(-0.04 * (t - 37)**2)

    glc_flux = e * tf * 9.0 * g / (g + 1.5)
    pdk_flux  = glc_flux * 0.85 * (o / (o + 0.05))
    tca_base  = pdk_flux * 0.90
    ppp_share = 0.12 if g > 2 else 0.06
    ppp_flux  = glc_flux * ppp_share * e * tf
    ferm_flux = max(0, glc_flux * 0.7 * max(0, 1 - o/0.21))
    oxphos    = tca_base * 3.5 * (o / (o + 0.02)) * tf
    gng_act   = max(0, (1.5 - g) / 1.5)
    gng_flux  = gng_act * e * tf * 3.0

    fluxes = {
        "HK": glc_flux*0.5,  "PGI": glc_flux*0.8,  "PFK": glc_flux*0.85,
        "ALD": glc_flux*0.85,"TPI": glc_flux*0.85,  "GAPDH": glc_flux*0.9,
        "PGK": glc_flux*0.9, "PGM": glc_flux*0.9,   "ENO": glc_flux*0.9,
        "PK":  glc_flux,     "LDH": ferm_flux,
        "PDH": pdk_flux, "CS": tca_base, "ACO": tca_base*0.95,
        "IDH": tca_base*0.92,"OGDH": tca_base*0.88,"SUCL": tca_base*0.85,
        "SDH": tca_base*0.82,"FUM": tca_base*0.80,  "MDH": tca_base*0.78,
        "OXPHOS": oxphos,
        "G6PDH": ppp_flux,   "LAC": ppp_flux*0.9,   "PGD": ppp_flux*0.85,
        "RPI": ppp_flux*0.7, "RPE": ppp_flux*0.6,   "TK1": ppp_flux*0.5,
        "TA":  ppp_flux*0.4, "TK2": ppp_flux*0.4,
        "PC": gng_flux*0.8,  "PEPCK": gng_flux*0.8, "FBPase": gng_flux*0.7,
        "G6Pase": gng_flux*0.65,"ME": gng_flux*0.5,
    }

    atp_total = (glc_flux*2 + tca_base*2 + oxphos)
    return {"fluxes": fluxes, "atp": atp_total,
            "pyruvate": glc_flux*0.5, "lactate": ferm_flux,
            "nadh": tca_base*1.5, "fadh2": tca_base*0.8,
            "citrate": tca_base, "r5p": ppp_flux*0.7,
            "nadph": ppp_flux*1.8, "warburg": "HIGH" if ferm_flux > glc_flux*0.3 else "LOW"}


# ── Model singleton ────────────────────────────────────────────────────────────
_model = None
_trained = False
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "gnn_weights.pt")

def get_model():
    global _model, _trained
    if not TORCH_AVAILABLE or not TG_AVAILABLE:
        return None
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                " Model not trained. Run `_train_model()` once before starting the server."
            )
        _model = MetabolicGNN()
        _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _model.eval()
        _trained = True
    return _model

def _train_model():
    global _model, _trained
    if not TORCH_AVAILABLE or not TG_AVAILABLE:
        return
    from torch_geometric.data import Data
    import torch.optim as optim

    print("[GNN] Generating training data...")
    inputs, labels = generate_training_samples(1500)

    model = MetabolicGNN()
    opt = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)

    # Build dataset
    dataset = []
    for inp, lab in zip(inputs, labels):
        data = build_data_object(*inp)
        data.y = torch.tensor(lab, dtype=torch.float)
        dataset.append(data)

    model.train()
    for epoch in range(100):
        total_loss = 0
        for data in dataset:
            opt.zero_grad()
            pred = model(data)
            loss = F.mse_loss(pred, data.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}/100, loss={total_loss/len(dataset):.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    _model = model
    _trained = True
    print("[GNN] Training complete, weights saved.")


def predict(glucose: float, oxygen: float, enzyme: float, temp: float) -> dict:
    """Main prediction entry point — uses GNN if available, else heuristic."""
    if TORCH_AVAILABLE and TG_AVAILABLE:
        model = get_model()
        if model is not None:
            model.eval()
            with torch.no_grad():
                data = build_data_object(glucose, oxygen, enzyme, temp)
                flux_tensor = model(data).numpy()
            fluxes = {rxn["id"]: float(flux_tensor[i]) for i, rxn in enumerate(REACTIONS)}
            atp = fluxes.get("PGK",0)*2 + fluxes.get("PK",0)*2 + fluxes.get("OXPHOS",0)
            return {
                "fluxes": fluxes,
                "atp": atp,
                "pyruvate": fluxes.get("PK",0)*0.5,
                "lactate": fluxes.get("LDH",0),
                "nadh": fluxes.get("MDH",0)*1.5,
                "fadh2": fluxes.get("SDH",0)*0.85,
                "citrate": fluxes.get("CS",0),
                "r5p": fluxes.get("RPI",0)*0.8,
                "nadph": fluxes.get("G6PDH",0)*1.8,
                "warburg": "HIGH" if fluxes.get("LDH",0)>fluxes.get("PK",0)*0.3 else "LOW",
                "model": "GNN",
            }

    result = heuristic_predict(glucose, oxygen, enzyme, temp)
    result["model"] = "heuristic"
    return result
