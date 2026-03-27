import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
"""
main.py — FastAPI backend for Cell Digital Twin
Endpoints:
  GET  /api/pathways          → full graph structure (nodes + edges) for all 4 pathways
  POST /api/simulate          → FBA heuristic flux simulation
  POST /api/simulate/gnn      → GNN-based flux prediction
  GET  /api/kegg/{pathway_id} → live KEGG REST data
  GET  /api/reactions         → all reaction metadata with regulation info
  GET  /api/metabolites       → all metabolite metadata
  GET  /api/atp_yield         → ATP accounting per condition
  POST /api/train             → (re)train GNN model
  GET  /api/model_status      → GNN training status
"""
from kegg_data import METABOLITES, REACTIONS, LAYOUT, RXN_IDS
import asyncio, time, math, os
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import gnn_model as gnn


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Warming up GNN model...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, gnn.get_model)
    print("[Startup] Ready.")
    yield

app = FastAPI(title="Cell Digital Twin API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimRequest(BaseModel):
    glucose: float = Field(8.0, ge=0, le=20, description="Glucose concentration mM")
    oxygen:  float = Field(0.21, ge=0, le=1, description="O2 fraction (0-1)")
    enzyme_activity: float = Field(1.0, ge=0.1, le=5.0, description="Global enzyme multiplier")
    temperature: float = Field(37.0, ge=20, le=45, description="Temperature °C")
    model: str = Field("fba", description="'fba' or 'gnn'")

class TrainRequest(BaseModel):
    n_samples: int = Field(1500, ge=100, le=5000)


def build_graph_payload(fluxes: dict | None = None):
    """Build nodes + edges JSON for the frontend Escher-style map."""
    nodes = []
    for mid, m in METABOLITES.items():
        pos = LAYOUT.get(mid, (500, 400))
        f = 0.0
        connected_rxns = [r for r in REACTIONS if mid in r["substrates"] or mid in r["products"]]
        if fluxes and connected_rxns:
            f = max((fluxes.get(r["id"], 0) for r in connected_rxns), default=0)
        nodes.append({
            "id": mid,
            "name": m["name"],
            "formula": m["formula"],
            "kegg": m["kegg"],
            "pathway": m["pathway"],
            "compartment": m["comp"],
            "charge": m["charge"],
            "x": pos[0],
            "y": pos[1],
            "flux": f,
        })

    edges = []
    for rxn in REACTIONS:
        flux_val = fluxes.get(rxn["id"], 0) if fluxes else 0

        src_pos_avg = _avg_pos([LAYOUT.get(s, (500,400)) for s in rxn["substrates"]])
        dst_pos_avg = _avg_pos([LAYOUT.get(p, (500,400)) for p in rxn["products"]])

        for s in rxn["substrates"]:
            sp = LAYOUT.get(s, (500,400))
            edges.append({
                "id": f"{rxn['id']}_s_{s}",
                "rxn_id": rxn["id"],
                "rxn_name": rxn["name"],
                "source": s,
                "target": f"rxn_{rxn['id']}",
                "type": "substrate",
                "pathway": rxn["pathway"],
                "flux": flux_val,
                "delta_g": rxn["delta_g"],
                "reversible": rxn["reversible"],
                "regulation": rxn["regulation"],
                "atp_yield": rxn["atp_yield"],
                "enzyme": rxn["enzyme"],
                "ec": rxn["ec"],
            })
        for p in rxn["products"]:
            edges.append({
                "id": f"{rxn['id']}_p_{p}",
                "rxn_id": rxn["id"],
                "rxn_name": rxn["name"],
                "source": f"rxn_{rxn['id']}",
                "target": p,
                "type": "product",
                "pathway": rxn["pathway"],
                "flux": flux_val,
                "delta_g": rxn["delta_g"],
                "reversible": rxn["reversible"],
                "regulation": rxn["regulation"],
                "atp_yield": rxn["atp_yield"],
                "enzyme": rxn["enzyme"],
                "ec": rxn["ec"],
            })

        rmid_x = (src_pos_avg[0] + dst_pos_avg[0]) / 2
        rmid_y = (src_pos_avg[1] + dst_pos_avg[1]) / 2
        nodes.append({
            "id": f"rxn_{rxn['id']}",
            "name": rxn["name"],
            "formula": rxn["ec"],
            "kegg": rxn["kegg_rxn"],
            "pathway": rxn["pathway"],
            "compartment": "reaction",
            "charge": 0,
            "x": rmid_x,
            "y": rmid_y,
            "flux": flux_val,
            "is_reaction": True,
            "regulation": rxn["regulation"],
            "delta_g": rxn["delta_g"],
            "reversible": rxn["reversible"],
            "atp_yield": rxn["atp_yield"],
            "enzyme": rxn["enzyme"],
            "ec": rxn["ec"],
        })

    return {"nodes": nodes, "edges": edges}

def _avg_pos(positions):
    if not positions: return (500, 400)
    return (sum(p[0] for p in positions)/len(positions),
            sum(p[1] for p in positions)/len(positions))

def atp_accounting(fluxes: dict) -> dict:
    """Compute per-pathway ATP yield."""
    glyco_atp  = fluxes.get("PGK",0)*2 + fluxes.get("PK",0)*2 - fluxes.get("HK",0) - fluxes.get("PFK",0)
    tca_atp    = fluxes.get("SUCL",0)  
    oxphos_atp = fluxes.get("OXPHOS",0)
    total      = max(0, glyco_atp + tca_atp + oxphos_atp)
    return {
        "glycolysis": round(max(0,glyco_atp),3),
        "tca":        round(max(0,tca_atp),3),
        "oxphos":     round(max(0,oxphos_atp),3),
        "total":      round(total, 3),
    }

def make_interpretation(result: dict, glucose: float, oxygen: float) -> list[str]:
    flags = []
    if oxygen < 0.02:   flags.append("🔴 Severe hypoxia — anaerobic crisis; LDH dominant")
    elif oxygen < 0.10: flags.append("🟡 Hypoxic shift — Warburg-like; lactate accumulating")
    else:               flags.append("🟢 Aerobic — TCA + OxPhos dominating energy production")
    if glucose < 0.5:   flags.append("⚠ Glucose starvation — gluconeogenesis maximal")
    elif glucose > 15:  flags.append("⬆ Hyperglycemia — glycolysis near maximum Vmax")
    if result.get("warburg") == "HIGH": flags.append("⚡ Warburg Effect detected (aerobic glycolysis)")
    if result.get("atp", 0) < 3:       flags.append("❗ ATP critical — energetic collapse risk")
    elif result.get("atp", 0) > 25:    flags.append("✅ ATP surplus — anabolic programs likely active")
    if result.get("nadph", 0) > 5:     flags.append("🧬 High NADPH — biosynthesis + antioxidant capacity up")
    return flags

@app.get("/api/health")
def health():
    return {"status": "ok", "torch": gnn.TORCH_AVAILABLE, "torch_geometric": gnn.TG_AVAILABLE,
            "gnn_trained": gnn._trained, "n_reactions": len(REACTIONS),
            "n_metabolites": len(METABOLITES)}

@app.get("/api/pathways")
def get_pathways():
    """Return full graph (no fluxes) for initial render."""
    return build_graph_payload(fluxes=None)

@app.get("/api/reactions")
def get_reactions():
    return {"reactions": REACTIONS}

@app.get("/api/metabolites")
def get_metabolites():
    return {"metabolites": METABOLITES}

@app.post("/api/simulate")
def simulate(req: SimRequest):
    """FBA heuristic simulation."""
    result = gnn.heuristic_predict(req.glucose, req.oxygen, req.enzyme_activity, req.temperature)
    graph = build_graph_payload(result["fluxes"])
    atp   = atp_accounting(result["fluxes"])
    interp = make_interpretation(result, req.glucose, req.oxygen)
    return {
        "fluxes": result["fluxes"],
        "metabolites": {
            "atp": result["atp"], "pyruvate": result["pyruvate"],
            "lactate": result["lactate"], "nadh": result["nadh"],
            "fadh2": result["fadh2"], "citrate": result["citrate"],
            "r5p": result["r5p"], "nadph": result.get("nadph", 0),
        },
        "atp_accounting": atp,
        "warburg": result["warburg"],
        "graph": graph,
        "interpretation": interp,
        "model_used": "FBA-heuristic",
    }

@app.post("/api/simulate/gnn")
def simulate_gnn(req: SimRequest):
    """GNN-based flux prediction."""
    result = gnn.predict(req.glucose, req.oxygen, req.enzyme_activity, req.temperature)
    graph  = build_graph_payload(result["fluxes"])
    atp    = atp_accounting(result["fluxes"])
    interp = make_interpretation(result, req.glucose, req.oxygen)
    return {
        "fluxes": result["fluxes"],
        "metabolites": {
            "atp": result["atp"], "pyruvate": result["pyruvate"],
            "lactate": result["lactate"], "nadh": result["nadh"],
            "fadh2": result["fadh2"], "citrate": result["citrate"],
            "r5p": result["r5p"], "nadph": result.get("nadph", 0),
        },
        "atp_accounting": atp,
        "warburg": result["warburg"],
        "graph": graph,
        "interpretation": interp,
        "model_used": result.get("model","GNN"),
    }

@app.get("/api/kegg/{pathway_id}")
async def get_kegg(pathway_id: str):
    """Fetch live KEGG REST data for a pathway."""
    import httpx
    valid = {"hsa00010","hsa00020","hsa00030","hsa00620","hsa00010"}
    if pathway_id not in valid:
        raise HTTPException(400, f"Pathway must be one of {valid}")
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(f"https://rest.kegg.jp/get/{pathway_id}")
            if r.status_code == 200:
                lines = r.text.split("\n")
                desc = next((l for l in lines if l.startswith("NAME")), "")
                return {"pathway_id": pathway_id, "raw": r.text[:3000],
                        "description": desc, "source": "KEGG REST API"}
    except Exception as e:
        pass
    fallback = {
        "hsa00010": "Glycolysis / Gluconeogenesis — converts glucose to pyruvate (glycolysis) or glucose from non-carbohydrate substrates (gluconeogenesis). 10 enzymatic steps in glycolysis, 11 in gluconeogenesis (shares 7 reversible reactions).",
        "hsa00020": "TCA Cycle (Citric Acid / Krebs Cycle) — central hub of aerobic metabolism. 8 reactions in mitochondria, generates NADH, FADH2, GTP, CO2 per turn.",
        "hsa00030": "Pentose Phosphate Pathway — generates NADPH (antioxidant/biosynthesis) and ribose-5-P (nucleotide synthesis). Oxidative + non-oxidative phases.",
    }
    return {"pathway_id": pathway_id, "raw": fallback.get(pathway_id,"No data"),
            "description": fallback.get(pathway_id,""), "source": "embedded"}

@app.get("/api/atp_yield")
def get_atp_yield(glucose: float = 8.0, oxygen: float = 0.21,
                  enzyme: float = 1.0, temp: float = 37.0):
    result = gnn.heuristic_predict(glucose, oxygen, enzyme, temp)
    return atp_accounting(result["fluxes"])

@app.post("/api/train")
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    if not gnn.TORCH_AVAILABLE or not gnn.TG_AVAILABLE:
        raise HTTPException(503, "PyTorch/PyG not available in this environment")
    background_tasks.add_task(gnn._train_model)
    return {"message": "Training started in background", "n_samples": req.n_samples}

@app.get("/api/model_status")
def model_status():
    return {
        "torch_available": gnn.TORCH_AVAILABLE,
        "torch_geometric_available": gnn.TG_AVAILABLE,
        "gnn_trained": gnn._trained,
        "model_path_exists": os.path.exists(gnn.MODEL_PATH),
        "n_reactions": len(REACTIONS),
        "n_metabolites": len(METABOLITES),
        "n_nodes_in_graph": gnn.N_NODES,
        "n_edges_in_graph": len(gnn.SRC),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
