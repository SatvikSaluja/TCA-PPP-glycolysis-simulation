"""
kegg_data.py — Real KEGG metabolic data for the 4 major pathways.

Since KEGG REST API may be rate-limited or unavailable in some environments,
we embed the canonical graph data derived from KEGG pathway maps:
  - hsa00010: Glycolysis / Gluconeogenesis
  - hsa00020: TCA Cycle (Citrate Cycle)
  - hsa00030: Pentose Phosphate Pathway
  plus Gluconeogenesis (reverse/shared with glycolysis, hsa00010)

All data structures are based on real KEGG KGML / BiGG metabolite IDs.
"""

from typing import Dict, List, Tuple

# ── Metabolites (nodes) ────────────────────────────────────────────────────────
# Each metabolite: {id, name, formula, kegg_id, pathway, compartment}
METABOLITES: Dict[str, dict] = {
    # Glycolysis / Gluconeogenesis
    "glc":  {"name":"Glucose",           "formula":"C6H12O6","kegg":"C00031","pathway":"glycolysis","comp":"cytosol","charge":0},
    "g6p":  {"name":"Glucose-6-P",       "formula":"C6H13O9P","kegg":"C00668","pathway":"glycolysis","comp":"cytosol","charge":-2},
    "f6p":  {"name":"Fructose-6-P",      "formula":"C6H13O9P","kegg":"C00085","pathway":"glycolysis","comp":"cytosol","charge":-2},
    "f16bp":{"name":"Fructose-1,6-BP",   "formula":"C6H14O12P2","kegg":"C00354","pathway":"glycolysis","comp":"cytosol","charge":-4},
    "dhap": {"name":"DHAP",              "formula":"C3H7O6P","kegg":"C00111","pathway":"glycolysis","comp":"cytosol","charge":-2},
    "gap":  {"name":"G3P",               "formula":"C3H7O7P","kegg":"C00118","pathway":"glycolysis","comp":"cytosol","charge":-2},
    "bpg":  {"name":"1,3-BPG",           "formula":"C3H8O10P2","kegg":"C00236","pathway":"glycolysis","comp":"cytosol","charge":-4},
    "pg3":  {"name":"3-PG",              "formula":"C3H7O7P","kegg":"C00197","pathway":"glycolysis","comp":"cytosol","charge":-3},
    "pg2":  {"name":"2-PG",              "formula":"C3H7O7P","kegg":"C00631","pathway":"glycolysis","comp":"cytosol","charge":-3},
    "pep":  {"name":"PEP",               "formula":"C3H5O6P","kegg":"C00074","pathway":"glycolysis","comp":"cytosol","charge":-3},
    "pyr":  {"name":"Pyruvate",          "formula":"C3H4O3","kegg":"C00022","pathway":"glycolysis","comp":"cytosol","charge":-1},
    "lac":  {"name":"Lactate",           "formula":"C3H6O3","kegg":"C00186","pathway":"fermentation","comp":"cytosol","charge":-1},
    "atp":  {"name":"ATP",               "formula":"C10H16N5O13P3","kegg":"C00002","pathway":"energy","comp":"cytosol","charge":-4},
    "adp":  {"name":"ADP",               "formula":"C10H15N5O10P2","kegg":"C00008","pathway":"energy","comp":"cytosol","charge":-3},
    "nad":  {"name":"NAD⁺",              "formula":"C21H28N7O14P2","kegg":"C00003","pathway":"redox","comp":"cytosol","charge":-1},
    "nadh": {"name":"NADH",              "formula":"C21H29N7O14P2","kegg":"C00004","pathway":"redox","comp":"cytosol","charge":-2},
    "pi":   {"name":"Phosphate",         "formula":"HO4P","kegg":"C00009","pathway":"energy","comp":"cytosol","charge":-2},

    # TCA Cycle (mitochondria)
    "accoa":{"name":"Acetyl-CoA",        "formula":"C23H38N7O17P3S","kegg":"C00024","pathway":"tca","comp":"mitochondria","charge":-4},
    "oaa":  {"name":"Oxaloacetate",      "formula":"C4H4O5","kegg":"C00036","pathway":"tca","comp":"mitochondria","charge":-2},
    "cit":  {"name":"Citrate",           "formula":"C6H8O7","kegg":"C00158","pathway":"tca","comp":"mitochondria","charge":-3},
    "icit": {"name":"Isocitrate",        "formula":"C6H8O7","kegg":"C00311","pathway":"tca","comp":"mitochondria","charge":-3},
    "akg":  {"name":"α-Ketoglutarate",   "formula":"C5H6O5","kegg":"C00026","pathway":"tca","comp":"mitochondria","charge":-2},
    "scoa": {"name":"Succinyl-CoA",      "formula":"C25H40N7O19P3S","kegg":"C00091","pathway":"tca","comp":"mitochondria","charge":-5},
    "suc":  {"name":"Succinate",         "formula":"C4H6O4","kegg":"C00042","pathway":"tca","comp":"mitochondria","charge":-2},
    "fum":  {"name":"Fumarate",          "formula":"C4H4O4","kegg":"C00122","pathway":"tca","comp":"mitochondria","charge":-2},
    "mal":  {"name":"Malate",            "formula":"C4H6O5","kegg":"C00149","pathway":"tca","comp":"mitochondria","charge":-2},
    "nadm": {"name":"NAD⁺ (mito)",       "formula":"C21H28N7O14P2","kegg":"C00003","pathway":"redox","comp":"mitochondria","charge":-1},
    "nadhm":{"name":"NADH (mito)",       "formula":"C21H29N7O14P2","kegg":"C00004","pathway":"redox","comp":"mitochondria","charge":-2},
    "fadh2":{"name":"FADH₂",             "formula":"C27H33N9O15P2","kegg":"C01352","pathway":"redox","comp":"mitochondria","charge":-2},
    "co2":  {"name":"CO₂",              "formula":"CO2","kegg":"C00011","pathway":"tca","comp":"mitochondria","charge":0},
    "h2o":  {"name":"H₂O",              "formula":"H2O","kegg":"C00001","pathway":"tca","comp":"mitochondria","charge":0},
    "gtp":  {"name":"GTP",              "formula":"C10H16N5O14P3","kegg":"C00044","pathway":"energy","comp":"mitochondria","charge":-4},
    "gdp":  {"name":"GDP",              "formula":"C10H15N5O11P2","kegg":"C00035","pathway":"energy","comp":"mitochondria","charge":-3},
    "coa":  {"name":"CoA",              "formula":"C21H36N7O16P3S","kegg":"C00010","pathway":"tca","comp":"mitochondria","charge":-4},

    # Pentose Phosphate Pathway
    "g6p_ppp":{"name":"G6P (PPP entry)","formula":"C6H13O9P","kegg":"C00668","pathway":"ppp","comp":"cytosol","charge":-2},
    "p6gl": {"name":"6-P-Gluconolact.","formula":"C6H11O9P","kegg":"C01236","pathway":"ppp","comp":"cytosol","charge":-2},
    "p6g":  {"name":"6-P-Gluconate",   "formula":"C6H13O10P","kegg":"C00345","pathway":"ppp","comp":"cytosol","charge":-3},
    "ru5p": {"name":"Ribulose-5-P",    "formula":"C5H11O8P","kegg":"C00199","pathway":"ppp","comp":"cytosol","charge":-2},
    "r5p":  {"name":"Ribose-5-P",      "formula":"C5H11O8P","kegg":"C00117","pathway":"ppp","comp":"cytosol","charge":-2},
    "x5p":  {"name":"Xylulose-5-P",    "formula":"C5H11O8P","kegg":"C00231","pathway":"ppp","comp":"cytosol","charge":-2},
    "s7p":  {"name":"Sedoheptulose-7P","formula":"C7H15O10P","kegg":"C05382","pathway":"ppp","comp":"cytosol","charge":-2},
    "e4p":  {"name":"Erythrose-4-P",   "formula":"C4H9O7P","kegg":"C00279","pathway":"ppp","comp":"cytosol","charge":-2},
    "nadph":{"name":"NADPH",           "formula":"C21H30N7O17P3","kegg":"C00005","pathway":"redox","comp":"cytosol","charge":-4},
    "nadp": {"name":"NADP⁺",           "formula":"C21H29N7O17P3","kegg":"C00006","pathway":"redox","comp":"cytosol","charge":-3},

    # Gluconeogenesis specific
    "oxal": {"name":"Oxaloacetate (GNG)","formula":"C4H4O5","kegg":"C00036","pathway":"gluconeo","comp":"cytosol","charge":-2},
    "mal_c":{"name":"Malate (cyto)",    "formula":"C4H6O5","kegg":"C00149","pathway":"gluconeo","comp":"cytosol","charge":-2},

    # OxPhos
    "o2":   {"name":"O₂",              "formula":"O2","kegg":"C00007","pathway":"oxphos","comp":"mitochondria","charge":0},
}

# ── Reactions (edges) ──────────────────────────────────────────────────────────
# Each reaction: {id, name, enzyme, ec, kegg_rxn, substrates, products,
#                 pathway, reversible, delta_g (kJ/mol), stoich_sub, stoich_prod}
REACTIONS: List[dict] = [
    # ── GLYCOLYSIS ────────────────────────────────────────────────
    {
        "id":"HK","name":"Hexokinase","enzyme":"HK1/HK2","ec":"2.7.1.1",
        "kegg_rxn":"R00299","substrates":["glc","atp"],"products":["g6p","adp"],
        "pathway":"glycolysis","reversible":False,"delta_g":-16.7,
        "stoich_sub":[1,1],"stoich_prod":[1,1],
        "regulation":"inhibited by G6P (product inhibition); activated by insulin",
        "atp_yield":-1,
    },
    {
        "id":"PGI","name":"Phosphoglucose Isomerase","enzyme":"GPI","ec":"5.3.1.9",
        "kegg_rxn":"R00771","substrates":["g6p"],"products":["f6p"],
        "pathway":"glycolysis","reversible":True,"delta_g":-2.9,
        "stoich_sub":[1],"stoich_prod":[1],
        "regulation":"near-equilibrium; inhibited by erythrose-4-P",
        "atp_yield":0,
    },
    {
        "id":"PFK","name":"Phosphofructokinase-1","enzyme":"PFKM","ec":"2.7.1.11",
        "kegg_rxn":"R00756","substrates":["f6p","atp"],"products":["f16bp","adp"],
        "pathway":"glycolysis","reversible":False,"delta_g":-14.2,
        "stoich_sub":[1,1],"stoich_prod":[1,1],
        "regulation":"ALLOSTERIC: inhibited by ATP/citrate; activated by AMP/ADP/F2,6-BP (KEY CONTROL POINT)",
        "atp_yield":-1,
    },
    {
        "id":"ALD","name":"Aldolase","enzyme":"ALDOA","ec":"4.1.2.13",
        "kegg_rxn":"R01068","substrates":["f16bp"],"products":["dhap","gap"],
        "pathway":"glycolysis","reversible":True,"delta_g":23.8,
        "stoich_sub":[1],"stoich_prod":[1,1],
        "regulation":"reversible; product removal drives forward",
        "atp_yield":0,
    },
    {
        "id":"TPI","name":"Triosephosphate Isomerase","enzyme":"TPI1","ec":"5.3.1.1",
        "kegg_rxn":"R01015","substrates":["dhap"],"products":["gap"],
        "pathway":"glycolysis","reversible":True,"delta_g":7.5,
        "stoich_sub":[1],"stoich_prod":[1],
        "regulation":"near-equilibrium",
        "atp_yield":0,
    },
    {
        "id":"GAPDH","name":"G3P Dehydrogenase","enzyme":"GAPDH","ec":"1.2.1.12",
        "kegg_rxn":"R01061","substrates":["gap","nad","pi"],"products":["bpg","nadh"],
        "pathway":"glycolysis","reversible":True,"delta_g":6.3,
        "stoich_sub":[1,1,1],"stoich_prod":[1,1],
        "regulation":"inhibited by high NADH/NAD ratio; arsenate uncouples",
        "atp_yield":0,
    },
    {
        "id":"PGK","name":"Phosphoglycerate Kinase","enzyme":"PGK1","ec":"2.7.2.3",
        "kegg_rxn":"R01512","substrates":["bpg","adp"],"products":["pg3","atp"],
        "pathway":"glycolysis","reversible":True,"delta_g":-18.9,
        "stoich_sub":[1,1],"stoich_prod":[1,1],
        "regulation":"substrate level phosphorylation × 2 (×2 G3P per glucose)",
        "atp_yield":2,
    },
    {
        "id":"PGM","name":"Phosphoglycerate Mutase","enzyme":"PGAM1","ec":"5.4.2.11",
        "kegg_rxn":"R01518","substrates":["pg3"],"products":["pg2"],
        "pathway":"glycolysis","reversible":True,"delta_g":4.4,
        "stoich_sub":[1],"stoich_prod":[1],
        "regulation":"near-equilibrium",
        "atp_yield":0,
    },
    {
        "id":"ENO","name":"Enolase","enzyme":"ENO1","ec":"4.2.1.11",
        "kegg_rxn":"R00658","substrates":["pg2"],"products":["pep","h2o"],
        "pathway":"glycolysis","reversible":True,"delta_g":1.8,
        "stoich_sub":[1],"stoich_prod":[1,1],
        "regulation":"inhibited by F2,6-BP",
        "atp_yield":0,
    },
    {
        "id":"PK","name":"Pyruvate Kinase","enzyme":"PKM2","ec":"2.7.1.40",
        "kegg_rxn":"R00200","substrates":["pep","adp"],"products":["pyr","atp"],
        "pathway":"glycolysis","reversible":False,"delta_g":-31.4,
        "stoich_sub":[1,1],"stoich_prod":[1,1],
        "regulation":"allosteric: activated by F1,6-BP; inhibited by ATP/Ala; PKM2 active in cancer",
        "atp_yield":2,
    },
    {
        "id":"LDH","name":"Lactate Dehydrogenase","enzyme":"LDHA","ec":"1.1.1.27",
        "kegg_rxn":"R00703","substrates":["pyr","nadh"],"products":["lac","nad"],
        "pathway":"fermentation","reversible":True,"delta_g":-25.1,
        "stoich_sub":[1,1],"stoich_prod":[1,1],
        "regulation":"favored under anaerobic conditions; regenerates NAD+",
        "atp_yield":0,
    },

    # ── TCA CYCLE ─────────────────────────────────────────────────
    {
        "id":"PDH","name":"Pyruvate Dehydrogenase","enzyme":"PDH complex","ec":"1.2.4.1",
        "kegg_rxn":"R00209","substrates":["pyr","nadm","coa"],"products":["accoa","nadhm","co2"],
        "pathway":"tca","reversible":False,"delta_g":-39.4,
        "stoich_sub":[1,1,1],"stoich_prod":[1,1,1],
        "regulation":"inhibited by Acetyl-CoA/NADH/ATP; activated by Ca2+, CoA, NAD+",
        "atp_yield":0,
    },
    {
        "id":"CS","name":"Citrate Synthase","enzyme":"CS","ec":"2.3.3.1",
        "kegg_rxn":"R00351","substrates":["accoa","oaa","h2o"],"products":["cit","coa"],
        "pathway":"tca","reversible":False,"delta_g":-31.4,
        "stoich_sub":[1,1,1],"stoich_prod":[1,1],
        "regulation":"inhibited by ATP/NADH/succinyl-CoA/citrate (KEY CONTROL POINT)",
        "atp_yield":0,
    },
    {
        "id":"ACO","name":"Aconitase","enzyme":"ACO2","ec":"4.2.1.3",
        "kegg_rxn":"R01325","substrates":["cit"],"products":["icit"],
        "pathway":"tca","reversible":True,"delta_g":6.3,
        "stoich_sub":[1],"stoich_prod":[1],
        "regulation":"inhibited by fluorocitrate; Fe-S cluster enzyme",
        "atp_yield":0,
    },
    {
        "id":"IDH","name":"Isocitrate Dehydrogenase","enzyme":"IDH2","ec":"1.1.1.41",
        "kegg_rxn":"R00709","substrates":["icit","nadm"],"products":["akg","nadhm","co2"],
        "pathway":"tca","reversible":False,"delta_g":-20.9,
        "stoich_sub":[1,1],"stoich_prod":[1,1,1],
        "regulation":"allosteric: activated by ADP/Ca2+; inhibited by ATP/NADH (KEY CONTROL POINT)",
        "atp_yield":0,
    },
    {
        "id":"OGDH","name":"α-KG Dehydrogenase","enzyme":"OGDH complex","ec":"1.2.4.2",
        "kegg_rxn":"R08549","substrates":["akg","nadm","coa"],"products":["scoa","nadhm","co2"],
        "pathway":"tca","reversible":False,"delta_g":-33.5,
        "stoich_sub":[1,1,1],"stoich_prod":[1,1,1],
        "regulation":"inhibited by succinyl-CoA/NADH/ATP; activated by Ca2+",
        "atp_yield":0,
    },
    {
        "id":"SUCL","name":"Succinyl-CoA Ligase","enzyme":"SUCLG1","ec":"6.2.1.4",
        "kegg_rxn":"R00405","substrates":["scoa","gdp","pi"],"products":["suc","gtp","coa"],
        "pathway":"tca","reversible":True,"delta_g":-2.9,
        "stoich_sub":[1,1,1],"stoich_prod":[1,1,1],
        "regulation":"substrate level phosphorylation (GTP)",
        "atp_yield":1,
    },
    {
        "id":"SDH","name":"Succinate Dehydrogenase","enzyme":"SDHA (Complex II)","ec":"1.3.5.1",
        "kegg_rxn":"R02164","substrates":["suc"],"products":["fum","fadh2"],
        "pathway":"tca","reversible":False,"delta_g":0.0,
        "stoich_sub":[1],"stoich_prod":[1,1],
        "regulation":"links TCA to ETC; inhibited by malonate (competitive)",
        "atp_yield":0,
    },
    {
        "id":"FUM","name":"Fumarase","enzyme":"FH","ec":"4.2.1.2",
        "kegg_rxn":"R01082","substrates":["fum","h2o"],"products":["mal"],
        "pathway":"tca","reversible":True,"delta_g":-3.8,
        "stoich_sub":[1,1],"stoich_prod":[1],
        "regulation":"near-equilibrium",
        "atp_yield":0,
    },
    {
        "id":"MDH","name":"Malate Dehydrogenase","enzyme":"MDH2","ec":"1.1.1.37",
        "kegg_rxn":"R00342","substrates":["mal","nadm"],"products":["oaa","nadhm"],
        "pathway":"tca","reversible":True,"delta_g":29.7,
        "stoich_sub":[1,1],"stoich_prod":[1,1],
        "regulation":"near-equilibrium; regenerates OAA for CS",
        "atp_yield":0,
    },
    {
        "id":"OXPHOS","name":"Oxidative Phosphorylation","enzyme":"ETC + ATP synthase","ec":"7.1.2.2",
        "kegg_rxn":"R02703","substrates":["nadhm","fadh2","o2","adp","pi"],"products":["atp","h2o","nadm"],
        "pathway":"oxphos","reversible":False,"delta_g":-220.0,
        "stoich_sub":[10,2,6,32,32],"stoich_prod":[32,12,10],
        "regulation":"inhibited by oligomycin/DCCD; uncoupled by DNP; regulated by membrane potential",
        "atp_yield":32,
    },

    # ── PENTOSE PHOSPHATE PATHWAY ──────────────────────────────────
    {
        "id":"G6PDH","name":"G6P Dehydrogenase","enzyme":"G6PD","ec":"1.1.1.49",
        "kegg_rxn":"R00835","substrates":["g6p_ppp","nadp"],"products":["p6gl","nadph"],
        "pathway":"ppp","reversible":False,"delta_g":-17.6,
        "stoich_sub":[1,1],"stoich_prod":[1,1],
        "regulation":"RATE-LIMITING: inhibited by NADPH; activated by NADP+; X-linked enzyme",
        "atp_yield":0,
    },
    {
        "id":"LAC","name":"Lactonase","enzyme":"PGLS","ec":"3.1.1.31",
        "kegg_rxn":"R01528","substrates":["p6gl","h2o"],"products":["p6g"],
        "pathway":"ppp","reversible":False,"delta_g":-21.8,
        "stoich_sub":[1,1],"stoich_prod":[1],
        "regulation":"spontaneous hydrolysis",
        "atp_yield":0,
    },
    {
        "id":"PGD","name":"6-PG Dehydrogenase","enzyme":"PGD","ec":"1.1.1.44",
        "kegg_rxn":"R01528","substrates":["p6g","nadp"],"products":["ru5p","nadph","co2"],
        "pathway":"ppp","reversible":False,"delta_g":-10.0,
        "stoich_sub":[1,1],"stoich_prod":[1,1,1],
        "regulation":"inhibited by NADPH; produces second NADPH",
        "atp_yield":0,
    },
    {
        "id":"RPI","name":"Ribose-5-P Isomerase","enzyme":"RPIA","ec":"5.3.1.6",
        "kegg_rxn":"R01529","substrates":["ru5p"],"products":["r5p"],
        "pathway":"ppp","reversible":True,"delta_g":-2.4,
        "stoich_sub":[1],"stoich_prod":[1],
        "regulation":"near-equilibrium",
        "atp_yield":0,
    },
    {
        "id":"RPE","name":"Ribulose-P Epimerase","enzyme":"RPE","ec":"5.1.3.1",
        "kegg_rxn":"R01529","substrates":["ru5p"],"products":["x5p"],
        "pathway":"ppp","reversible":True,"delta_g":-2.4,
        "stoich_sub":[1],"stoich_prod":[1],
        "regulation":"near-equilibrium",
        "atp_yield":0,
    },
    {
        "id":"TK1","name":"Transketolase (1)","enzyme":"TKT","ec":"2.2.1.1",
        "kegg_rxn":"R01529","substrates":["x5p","r5p"],"products":["s7p","gap"],
        "pathway":"ppp","reversible":True,"delta_g":-0.4,
        "stoich_sub":[1,1],"stoich_prod":[1,1],
        "regulation":"TPP cofactor; near-equilibrium",
        "atp_yield":0,
    },
    {
        "id":"TA","name":"Transaldolase","enzyme":"TALDO1","ec":"2.2.1.2",
        "kegg_rxn":"R01829","substrates":["s7p","gap"],"products":["e4p","f6p"],
        "pathway":"ppp","reversible":True,"delta_g":-0.5,
        "stoich_sub":[1,1],"stoich_prod":[1,1],
        "regulation":"near-equilibrium",
        "atp_yield":0,
    },
    {
        "id":"TK2","name":"Transketolase (2)","enzyme":"TKT","ec":"2.2.1.1",
        "kegg_rxn":"R01829","substrates":["x5p","e4p"],"products":["f6p","gap"],
        "pathway":"ppp","reversible":True,"delta_g":-0.4,
        "stoich_sub":[1,1],"stoich_prod":[1,1],
        "regulation":"same enzyme as TK1",
        "atp_yield":0,
    },

    # ── GLUCONEOGENESIS (unique steps) ────────────────────────────
    {
        "id":"PC","name":"Pyruvate Carboxylase","enzyme":"PC","ec":"6.4.1.1",
        "kegg_rxn":"R00344","substrates":["pyr","co2","atp"],"products":["oaa","adp","pi"],
        "pathway":"gluconeo","reversible":False,"delta_g":-4.6,
        "stoich_sub":[1,1,1],"stoich_prod":[1,1,1],
        "regulation":"activated by Acetyl-CoA (key signal of energy surplus)",
        "atp_yield":-1,
    },
    {
        "id":"PEPCK","name":"PEPCK","enzyme":"PCK1","ec":"4.1.1.32",
        "kegg_rxn":"R00341","substrates":["oaa","gtp"],"products":["pep","co2","gdp"],
        "pathway":"gluconeo","reversible":False,"delta_g":0.8,
        "stoich_sub":[1,1],"stoich_prod":[1,1,1],
        "regulation":"induced by glucagon/cortisol; repressed by insulin",
        "atp_yield":0,
    },
    {
        "id":"FBPase","name":"Fructose-1,6-bisphosphatase","enzyme":"FBP1","ec":"3.1.3.11",
        "kegg_rxn":"R04780","substrates":["f16bp","h2o"],"products":["f6p","pi"],
        "pathway":"gluconeo","reversible":False,"delta_g":-16.3,
        "stoich_sub":[1,1],"stoich_prod":[1,1],
        "regulation":"inhibited by AMP/F2,6-BP; activated by ATP/citrate",
        "atp_yield":0,
    },
    {
        "id":"G6Pase","name":"Glucose-6-Phosphatase","enzyme":"G6PC","ec":"3.1.3.9",
        "kegg_rxn":"R00305","substrates":["g6p","h2o"],"products":["glc","pi"],
        "pathway":"gluconeo","reversible":False,"delta_g":-13.8,
        "stoich_sub":[1,1],"stoich_prod":[1,1],
        "regulation":"ER membrane; hepatic/renal expression; absent in muscle",
        "atp_yield":0,
    },
    {
        "id":"ME","name":"Malic Enzyme","enzyme":"ME1","ec":"1.1.1.40",
        "kegg_rxn":"R00214","substrates":["mal_c","nadp"],"products":["pyr","co2","nadph"],
        "pathway":"gluconeo","reversible":False,"delta_g":-2.1,
        "stoich_sub":[1,1],"stoich_prod":[1,1,1],
        "regulation":"activated by fumarate; inhibited by ATP",
        "atp_yield":0,
    },
]

# ── Spatial layout for Escher-style map ───────────────────────────────────────
# Positions normalized to a 1000×800 canvas (can be scaled)
LAYOUT: Dict[str, Tuple[float,float]] = {
    # Glycolysis — vertical spine left-center
    "glc":    (400, 30),
    "g6p":    (400, 110),
    "f6p":    (400, 190),
    "f16bp":  (400, 270),
    "dhap":   (270, 340),
    "gap":    (530, 340),
    "bpg":    (530, 420),
    "pg3":    (530, 500),
    "pg2":    (530, 570),
    "pep":    (530, 640),
    "pyr":    (530, 720),
    "lac":    (680, 760),

    # Shared cofactors — float near
    "atp":    (180, 60),
    "adp":    (180, 130),
    "nad":    (650, 340),
    "nadh":   (650, 420),
    "pi":     (180, 440),

    # TCA — right circle
    "accoa":  (700, 720),
    "oaa":    (840, 640),
    "cit":    (940, 550),
    "icit":   (970, 440),
    "akg":    (940, 330),
    "scoa":   (860, 230),
    "suc":    (730, 200),
    "fum":    (640, 270),
    "mal":    (650, 380),
    "co2":    (870, 440),
    "gtp":    (780, 180),
    "gdp":    (860, 150),
    "coa":    (760, 660),
    "nadm":   (980, 500),
    "nadhm":  (980, 370),
    "fadh2":  (640, 195),
    "h2o":    (700, 295),
    "o2":     (850, 760),
    "oxal":   (840, 640),

    # OxPhos
    "oaa":    (840, 640),

    # PPP — left branch
    "g6p_ppp":(200, 110),
    "p6gl":   (110, 190),
    "p6g":    (110, 270),
    "ru5p":   (110, 350),
    "r5p":    (40,  430),
    "x5p":    (200, 430),
    "s7p":    (110, 510),
    "e4p":    (200, 560),
    "nadph":  (30,  190),
    "nadp":   (30,  270),

    # Gluconeogenesis extras
    "mal_c":  (400, 680),
    "oxal":   (400, 760),
}

# ── GNN training data (heuristic but biologically-constrained fluxes) ──────────
# Each sample: [glucose_mM, oxygen_fraction, enzyme_mult, temp_C] → flux vector
def generate_training_samples(n: int = 2000) -> Tuple[List, List]:
    import random, math
    inputs, outputs = [], []
    rxn_ids = [r["id"] for r in REACTIONS]

    for _ in range(n):
        g  = random.uniform(0.1, 20.0)   # glucose mM
        o  = random.uniform(0.0, 1.0)    # O2 fraction
        e  = random.uniform(0.1, 3.0)    # enzyme mult
        t  = random.uniform(25.0, 42.0)  # temp °C
        tf = math.exp(-0.04 * (t - 37)**2)

        # FBA-like heuristic labels (biologically constrained)
        glc_flux = e * tf * 9.0 * g / (g + 1.5)
        pdk_flux = glc_flux * 0.85 * (o / (o + 0.05))
        tca_base = pdk_flux * 0.9
        ppp_share = 0.12 if g > 2 else 0.06
        ppp_flux  = glc_flux * ppp_share * e * tf
        ferm_flux = max(0, glc_flux * 0.7 * max(0, 1 - o/0.21))
        oxphos    = tca_base * 3.5 * (o / (o + 0.02))
        gng_act   = max(0, 1.5 - g) / 1.5
        gng_flux  = gng_act * e * tf * 3.0

        flux_map = {
            "HK": glc_flux*0.5, "PGI": glc_flux*0.8, "PFK": glc_flux*0.85,
            "ALD": glc_flux*0.85, "TPI": glc_flux*0.85, "GAPDH": glc_flux*0.9,
            "PGK": glc_flux*0.9, "PGM": glc_flux*0.9, "ENO": glc_flux*0.9,
            "PK": glc_flux, "LDH": ferm_flux,
            "PDH": pdk_flux, "CS": tca_base, "ACO": tca_base*0.95,
            "IDH": tca_base*0.92, "OGDH": tca_base*0.88, "SUCL": tca_base*0.85,
            "SDH": tca_base*0.82, "FUM": tca_base*0.80, "MDH": tca_base*0.78,
            "OXPHOS": oxphos,
            "G6PDH": ppp_flux, "LAC": ppp_flux*0.9, "PGD": ppp_flux*0.85,
            "RPI": ppp_flux*0.7, "RPE": ppp_flux*0.6, "TK1": ppp_flux*0.5,
            "TA": ppp_flux*0.4, "TK2": ppp_flux*0.4,
            "PC": gng_flux*0.8, "PEPCK": gng_flux*0.8, "FBPase": gng_flux*0.7,
            "G6Pase": gng_flux*0.65, "ME": gng_flux*0.5,
        }
        fluxes = [max(0, flux_map.get(r, 0) + random.gauss(0, 0.2)) for r in rxn_ids]
        inputs.append([g, o, e, t])
        outputs.append(fluxes)

    return inputs, outputs
