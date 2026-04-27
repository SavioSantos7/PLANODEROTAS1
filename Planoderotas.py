import io
import math
import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# CONFIG
# =========================
OCCUPANCY_M3 = 0.90
OCCUPANCY_KG = 0.90

FLEET_PRIORITY = {
    "KANGU": 1,
    "FF": 2,
    "SPOT": 3,
    "SPOT DPC": 4,  # <- última prioridade
}

# HUB exclusivo para modais elétricos
ELECTRIC_HUB = "BRRC01"

# Pesos do score de cauda (hub_tail_score)
SCORE_HEAVY_WEIGHT = 0.55
SCORE_P95_WEIGHT = 0.45
SCORE_HEAVY_THRESHOLD = 0.75  # % do baseline efetivo para considerar "pesado"

# Limites para auto-detecção de colunas invertidas (Modal x Tipo Frota)
SWAP_MIN_FLEET_PCT = 0.50
SWAP_MIN_DELTA = 0.30

# =========================
# SANITY HELPERS (auto-correção de colunas Modal x Tipo Frota)
# =========================
def _norm_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()

def _fleet_match_pct(s: pd.Series) -> float:
    fleet_set = set([k.upper() for k in FLEET_PRIORITY.keys()]) | {"KANGU"}
    if s is None or len(s) == 0:
        return 0.0
    v = _norm_str_series(s)
    return float(v.isin(fleet_set).mean())

def _maybe_swap_modal_frota(plan_df: pd.DataFrame, col_modal: str, col_frota: str):
    """Se detectar que col_modal parece 'Tipo Frota' (FF/SPOT/...) e col_frota parece 'Modal' (VUC/MÉDIO/...), troca."""
    try:
        pct_modal = _fleet_match_pct(plan_df[col_modal])
        pct_frota = _fleet_match_pct(plan_df[col_frota])
    except Exception as e:
        st.warning(f"⚠️ Auto-correção de colunas (Modal/Tipo Frota) falhou: {e}. Usando colunas como detectadas.")
        return col_modal, col_frota
    if pct_frota < SWAP_MIN_FLEET_PCT and pct_modal > SWAP_MIN_FLEET_PCT and (pct_modal - pct_frota) > SWAP_MIN_DELTA:
        st.info("🔄 Colunas 'Modal' e 'Tipo Frota' detectadas invertidas — corrigido automaticamente.")
        return col_frota, col_modal
    return col_modal, col_frota

# ✅ AJUSTE: capacidades (kg) atualizadas para os modais solicitados (1800 kg)
CAPACITY_ROWS = [
    ("Vuc", 16, 1600),
    ("Van", 8, 1500),
    ("Médio", 25, 3500),
    ("Truck", 50, 12000),
    ("Carreta", 90, 24000),

    ("Vuc EL", 16, 1600),
    ("VUC com ajudante", 17, 1400),
    ("HR", 12, 1800),
    ("M1 Rental Médio DD*FM", 37, 3500),
    ("Toco", 40, 6000),

    ("MELIONE RENTAL VAN", 8, 2200),
    ("M1 Rental Vuc DD*FM", 17, 1600),

    # ====== AJUSTADOS/ADICIONADOS PARA 1800 KG ======
    ("Melione VUC Elétrico", 16, 1800),
    ("VUC Elétrico", 16, 1800),
    ("Vuc Rental TKS", 20, 1800),
    ("Melione Vuc Rental TKS", 20, 1800),
    ("Rental VUC FM", 17, 1800),
    ("VUC Dedicado com Ajudante", 17, 1800),
    ("VUC Dedicado FBM 4K", 17, 1800),
    ("VUC Dedicado FBM 7K", 17, 1800),
    # ===============================================

    ("M1 VUC DD*FF", 17, 1600),
    ("MeliOne Yellow Pool", 8, 2200),
    ("Rental Medio FM", 37, 3500),
    ("Van Frota Fixa - Equipe dupla", 8, 1500),
    ("Utilitários", 3, 650),
]

# =========================
# HELPERS
# =========================
def norm(s: str) -> str:
    """Normaliza textos para comparações/matches.

    - Uppercase
    - Remove acentos/diacríticos (ex.: 'MÉDIO' -> 'MEDIO')
    - Mantém apenas A-Z, 0-9 e espaços
    """
    s = str(s).upper()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Z0-9]+", " ", s).strip()
    return s


# =========================
# REGRAS ESPECÍFICAS
# =========================
ELECTRIC_MODALS = {
    norm("Vuc EL"),
    norm("Melione VUC Elétrico"),
    norm("VUC Elétrico"),
}

def is_electric_modal(modal: str) -> bool:
    """Identifica modais elétricos (tolerante a variações com/sem acento)."""
    m = norm(modal)
    # Cobertura explícita dos nomes conhecidos
    if m in ELECTRIC_MODALS:
        return True
    # Cobertura para variações comuns (ex.: 'VUC ELETRICO', 'VUC ELÉTRICO', etc.)
    if "VUC" in m and ("ELETRICO" in m or "EL TRICO" in m):
        return True
    # Cobertura para abreviação 'VUC EL'
    if m.startswith("VUC EL") or " VUC EL " in f" {m} ":
        return True
    return False



def cluster_synergy_key(cluster: str) -> str:
    """
    Regra de sinergia:
    - Todo cluster contém '.'
    - Se o "prefixo antes do ponto" se repete, então esses clusters podem compartilhar rotas.
      Ex: 'CLUSTER 1.1 OESTE' e 'CLUSTER 1.2 SUDOESTE' => chave 'CLUSTER 1'
    """
    s = str(cluster).strip()
    if "." not in s:
        return s
    return s.split(".", 1)[0].strip()


def vehicle_class(modal: str) -> str:
    m = norm(modal)
    if "CARRETA" in m:
        return "CARRETA"
    if "TRUCK" in m:
        return "TRUCK"
    if "TOCO" in m:
        return "TOCO"
    if "MEDIO" in m:
        return "MEDIO"
    if "HR" in m:
        return "HR"
    if "VAN" in m:
        return "VAN"
    if "VUC" in m:
        return "VUC"
    return "OUTRO"


cap_df = pd.DataFrame(CAPACITY_ROWS, columns=["perfil", "cap_m3", "cap_kg"])
cap_df["perfil_norm"] = cap_df["perfil"].map(norm)


def capacity_for_modal(modal: str):
    m = norm(modal)

    exact = cap_df[cap_df["perfil_norm"] == m]
    if len(exact):
        r = exact.iloc[0]
        return float(r.cap_m3), float(r.cap_kg), r.perfil

    matches = []
    for _, row in cap_df.iterrows():
        if row.perfil_norm and row.perfil_norm in m:
            matches.append((len(row.perfil_norm), row.cap_m3, row.cap_kg, row.perfil))
    if matches:
        matches.sort(reverse=True)
        _, m3, kg, perfil = matches[0]
        return float(m3), float(kg), perfil

    if "MEDIO" in m:
        r = (
            cap_df[cap_df["perfil_norm"].str.contains("MEDIO")]
            .sort_values("cap_m3", ascending=False)
            .iloc[0]
        )
        return float(r.cap_m3), float(r.cap_kg), r.perfil
    if "VUC" in m:
        r = cap_df[cap_df["perfil_norm"] == "VUC"].iloc[0]
        return float(r.cap_m3), float(r.cap_kg), r.perfil
    if "VAN" in m:
        r = cap_df[cap_df["perfil_norm"] == "VAN"].iloc[0]
        return float(r.cap_m3), float(r.cap_kg), r.perfil

    return np.nan, np.nan, None


def find_col(df, candidates):
    cols_norm = {norm(c): c for c in df.columns}
    for cand in candidates:
        c = cols_norm.get(norm(cand))
        if c is not None:
            return c
    for cand in candidates:
        cn = norm(cand)
        for k, orig in cols_norm.items():
            if cn in k:
                return orig
    return None


def parse_number_series(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    x = x.str.replace(".", "", regex=False)
    x = x.str.replace(",", ".", regex=False)
    return pd.to_numeric(x, errors="coerce")


# =========================
# CAPACIDADES EFETIVAS
# =========================
# ✅ Ajuste de baseline do VUC para 1800 kg (efetivo = 1800 * OCCUPANCY_KG)
VUC_BASE_M3_EFF = 16 * OCCUPANCY_M3
VUC_BASE_KG_EFF = 1800 * OCCUPANCY_KG

MEDIO_BASE_M3_EFF = 37 * OCCUPANCY_M3
MEDIO_BASE_KG_EFF = 3500 * OCCUPANCY_KG

# ✅ NOVA REGRA do MIN_MEDIO (oversize nominal)
MIN_MEDIO_OVERSIZE_M3 = 16.0
MIN_MEDIO_OVERSIZE_KG = 1800.0


def split_oversize_vs_vuc(is_hub: pd.DataFrame):
    """
    ✅ Regra MIN_MEDIO:
    tudo que tiver >= 16 m3 OU >= 1800 kg entra no bloco obrigatório de MIN_MEDIO
    """
    overs = is_hub[
        (is_hub["Peso_kg"] >= MIN_MEDIO_OVERSIZE_KG)
        | (is_hub["Volume_m3"] >= MIN_MEDIO_OVERSIZE_M3)
    ]
    rem = is_hub.drop(overs.index)
    return overs, rem


def required_units_by_capacity(sum_kg, sum_m3, cap_kg_eff, cap_m3_eff):
    if cap_kg_eff <= 0 or cap_m3_eff <= 0:
        return 0
    return int(math.ceil(max(sum_kg / cap_kg_eff, sum_m3 / cap_m3_eff)))


# =========================
# SCORE (EXTRAS)
# =========================
def hub_tail_score(is_hub: pd.DataFrame):
    kg = is_hub["Peso_kg"].astype(float)
    m3 = is_hub["Volume_m3"].astype(float)

    # (mantém lógica de score baseada na baseline efetiva do VUC)
    overs = (kg > VUC_BASE_KG_EFF) | (m3 > VUC_BASE_M3_EFF)
    fits = ~overs
    df_fit = is_hub[fits].copy()

    thr_kg = SCORE_HEAVY_THRESHOLD * VUC_BASE_KG_EFF
    thr_m3 = SCORE_HEAVY_THRESHOLD * VUC_BASE_M3_EFF
    heavy = df_fit[(df_fit["Peso_kg"] >= thr_kg) | (df_fit["Volume_m3"] >= thr_m3)]

    heavy_kg = float(heavy["Peso_kg"].sum())
    heavy_m3 = float(heavy["Volume_m3"].sum())

    p95_kg = float(np.nanpercentile(kg, 95)) if len(kg) else 0.0
    p95_m3 = float(np.nanpercentile(m3, 95)) if len(m3) else 0.0

    score = (
        SCORE_HEAVY_WEIGHT
        * max(
            heavy_kg / VUC_BASE_KG_EFF if VUC_BASE_KG_EFF else 0,
            heavy_m3 / VUC_BASE_M3_EFF if VUC_BASE_M3_EFF else 0,
        )
        + SCORE_P95_WEIGHT
        * (
            0.5 * (p95_kg / VUC_BASE_KG_EFF if VUC_BASE_KG_EFF else 0)
            + 0.5 * (p95_m3 / VUC_BASE_M3_EFF if VUC_BASE_M3_EFF else 0)
        )
    )

    extra_need = int(
        math.ceil(
            max(
                heavy_kg / MEDIO_BASE_KG_EFF if MEDIO_BASE_KG_EFF else 0,
                heavy_m3 / MEDIO_BASE_M3_EFF if MEDIO_BASE_M3_EFF else 0,
            )
        )
    )

    return {"score": float(score), "extra_need": int(extra_need), "p95_kg": p95_kg, "p95_m3": p95_m3}


def proportional_split(scores: dict, needs: dict, total_supply: int):
    hubs = [h for h in scores if scores[h] > 0 and needs.get(h, 0) > 0]
    if total_supply <= 0 or not hubs:
        return {}
    tot = sum(scores[h] for h in hubs)
    if tot <= 0:
        return {}

    raw = {h: total_supply * (scores[h] / tot) for h in hubs}
    base = {h: min(needs[h], int(math.floor(raw[h]))) for h in hubs}

    used = sum(base.values())
    rem = total_supply - used

    frac = sorted(
        [(h, raw[h] - math.floor(raw[h])) for h in hubs], key=lambda x: x[1], reverse=True
    )
    i = 0
    while rem > 0 and frac:
        h = frac[i][0]
        if base[h] < needs[h]:
            base[h] += 1
            rem -= 1
        i = (i + 1) % len(frac)
        if all(base[x] >= needs[x] for x in hubs):
            break
    return base


# =========================
# POOL ALLOCATION
# =========================
def selector_class(cls_name: str):
    return lambda r: r["vehicle_class"] == cls_name


def is_big_vehicle_row(r):
    if pd.isna(r["cap_m3_eff"]) or pd.isna(r["cap_kg_eff"]):
        return False
    return (r["cap_m3_eff"] >= VUC_BASE_M3_EFF) or (r["cap_kg_eff"] >= VUC_BASE_KG_EFF)


def selector_big(r):
    return is_big_vehicle_row(r)


def allocate_one_best(
    plan_pool: pd.DataFrame,
    selector_fn,
    demand_cluster: str | None = None,
    demand_hub: str | None = None,
    group_key: str | None = None,
    tracker: dict | None = None,
    group_supply: dict | None = None,
):
    """Seleciona 1 veículo respeitando:
    - prioridade de frota (Kangu -> FF -> Spot -> Spot DPC)
    - preferências de capacidade já existentes (cap_m3_eff/cap_kg_eff)
    - Kangu NÃO pode fazer sinergia entre clusters
    - uso proporcional entre Transportadoras dentro do mesmo grupo de sinergia,
      aplicado PARA TODOS OS MODAIS (vehicle_class).
    """
    eligible = plan_pool[(plan_pool["avail"] > 0)].copy()
    eligible = eligible[eligible.apply(selector_fn, axis=1)].copy()

    # Regra: Kangu NÃO pode fazer sinergia entre clusters.
    if demand_cluster is not None and not eligible.empty:
        kangu_mask = eligible["Tipo Frota"].astype(str).str.upper().str.strip().eq("KANGU")
        eligible = pd.concat(
            [
                eligible[~kangu_mask],
                eligible[kangu_mask & (eligible["Cluster"].astype(str) == str(demand_cluster))],
            ],
            axis=0,
        )

    if eligible.empty:
        return None, plan_pool

    # Regra: modais elétricos rodam exclusivamente no HUB BRRC01.
    # Se a demanda for de outro HUB, esses modais não podem ser selecionados.
    electric_mask = eligible["Modal"].astype(str).apply(lambda x: is_electric_modal(x))
    if demand_hub is not None:
        if str(demand_hub).strip().upper() != ELECTRIC_HUB:
            eligible = eligible[~electric_mask].copy()
    else:
        # Sem HUB informado: por segurança, não aloca elétricos.
        eligible = eligible[~electric_mask].copy()

    if eligible.empty:
        return None, plan_pool

    if tracker is None:
        tracker = {}
    if group_supply is None:
        group_supply = {}

    # 1) Decide o bucket (fleet_priority + vehicle_class)
    base_sorted = eligible.sort_values(
        ["fleet_priority", "cap_m3_eff", "cap_kg_eff", "avail"],
        ascending=[True, False, False, False],
    )
    base_row = base_sorted.iloc[0]
    fp_target = int(base_row.get("fleet_priority", 9))
    vc_target = str(base_row.get("vehicle_class", ""))

    # 2) Proporcionalidade por transportadora dentro do bucket
    bucket = eligible[
        (eligible["fleet_priority"].astype(int) == fp_target)
        & (eligible["vehicle_class"].astype(str) == vc_target)
    ].copy()

    if bucket.empty:
        bucket = eligible.copy()

    # Preferência: no HUB BRRC01, quando o bucket é VUC e existe VUC elétrico no mesmo bucket,
    # prioriza elétricos (sem quebrar a prioridade de frota).
    if demand_hub is not None and str(demand_hub).strip().upper() == ELECTRIC_HUB and vc_target == "VUC":
        bucket_e = bucket[bucket["Modal"].astype(str).apply(lambda x: is_electric_modal(x))].copy()
        if not bucket_e.empty:
            bucket = bucket_e

    gk = str(group_key) if group_key is not None else ""

    def usage_ratio_row(r):
        vc = str(r.get("vehicle_class", ""))
        fp = int(r.get("fleet_priority", 9))
        tr = str(r.get("Transportadora", ""))
        denom = float(group_supply.get((gk, vc, fp, tr), 0.0))
        if denom <= 0:
            denom = float(r.get("init_avail", 1)) if float(r.get("init_avail", 0) or 0) > 0 else 1.0
        used = float(tracker.get((gk, vc, fp, tr), 0))
        return used / denom

    bucket["_usage_ratio"] = bucket.apply(usage_ratio_row, axis=1)
    bucket["_used_abs"] = bucket.apply(
        lambda r: float(
            tracker.get(
                (
                    gk,
                    str(r.get("vehicle_class", "")),
                    int(r.get("fleet_priority", 9)),
                    str(r.get("Transportadora", "")),
                ),
                0,
            )
        ),
        axis=1,
    )

    bucket = bucket.sort_values(
        ["_usage_ratio", "cap_m3_eff", "cap_kg_eff", "_used_abs", "avail"],
        ascending=[True, False, False, True, False],
    )

    row = bucket.iloc[0]
    idx = row.name
    plan_pool.loc[idx, "avail"] = int(plan_pool.loc[idx, "avail"]) - 1

    vc = str(row.get("vehicle_class", ""))
    fp = int(row.get("fleet_priority", 9))
    tr = str(row.get("Transportadora", ""))
    tracker[(gk, vc, fp, tr)] = int(tracker.get((gk, vc, fp, tr), 0)) + 1

    return row, plan_pool


def cluster_demand_score(df_cluster: pd.DataFrame) -> float:
    sum_kg = float(df_cluster["Peso_kg"].sum())
    sum_m3 = float(df_cluster["Volume_m3"].sum())
    if VUC_BASE_KG_EFF <= 0 or VUC_BASE_M3_EFF <= 0:
        return sum_kg + sum_m3
    return float(max(sum_kg / VUC_BASE_KG_EFF, sum_m3 / VUC_BASE_M3_EFF))


def allocate_for_cluster(
    cluster_name: str,
    group_key: str,
    is_cluster: pd.DataFrame,
    plan_pool: pd.DataFrame,
    group_supply: dict,
    tracker: dict,
    all_scores: list,
    all_faltas: list,
):
    records = []

    # 0) score hubs
    hub_meta = {}
    for hub, df_hub in is_cluster.groupby("HUB"):
        s = hub_tail_score(df_hub)
        hub_meta[hub] = {"score": s["score"], "extra_need": s["extra_need"]}
        all_scores.append({"Grupo_Sinergia": group_key, "Cluster": cluster_name, "HUB": hub, **s})

    hubs_sorted = sorted([(h, hub_meta[h]["score"]) for h in hub_meta], key=lambda x: x[1], reverse=True)

    # 1) demanda por HUB (após remover oversize)
    hub_demand = {}
    for hub, df_hub in is_cluster.groupby("HUB"):
        overs, rem = split_oversize_vs_vuc(df_hub)
        hub_demand[hub] = {
            "rem_kg": float(rem["Peso_kg"].sum()),
            "rem_m3": float(rem["Volume_m3"].sum()),
            "ov_kg": float(overs["Peso_kg"].sum()),
            "ov_m3": float(overs["Volume_m3"].sum()),
        }

    # 2) MIN_MEDIO (obrigatório) - oversize pela regra nova (>=16m3 OU >=1800kg)
    for hub in sorted(hub_demand.keys()):
        sum_ov_kg = hub_demand[hub]["ov_kg"]
        sum_ov_m3 = hub_demand[hub]["ov_m3"]
        min_medio = required_units_by_capacity(sum_ov_kg, sum_ov_m3, MEDIO_BASE_KG_EFF, MEDIO_BASE_M3_EFF)

        for _ in range(min_medio):
            row, plan_pool = allocate_one_best(
                plan_pool,
                selector_class("MEDIO"),
                demand_cluster=cluster_name,
                demand_hub=hub,
                group_key=group_key,
                tracker=tracker,
                group_supply=group_supply,
            )
            if row is None:
                all_faltas.append({"Grupo_Sinergia": group_key, "Cluster": cluster_name, "HUB": hub, "Tipo": "MIN_MEDIO", "Faltou": 1})
                break

            records.append({
                "Grupo_Sinergia": group_key,
                "Cluster": cluster_name,
                "Cluster_Oferta": row["Cluster"],
                "HUB": hub,
                "Tipo": "MIN_MEDIO",
                "Transportadora": row["Transportadora"],
                "Tipo Frota": row["Tipo Frota"],
                "Modal": row["Modal"],
                "Veiculos": 1,
            })

    # 2.5) PRÉ-DISTRIBUIÇÃO DE MÉDIOS POR PERFIL MÉDIO (pós MIN_MEDIO)
    # Ideia: reservar veículos classe MEDIO para HUBs cuja média de Peso/m³ (das IS remanescentes)
    # é maior, evitando que o consumo greedy do MIN_FILL concentre médios em HUBs que aparecem primeiro.
    hub_mean_scores = {}
    hub_medio_needs = {}
    for hub, df_hub in is_cluster.groupby("HUB"):
        overs, rem = split_oversize_vs_vuc(df_hub)

        if rem.empty:
            mean_kg = 0.0
            mean_m3 = 0.0
        else:
            mean_kg = float(rem["Peso_kg"].mean())
            mean_m3 = float(rem["Volume_m3"].mean())

        # Score normalizado pelo baseline efetivo do VUC (pós-ocupação)
        score = 0.0
        if mean_kg > 0.0 or mean_m3 > 0.0:
            score = (mean_kg / max(VUC_BASE_KG_EFF, 1e-9)) + (mean_m3 / max(VUC_BASE_M3_EFF, 1e-9))

        need = required_units_by_capacity(
            float(hub_demand[hub]["rem_kg"]),
            float(hub_demand[hub]["rem_m3"]),
            MEDIO_BASE_KG_EFF,
            MEDIO_BASE_M3_EFF,
        )
        if need > 0:
            hub_mean_scores[hub] = score
            hub_medio_needs[hub] = need

    remaining_medio_supply = int(plan_pool[plan_pool["vehicle_class"].astype(str) == "MEDIO"]["avail"].sum())
    if remaining_medio_supply > 0 and hub_medio_needs:
        # Se todos os scores forem 0, cai para distribuição uniforme (ainda respeitando 'needs').
        if sum(max(hub_mean_scores.get(h, 0.0), 0.0) for h in hub_medio_needs) <= 1e-12:
            scores_mean = {h: 1.0 for h in hub_medio_needs}
        else:
            scores_mean = {h: max(hub_mean_scores.get(h, 0.0), 1e-9) for h in hub_medio_needs}

        medio_by_hub = proportional_split(scores_mean, hub_medio_needs, remaining_medio_supply)

        for hub, _ in sorted(medio_by_hub.items(), key=lambda kv: scores_mean.get(kv[0], 0.0), reverse=True):
            units = int(medio_by_hub.get(hub, 0))
            if units <= 0:
                continue

            for _ in range(units):
                if hub_demand[hub]["rem_kg"] <= 1e-6 and hub_demand[hub]["rem_m3"] <= 1e-6:
                    break

                row, plan_pool = allocate_one_best(
                    plan_pool,
                    selector_class("MEDIO"),
                    demand_cluster=cluster_name,
                    demand_hub=hub,
                    group_key=group_key,
                    tracker=tracker,
                    group_supply=group_supply,
                )
                if row is None:
                    all_faltas.append({"Grupo_Sinergia": group_key, "Cluster": cluster_name, "HUB": hub, "Tipo": "MIN_FILL", "Faltou": 1})
                    break

                # Mantém a estrutura do código/visualizações: classifica como MIN_FILL
                records.append({
                    "Grupo_Sinergia": group_key,
                    "Cluster": cluster_name,
                    "Cluster_Oferta": row["Cluster"],
                    "HUB": hub,
                    "Tipo": "MIN_FILL",
                    "Transportadora": row["Transportadora"],
                    "Tipo Frota": row["Tipo Frota"],
                    "Modal": row["Modal"],
                    "Veiculos": 1,
                })

                hub_demand[hub]["rem_kg"] = max(0.0, float(hub_demand[hub]["rem_kg"]) - float(row["cap_kg_eff"]))
                hub_demand[hub]["rem_m3"] = max(0.0, float(hub_demand[hub]["rem_m3"]) - float(row["cap_m3_eff"]))

    # 3) EXTRAS (UPGRADE)
    remaining_big_supply = int(plan_pool[plan_pool.apply(is_big_vehicle_row, axis=1)]["avail"].sum())
    scores = {h: hub_meta[h]["score"] for h, _ in hubs_sorted}
    needs  = {h: max(0, hub_meta[h]["extra_need"]) for h, _ in hubs_sorted}
    extras_by_hub = proportional_split(scores, needs, remaining_big_supply)

    for hub, _ in hubs_sorted:
        extra_units = int(extras_by_hub.get(hub, 0))
        if extra_units <= 0:
            continue

        for _ in range(extra_units):
            if hub_demand[hub]["rem_kg"] <= 1e-6 and hub_demand[hub]["rem_m3"] <= 1e-6:
                break

            row, plan_pool = allocate_one_best(
                plan_pool,
                selector_big,
                demand_cluster=cluster_name,
                demand_hub=hub,
                group_key=group_key,
                tracker=tracker,
                group_supply=group_supply,
            )
            if row is None:
                all_faltas.append({"Grupo_Sinergia": group_key, "Cluster": cluster_name, "HUB": hub, "Tipo": "EXTRA_BIG", "Faltou": 1})
                break

            records.append({
                "Grupo_Sinergia": group_key,
                "Cluster": cluster_name,
                "Cluster_Oferta": row["Cluster"],
                "HUB": hub,
                "Tipo": "EXTRA_BIG",
                "Transportadora": row["Transportadora"],
                "Tipo Frota": row["Tipo Frota"],
                "Modal": row["Modal"],
                "Veiculos": 1,
            })

            hub_demand[hub]["rem_kg"] = max(0.0, hub_demand[hub]["rem_kg"] - float(row["cap_kg_eff"]))
            hub_demand[hub]["rem_m3"] = max(0.0, hub_demand[hub]["rem_m3"] - float(row["cap_m3_eff"]))

    # 4) MIN_FILL
    for hub in sorted(hub_demand.keys()):
        rem_kg = float(hub_demand[hub]["rem_kg"])
        rem_m3 = float(hub_demand[hub]["rem_m3"])

        max_iter = len(plan_pool) + 10
        iter_count = 0
        while (rem_kg > 1e-6 or rem_m3 > 1e-6) and iter_count < max_iter:
            iter_count += 1
            row, plan_pool = allocate_one_best(
                plan_pool,
                lambda r: True,
                demand_cluster=cluster_name,
                demand_hub=hub,
                group_key=group_key,
                tracker=tracker,
                group_supply=group_supply,
            )
            if row is None:
                records.append({
                    "Grupo_Sinergia": group_key,
                    "Cluster": cluster_name,
                    "Cluster_Oferta": "",
                    "HUB": hub,
                    "Tipo": "MIN_FILL",
                    "Transportadora": "(SEM OFERTA)",
                    "Tipo Frota": "",
                    "Modal": "(SEM OFERTA)",
                    "Veiculos": 1,
                })
                break

            records.append({
                "Grupo_Sinergia": group_key,
                "Cluster": cluster_name,
                "Cluster_Oferta": row["Cluster"],
                "HUB": hub,
                "Tipo": "MIN_FILL",
                "Transportadora": row["Transportadora"],
                "Tipo Frota": row["Tipo Frota"],
                "Modal": row["Modal"],
                "Veiculos": 1,
            })

            rem_kg = max(0.0, rem_kg - float(row["cap_kg_eff"]))
            rem_m3 = max(0.0, rem_m3 - float(row["cap_m3_eff"]))

        hub_demand[hub]["rem_kg"] = rem_kg
        hub_demand[hub]["rem_m3"] = rem_m3

    return records, plan_pool


# =========================
# CORE RUNNER
# =========================
def run_allocation(plan_df: pd.DataFrame, is_df: pd.DataFrame, enable_synergy: bool = True, return_debug: bool = False):
    # Detecta colunas do Plano
    col_cluster_p = find_col(plan_df, ["Cluster"])
    col_transp = find_col(plan_df, ["Transportadora", "Carrier", "Transporter"])
    col_modal = find_col(plan_df, ["Modal", "Perfil"])
    col_frota = find_col(plan_df, ["Tipo Frota", "Frota", "Fleet Type"])
    col_avail = find_col(plan_df, ["Disponibilidade de Modais", "Disponibilidade", "Qtd", "Quantidade"])

    missing_plan = [("Cluster", col_cluster_p), ("Transportadora", col_transp), ("Modal", col_modal), ("Tipo Frota", col_frota), ("Disponibilidade", col_avail)]
    missing_plan = [name for name, col in missing_plan if col is None]
    if missing_plan:
        raise ValueError(f"PlanoRotas: não encontrei as colunas necessárias: {', '.join(missing_plan)}")


    # 🔎 Sanity check: se as colunas Modal e Tipo Frota estiverem invertidas (por header trocado ou detecção ambígua), corrige.
    col_modal, col_frota = _maybe_swap_modal_frota(plan_df, col_modal, col_frota)
    # Detecta colunas IS
    col_cluster_i = find_col(is_df, ["CLUSTER", "Cluster"])
    col_hub = find_col(is_df, ["HUB", "Warehouse", "WH", "WAREHOUSE_ID"])
    col_kg = find_col(is_df, ["Peso(kg)", "Peso", "KG", "WEIGHT"])
    col_m3 = find_col(is_df, ["Volume(m³)", "Volume", "M3", "M³", "CUBAGEM"])

    missing_is = [("Cluster", col_cluster_i), ("HUB", col_hub), ("Peso", col_kg), ("Volume", col_m3)]
    missing_is = [name for name, col in missing_is if col is None]
    if missing_is:
        raise ValueError(f"ISs: não encontrei as colunas necessárias: {', '.join(missing_is)}")

    plan = plan_df.rename(
        columns={
            col_cluster_p: "Cluster",
            col_transp: "Transportadora",
            col_modal: "Modal",
            col_frota: "Tipo Frota",
            col_avail: "Disponibilidade",
        }
    ).copy()

    plan["Disponibilidade"] = pd.to_numeric(plan["Disponibilidade"], errors="coerce").fillna(0).astype(int)
    plan = plan[plan["Disponibilidade"] > 0].copy()

    plan["cap_m3"], plan["cap_kg"], plan["perfil_cap"] = zip(*plan["Modal"].map(capacity_for_modal))
    plan["cap_m3_eff"] = plan["cap_m3"] * OCCUPANCY_M3
    plan["cap_kg_eff"] = plan["cap_kg"] * OCCUPANCY_KG
    plan["vehicle_class"] = plan["Modal"].map(vehicle_class)
    plan["fleet_priority"] = plan["Tipo Frota"].map(lambda x: FLEET_PRIORITY.get(str(x).upper(), 9))
    plan["avail"] = plan["Disponibilidade"].astype(int)
    plan["init_avail"] = plan["avail"]

    isdata = is_df.rename(
        columns={
            col_cluster_i: "Cluster",
            col_hub: "HUB",
            col_kg: "Peso_kg",
            col_m3: "Volume_m3",
        }
    ).copy()

    isdata["Peso_kg"] = parse_number_series(isdata["Peso_kg"])
    isdata["Volume_m3"] = parse_number_series(isdata["Volume_m3"])
    isdata = isdata.dropna(subset=["Cluster", "HUB", "Peso_kg", "Volume_m3"]).copy()

    # Apenas clusters que existem nos dois lados (igualdade exata)
    common_clusters = sorted(list(set(plan["Cluster"].astype(str)).intersection(set(isdata["Cluster"].astype(str)))))
    if not common_clusters:
        raise ValueError("Não encontrei clusters em comum entre Plano e ISs.")

    # =========================
    # SINERGIA POR PREFIXO (antes do ponto)
    # =========================
    if enable_synergy:
        plan["Grupo_Sinergia"] = plan["Cluster"].map(cluster_synergy_key)
        isdata["Grupo_Sinergia"] = isdata["Cluster"].map(cluster_synergy_key)
    else:
        plan["Grupo_Sinergia"] = plan["Cluster"].astype(str)
        isdata["Grupo_Sinergia"] = isdata["Cluster"].astype(str)

    # Grupos só com clusters em comum
    groups = {}
    for c in common_clusters:
        g = cluster_synergy_key(c) if enable_synergy else str(c)
        groups.setdefault(g, []).append(str(c))

    all_allocs, all_saldos, all_scores, all_faltas = [], [], [], []
    tracker = {}

    # Processa por grupo (pool compartilhado dentro do grupo)
    for group_key, member_clusters in sorted(groups.items(), key=lambda x: x[0]):
        plan_pool = plan[plan["Cluster"].astype(str).isin(member_clusters)].copy()

        # base de oferta inicial por transportadora (para distribuição proporcional)
        group_supply = (
            plan_pool.groupby(["vehicle_class", "fleet_priority", "Transportadora"], as_index=False)["init_avail"]
            .sum()
        )
        group_supply = {
            (str(group_key), str(r["vehicle_class"]), int(r["fleet_priority"]), str(r["Transportadora"])): float(r["init_avail"])
            for _, r in group_supply.iterrows()
        }

        # ordem de alocação dos clusters dentro do grupo (maior demanda primeiro)
        demand_clusters = []
        for c in member_clusters:
            df_c = isdata[isdata["Cluster"].astype(str) == str(c)].copy()
            if df_c.empty:
                continue
            demand_clusters.append((c, cluster_demand_score(df_c)))
        demand_clusters.sort(key=lambda x: x[1], reverse=True)

        # roda cada cluster (demanda) usando o MESMO pool
        for cluster_name, _score in demand_clusters:
            is_cluster = isdata[isdata["Cluster"].astype(str) == str(cluster_name)].copy()
            if is_cluster.empty or plan_pool.empty:
                continue

            records, plan_pool = allocate_for_cluster(
                cluster_name=str(cluster_name),
                group_key=str(group_key),
                is_cluster=is_cluster,
                plan_pool=plan_pool,
                group_supply=group_supply,
                tracker=tracker,
                all_scores=all_scores,
                all_faltas=all_faltas,
            )

            alloc_df = pd.DataFrame(records)
            if alloc_df.empty:
                continue
            all_allocs.append(alloc_df)

        # saldo do pool ao final do grupo
        if not plan_pool.empty:
            saldo = (
                plan_pool.groupby(["Grupo_Sinergia", "Cluster", "Transportadora", "Tipo Frota", "Modal"], as_index=False)["avail"]
                .sum()
                .rename(columns={"avail": "Disponibilidade_Restante"})
                .sort_values(["Grupo_Sinergia", "Cluster", "Disponibilidade_Restante"], ascending=[True, True, False])
            )
            all_saldos.append(saldo)

    debug_alloc = pd.concat(all_allocs, ignore_index=True) if all_allocs else pd.DataFrame()
    saldo_debug = pd.concat(all_saldos, ignore_index=True) if all_saldos else pd.DataFrame()

    # =========================
    # OUTPUT FINAL
    # =========================
    if debug_alloc.empty:
        final_output = pd.DataFrame(columns=["Cluster", "HUB", "Transportadora", "Tipo Frota", "Modal", "Veiculos"])
    else:
        final_output = (
            debug_alloc.groupby(["Cluster", "HUB", "Transportadora", "Tipo Frota", "Modal"], as_index=False)["Veiculos"]
            .sum()
            .sort_values(["Cluster", "HUB", "Tipo Frota", "Transportadora", "Modal"], ascending=[True, True, True, True, True])
        )

    if saldo_debug.empty:
        final_saldo = pd.DataFrame(columns=["Cluster", "Transportadora", "Tipo Frota", "Modal", "Disponibilidade_Restante"])
    else:
        final_saldo = (
            saldo_debug.groupby(["Cluster", "Transportadora", "Tipo Frota", "Modal"], as_index=False)["Disponibilidade_Restante"]
            .sum()
            .sort_values(["Cluster", "Tipo Frota", "Transportadora", "Modal"], ascending=[True, True, True, True])
        )

    # ✅ AJUSTE ANTERIOR: saldo_plano só com Disponibilidade_Restante >= 1
    if not final_saldo.empty:
        final_saldo = final_saldo[final_saldo["Disponibilidade_Restante"] >= 1].copy()


    # 🔎 Sanity check no output: se por algum motivo Modal/Tipo Frota estiverem invertidos, corrige antes de retornar.
    if not final_output.empty:
        pct_modal = _fleet_match_pct(final_output['Modal'])
        pct_frota = _fleet_match_pct(final_output['Tipo Frota'])
        if pct_modal > 0.60 and pct_frota < 0.40:
            final_output = final_output.rename(columns={'Modal': '_tmp_modal', 'Tipo Frota': 'Modal'}).rename(columns={'_tmp_modal': 'Tipo Frota'})

    if not final_saldo.empty:
        pct_modal_s = _fleet_match_pct(final_saldo['Modal'])
        pct_frota_s = _fleet_match_pct(final_saldo['Tipo Frota'])
        if pct_modal_s > 0.60 and pct_frota_s < 0.40:
            final_saldo = final_saldo.rename(columns={'Modal': '_tmp_modal', 'Tipo Frota': 'Modal'}).rename(columns={'_tmp_modal': 'Tipo Frota'})
    if return_debug:
        plan_common = plan[plan["Cluster"].astype(str).isin(common_clusters)].copy()
        # ✅ NOVO: retorna também isdata normalizado pra análises de demanda x capacidade x plano
        return final_output, final_saldo, debug_alloc, saldo_debug, plan_common, isdata

    return final_output, final_saldo


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _safe_pct(num, den):
    return float(num) / float(den) if den and den != 0 else 0.0


def build_analyses(output_final: pd.DataFrame, saldo_final: pd.DataFrame, debug_alloc: pd.DataFrame, plan_common: pd.DataFrame) -> dict:
    analyses = {}

    if plan_common is None or plan_common.empty:
        plan_common = pd.DataFrame(columns=["Grupo_Sinergia", "Cluster", "Transportadora", "Tipo Frota", "Modal", "Disponibilidade"])
    else:
        if "Grupo_Sinergia" not in plan_common.columns:
            plan_common = plan_common.copy()
            plan_common["Grupo_Sinergia"] = plan_common["Cluster"].map(cluster_synergy_key)
        if "vehicle_class" not in plan_common.columns:
            plan_common = plan_common.copy()
            plan_common["vehicle_class"] = plan_common["Modal"].map(vehicle_class)

    used_rows = output_final.copy() if output_final is not None else pd.DataFrame()
    if not used_rows.empty:
        used_rows["vehicle_class"] = used_rows["Modal"].map(vehicle_class)

    oferta = (plan_common.groupby(["Tipo Frota"], as_index=False)["Disponibilidade"].sum().rename(columns={"Disponibilidade": "Oferta"}))
    usado = (
        used_rows.loc[~used_rows["Transportadora"].astype(str).str.contains(r"\(SEM OFERTA\)", regex=True), :]
        .groupby(["Tipo Frota"], as_index=False)["Veiculos"].sum().rename(columns={"Veiculos": "Usado"})
    ) if not used_rows.empty else pd.DataFrame(columns=["Tipo Frota", "Usado"])

    saldo = (
        saldo_final.groupby(["Tipo Frota"], as_index=False)["Disponibilidade_Restante"].sum().rename(columns={"Disponibilidade_Restante": "Saldo"})
    ) if saldo_final is not None and not saldo_final.empty else pd.DataFrame(columns=["Tipo Frota", "Saldo"])

    resumo_frota = (oferta.merge(usado, on="Tipo Frota", how="outer").merge(saldo, on="Tipo Frota", how="outer").fillna(0))
    resumo_frota["Utilizacao_%"] = resumo_frota.apply(lambda r: _safe_pct(r.get("Usado", 0), r.get("Oferta", 0)), axis=1)
    analyses["Resumo_Frota"] = resumo_frota.sort_values(["Tipo Frota"], ascending=True)

    oferta_cls = (plan_common.groupby(["Tipo Frota", "vehicle_class"], as_index=False)["Disponibilidade"].sum().rename(columns={"Disponibilidade": "Oferta"}))
    usado_cls = (
        used_rows.loc[~used_rows["Transportadora"].astype(str).str.contains(r"\(SEM OFERTA\)", regex=True), :]
        .groupby(["Tipo Frota", "vehicle_class"], as_index=False)["Veiculos"].sum().rename(columns={"Veiculos": "Usado"})
    ) if not used_rows.empty else pd.DataFrame(columns=["Tipo Frota", "vehicle_class", "Usado"])

    saldo_cls = pd.DataFrame(columns=["Tipo Frota", "vehicle_class", "Saldo"])
    if saldo_final is not None and not saldo_final.empty:
        tmp = saldo_final.copy()
        tmp["vehicle_class"] = tmp["Modal"].map(vehicle_class)
        saldo_cls = (tmp.groupby(["Tipo Frota", "vehicle_class"], as_index=False)["Disponibilidade_Restante"].sum().rename(columns={"Disponibilidade_Restante": "Saldo"}))

    resumo_cls = (oferta_cls.merge(usado_cls, on=["Tipo Frota", "vehicle_class"], how="outer")
                  .merge(saldo_cls, on=["Tipo Frota", "vehicle_class"], how="outer").fillna(0))
    resumo_cls["Utilizacao_%"] = resumo_cls.apply(lambda r: _safe_pct(r.get("Usado", 0), r.get("Oferta", 0)), axis=1)
    analyses["Resumo_Classe"] = resumo_cls.sort_values(["Tipo Frota", "vehicle_class"], ascending=True)

    analyses["Uso_Cluster_Frota"] = (
        used_rows.groupby(["Cluster", "Tipo Frota"], as_index=False)["Veiculos"].sum().sort_values(["Cluster", "Tipo Frota"], ascending=True)
    ) if not used_rows.empty else pd.DataFrame(columns=["Cluster", "Tipo Frota", "Veiculos"])

    analyses["Uso_HUB_Frota"] = (
        used_rows.groupby(["HUB", "Tipo Frota"], as_index=False)["Veiculos"].sum().sort_values(["HUB", "Tipo Frota"], ascending=True)
    ) if not used_rows.empty else pd.DataFrame(columns=["HUB", "Tipo Frota", "Veiculos"])

    oferta_car = plan_common.groupby(["Transportadora", "Tipo Frota"], as_index=False)["Disponibilidade"].sum().rename(columns={"Disponibilidade": "Oferta"})
    usado_car = (
        used_rows.loc[~used_rows["Transportadora"].astype(str).str.contains(r"\(SEM OFERTA\)", regex=True), :]
        .groupby(["Transportadora", "Tipo Frota"], as_index=False)["Veiculos"].sum().rename(columns={"Veiculos": "Usado"})
    ) if not used_rows.empty else pd.DataFrame(columns=["Transportadora", "Tipo Frota", "Usado"])

    saldo_car = pd.DataFrame(columns=["Transportadora", "Tipo Frota", "Saldo"])
    if saldo_final is not None and not saldo_final.empty:
        saldo_car = saldo_final.groupby(["Transportadora", "Tipo Frota"], as_index=False)["Disponibilidade_Restante"].sum().rename(columns={"Disponibilidade_Restante": "Saldo"})

    dist_car = oferta_car.merge(usado_car, on=["Transportadora", "Tipo Frota"], how="outer").merge(saldo_car, on=["Transportadora", "Tipo Frota"], how="outer").fillna(0)
    dist_car_tot = dist_car.groupby(["Tipo Frota"], as_index=False)[["Oferta", "Usado", "Saldo"]].sum().rename(columns={"Oferta": "Oferta_Total", "Usado": "Usado_Total", "Saldo": "Saldo_Total"})
    dist_car = dist_car.merge(dist_car_tot, on="Tipo Frota", how="left")
    dist_car["Oferta_%"] = dist_car.apply(lambda r: _safe_pct(r.get("Oferta", 0), r.get("Oferta_Total", 0)), axis=1)
    dist_car["Uso_%"] = dist_car.apply(lambda r: _safe_pct(r.get("Usado", 0), r.get("Usado_Total", 0)), axis=1)
    dist_car["Delta_pp"] = (dist_car["Uso_%"] - dist_car["Oferta_%"]) * 100
    analyses["Distribuicao_Transportadora"] = dist_car.sort_values(["Tipo Frota", "Oferta"], ascending=[True, False])

    analyses["Uso_Cluster_Transportadora"] = (
        used_rows.loc[~used_rows["Transportadora"].astype(str).str.contains(r"\(SEM OFERTA\)", regex=True), :]
        .groupby(["Cluster", "Transportadora"], as_index=False)["Veiculos"].sum().sort_values(["Cluster", "Veiculos"], ascending=[True, False])
    ) if not used_rows.empty else pd.DataFrame(columns=["Cluster", "Transportadora", "Veiculos"])

    sinergia = pd.DataFrame(columns=["Grupo_Sinergia", "Cluster", "Cluster_Oferta", "Tipo Frota", "vehicle_class", "Veiculos"])
    if debug_alloc is not None and not debug_alloc.empty and "Cluster_Oferta" in debug_alloc.columns:
        tmp = debug_alloc.copy()
        tmp["vehicle_class"] = tmp["Modal"].map(vehicle_class)
        tmp = tmp[(tmp["Cluster_Oferta"].astype(str) != tmp["Cluster"].astype(str))].copy()
        tmp = tmp[tmp["Tipo Frota"].astype(str).str.upper().str.strip() != "KANGU"]
        if not tmp.empty:
            sinergia = (tmp.groupby(["Grupo_Sinergia", "Cluster", "Cluster_Oferta", "Tipo Frota", "vehicle_class"], as_index=False)["Veiculos"].sum()
                        .sort_values(["Grupo_Sinergia", "Cluster", "Veiculos"], ascending=[True, True, False]))
    analyses["Sinergia_Emprestimos"] = sinergia

    prop = pd.DataFrame(columns=["Grupo_Sinergia", "Tipo Frota", "vehicle_class", "Transportadora", "Oferta", "Usado", "Oferta_%", "Uso_%", "Delta_pp"])
    if "Grupo_Sinergia" in plan_common.columns and not plan_common.empty and not used_rows.empty:
        oferta_b = (plan_common.groupby(["Grupo_Sinergia", "Tipo Frota", "vehicle_class", "Transportadora"], as_index=False)["Disponibilidade"].sum()
                    .rename(columns={"Disponibilidade": "Oferta"}))
        usado_b = debug_alloc.loc[~debug_alloc["Transportadora"].astype(str).str.contains(r"\(SEM OFERTA\)", regex=True), :].copy() if debug_alloc is not None and not debug_alloc.empty else pd.DataFrame()
        if not usado_b.empty:
            usado_b["vehicle_class"] = usado_b["Modal"].map(vehicle_class)
            usado_b = (usado_b.groupby(["Grupo_Sinergia", "Tipo Frota", "vehicle_class", "Transportadora"], as_index=False)["Veiculos"].sum()
                       .rename(columns={"Veiculos": "Usado"}))
            prop = oferta_b.merge(usado_b, on=["Grupo_Sinergia", "Tipo Frota", "vehicle_class", "Transportadora"], how="outer").fillna(0)
            totals = prop.groupby(["Grupo_Sinergia", "Tipo Frota", "vehicle_class"], as_index=False)[["Oferta", "Usado"]].sum().rename(columns={"Oferta": "Oferta_Total", "Usado": "Usado_Total"})
            prop = prop.merge(totals, on=["Grupo_Sinergia", "Tipo Frota", "vehicle_class"], how="left")
            prop["Oferta_%"] = prop.apply(lambda r: _safe_pct(r.get("Oferta", 0), r.get("Oferta_Total", 0)), axis=1)
            prop["Uso_%"] = prop.apply(lambda r: _safe_pct(r.get("Usado", 0), r.get("Usado_Total", 0)), axis=1)
            prop["Delta_pp"] = (prop["Uso_%"] - prop["Oferta_%"]) * 100
            prop = prop.sort_values(["Grupo_Sinergia", "Tipo Frota", "vehicle_class", "Oferta"], ascending=[True, True, True, False])
    analyses["Proporcionalidade_Bucket"] = prop

    return analyses


def to_excel_bytes_multi(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = str(name)[:31]
            (df if df is not None else pd.DataFrame()).to_excel(writer, sheet_name=safe_name, index=False)
    buf.seek(0)
    return buf.read()


# =========================
# ✅ NOVO BLOCO: DEMANDA (ISsDia) x OUTPUT x PLANOROTAS
# =========================
def _add_caps_to_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquecer output_consolidado/debug_alloc com capacidade efetiva por linha
    (m3/kg) de acordo com o Modal.
    """
    if df is None:
        return pd.DataFrame()

    if df.empty:
        tmp = df.copy()
        tmp["cap_m3_eff"] = []
        tmp["cap_kg_eff"] = []
        return tmp

    tmp = df.copy()
    caps = tmp["Modal"].map(lambda m: capacity_for_modal(m))
    tmp["cap_m3"] = [c[0] for c in caps]
    tmp["cap_kg"] = [c[1] for c in caps]
    tmp["cap_m3_eff"] = tmp["cap_m3"] * OCCUPANCY_M3
    tmp["cap_kg_eff"] = tmp["cap_kg"] * OCCUPANCY_KG
    return tmp


def build_demand_vs_output_vs_plan(
    isdata: pd.DataFrame,
    output_final: pd.DataFrame,
    plan_common: pd.DataFrame,
    saldo_final: pd.DataFrame,
) -> dict:
    """
    Tabelas para checar:
    - Demanda (ISsDia) em m3/kg
    - Capacidade alocada (output) em m3/kg
    - Oferta (PlanoRotas) e Saldo (saldo_plano)
    - Flag de falta de disponibilidade (SEM OFERTA / gaps positivos)
    """
    analyses = {}

    if isdata is None or isdata.empty:
        analyses["Demanda_vs_Capacidade_Cluster"] = pd.DataFrame()
        analyses["Demanda_vs_Capacidade_HUB"] = pd.DataFrame()
        analyses["Faltas_Resumo_Cluster"] = pd.DataFrame()
        return analyses

    # -------------------------
    # Demanda por Cluster e por HUB
    # -------------------------
    dem_cluster = (
        isdata.groupby(["Cluster"], as_index=False)
        .agg(Demanda_m3=("Volume_m3", "sum"), Demanda_kg=("Peso_kg", "sum"), ISs=("Volume_m3", "size"))
    )

    dem_hub = (
        isdata.groupby(["Cluster", "HUB"], as_index=False)
        .agg(Demanda_m3=("Volume_m3", "sum"), Demanda_kg=("Peso_kg", "sum"), ISs=("Volume_m3", "size"))
    )

    # -------------------------
    # Output: capacidade alocada
    # -------------------------
    out = output_final.copy() if output_final is not None else pd.DataFrame()
    if out.empty:
        out = pd.DataFrame(columns=["Cluster", "HUB", "Transportadora", "Tipo Frota", "Modal", "Veiculos"])

    out_caps = _add_caps_to_output(out)
    out_caps["Sem_Oferta"] = out_caps["Transportadora"].astype(str).str.contains(r"\(SEM OFERTA\)", regex=True)

    out_caps["CapAloc_m3"] = out_caps["Veiculos"] * out_caps["cap_m3_eff"].fillna(0.0)
    out_caps["CapAloc_kg"] = out_caps["Veiculos"] * out_caps["cap_kg_eff"].fillna(0.0)

    cap_cluster = (
        out_caps.groupby(["Cluster"], as_index=False)
        .agg(
            Veiculos_Alocados=("Veiculos", "sum"),
            Veiculos_SemOferta=("Sem_Oferta", "sum"),
            Capacidade_Aloc_m3=("CapAloc_m3", "sum"),
            Capacidade_Aloc_kg=("CapAloc_kg", "sum"),
        )
    )

    cap_hub = (
        out_caps.groupby(["Cluster", "HUB"], as_index=False)
        .agg(
            Veiculos_Alocados=("Veiculos", "sum"),
            Veiculos_SemOferta=("Sem_Oferta", "sum"),
            Capacidade_Aloc_m3=("CapAloc_m3", "sum"),
            Capacidade_Aloc_kg=("CapAloc_kg", "sum"),
        )
    )

    # -------------------------
    # PlanoRotas: oferta + saldo (por Cluster)
    # -------------------------
    if plan_common is None or plan_common.empty:
        oferta_cluster = pd.DataFrame(columns=["Cluster", "Oferta_PlanoRotas"])
    else:
        oferta_cluster = (
            plan_common.groupby(["Cluster"], as_index=False)["Disponibilidade"]
            .sum()
            .rename(columns={"Disponibilidade": "Oferta_PlanoRotas"})
        )

    if saldo_final is None or saldo_final.empty:
        saldo_cluster = pd.DataFrame(columns=["Cluster", "Saldo_PlanoRotas"])
    else:
        saldo_cluster = (
            saldo_final.groupby(["Cluster"], as_index=False)["Disponibilidade_Restante"]
            .sum()
            .rename(columns={"Disponibilidade_Restante": "Saldo_PlanoRotas"})
        )

    # -------------------------
    # Tabela Cluster: Demanda x Capacidade x Plano
    # -------------------------
    cluster_tbl = (
        dem_cluster.merge(cap_cluster, on="Cluster", how="left")
        .merge(oferta_cluster, on="Cluster", how="left")
        .merge(saldo_cluster, on="Cluster", how="left")
        .fillna(0)
    )

    cluster_tbl["Gap_m3"] = cluster_tbl["Demanda_m3"] - cluster_tbl["Capacidade_Aloc_m3"]
    cluster_tbl["Gap_kg"] = cluster_tbl["Demanda_kg"] - cluster_tbl["Capacidade_Aloc_kg"]

    cluster_tbl["Falta_Disponibilidade"] = (
        (cluster_tbl["Veiculos_SemOferta"] > 0)
        | (cluster_tbl["Gap_m3"] > 1e-6)
        | (cluster_tbl["Gap_kg"] > 1e-6)
    )

    # -------------------------
    # Tabela HUB: Demanda x Capacidade (Plano não é por HUB)
    # -------------------------
    hub_tbl = (
        dem_hub.merge(cap_hub, on=["Cluster", "HUB"], how="left")
        .fillna(0)
    )

    hub_tbl["Gap_m3"] = hub_tbl["Demanda_m3"] - hub_tbl["Capacidade_Aloc_m3"]
    hub_tbl["Gap_kg"] = hub_tbl["Demanda_kg"] - hub_tbl["Capacidade_Aloc_kg"]
    hub_tbl["Falta_Disponibilidade"] = (
        (hub_tbl["Veiculos_SemOferta"] > 0)
        | (hub_tbl["Gap_m3"] > 1e-6)
        | (hub_tbl["Gap_kg"] > 1e-6)
    )

    # -------------------------
    # Resumo executivo de falta
    # -------------------------
    faltas = cluster_tbl[cluster_tbl["Falta_Disponibilidade"]].copy()
    faltas = faltas.sort_values(["Veiculos_SemOferta", "Gap_m3", "Gap_kg"], ascending=[False, False, False])

    analyses["Demanda_vs_Capacidade_Cluster"] = cluster_tbl.sort_values(["Falta_Disponibilidade", "Demanda_m3"], ascending=[False, False])
    analyses["Demanda_vs_Capacidade_HUB"] = hub_tbl.sort_values(["Falta_Disponibilidade", "Demanda_m3"], ascending=[False, False])
    analyses["Faltas_Resumo_Cluster"] = faltas

    return analyses


# =========================
# GERAÇÃO DE RELATÓRIO PDF
# =========================
def generate_pdf_report(
    output_final: pd.DataFrame,
    saldo_final: pd.DataFrame,
    analyses: dict,
    isdata: pd.DataFrame,
    occupancy_m3: float,
    occupancy_kg: float,
) -> bytes:
    """Gera um relatório PDF completo com métricas, gráficos e tabelas."""
    import io as _io
    from datetime import datetime
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak, Image as RLImage,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    # ── Paleta ────────────────────────────────────────────
    NAVY   = colors.HexColor("#1E2761")
    TEAL   = colors.HexColor("#028090")
    ACCENT = colors.HexColor("#02C39A")
    AMBER  = colors.HexColor("#F59E0B")
    CORAL  = colors.HexColor("#EF4444")
    LGRAY  = colors.HexColor("#F0F4FF")
    GRAY   = colors.HexColor("#64748B")
    WHITE  = colors.white

    buf = _io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
    )
    W = A4[0] - 3.6*cm  # largura útil

    # ── Estilos ───────────────────────────────────────────
    styles = getSampleStyleSheet()
    sTitle    = ParagraphStyle("sTitle",    fontSize=20, textColor=WHITE,     alignment=TA_LEFT,   fontName="Helvetica-Bold", leading=26)
    sSub      = ParagraphStyle("sSub",      fontSize=10, textColor=ACCENT,    alignment=TA_LEFT,   fontName="Helvetica")
    sDate     = ParagraphStyle("sDate",     fontSize=8,  textColor=WHITE,     alignment=TA_RIGHT,  fontName="Helvetica")
    sH2       = ParagraphStyle("sH2",       fontSize=12, textColor=NAVY,      fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4)
    sH3       = ParagraphStyle("sH3",       fontSize=10, textColor=TEAL,      fontName="Helvetica-Bold", spaceBefore=8,  spaceAfter=2)
    sNormal   = ParagraphStyle("sNormal",   fontSize=8,  textColor=GRAY,      fontName="Helvetica", leading=12)
    sAlert    = ParagraphStyle("sAlert",    fontSize=8,  textColor=CORAL,     fontName="Helvetica-Bold")
    sOk       = ParagraphStyle("sOk",       fontSize=8,  textColor=ACCENT,    fontName="Helvetica-Bold")
    sMetricV  = ParagraphStyle("sMetricV",  fontSize=18, textColor=NAVY,      fontName="Helvetica-Bold", alignment=TA_CENTER)
    sMetricL  = ParagraphStyle("sMetricL",  fontSize=7,  textColor=GRAY,      fontName="Helvetica",      alignment=TA_CENTER)
    sTH       = ParagraphStyle("sTH",       fontSize=7,  textColor=WHITE,     fontName="Helvetica-Bold", alignment=TA_CENTER)
    sTD       = ParagraphStyle("sTD",       fontSize=7,  textColor=GRAY,      fontName="Helvetica",      alignment=TA_CENTER)
    sTDL      = ParagraphStyle("sTDL",      fontSize=7,  textColor=GRAY,      fontName="Helvetica",      alignment=TA_LEFT)

    story = []

    # ── Helper: gráfico matplotlib → imagem ─────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    def bar_chart_image(labels, values, title, color="#028090", width_cm=16, height_cm=7):
        fig, ax = plt.subplots(figsize=(width_cm/2.54, height_cm/2.54))
        bars = ax.bar(labels, values, color=color, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=10, color="#1E2761", fontweight="bold", pad=8)
        ax.set_ylabel("Veículos", fontsize=8, color="#64748B")
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    str(int(val)), ha="center", va="bottom", fontsize=7, color="#1E2761", fontweight="bold")
        plt.tight_layout()
        buf_img = _io.BytesIO()
        fig.savefig(buf_img, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf_img.seek(0)
        return RLImage(buf_img, width=width_cm*cm, height=height_cm*cm)

    def grouped_bar_image(df_melt, x_col, y_col, color_col, title, color_map=None, width_cm=16, height_cm=7):
        groups = df_melt[x_col].unique()
        series = df_melt[color_col].unique()
        x = np.arange(len(groups))
        width = 0.35
        fig, ax = plt.subplots(figsize=(width_cm/2.54, height_cm/2.54))
        default_colors = ["#2E4DA3", "#02C39A", "#F59E0B", "#EF4444"]
        for i, s in enumerate(series):
            vals = [df_melt[(df_melt[x_col]==g) & (df_melt[color_col]==s)][y_col].values[0]
                    if len(df_melt[(df_melt[x_col]==g) & (df_melt[color_col]==s)]) > 0 else 0
                    for g in groups]
            c = (color_map or {}).get(s, default_colors[i % len(default_colors)])
            bars = ax.bar(x + i*width - width*(len(series)-1)/2, vals, width, label=s, color=c, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            str(int(val)), ha="center", va="bottom", fontsize=6, color="#1E2761")
        ax.set_title(title, fontsize=10, color="#1E2761", fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=7, rotation=20)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#F8FAFC")
        fig.patch.set_facecolor("white")
        ax.legend(fontsize=7)
        plt.tight_layout()
        buf_img = _io.BytesIO()
        fig.savefig(buf_img, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf_img.seek(0)
        return RLImage(buf_img, width=width_cm*cm, height=height_cm*cm)

    # ── Helper: tabela estilizada ─────────────────────────
    def make_table(df, col_widths=None, max_rows=20):
        if df is None or df.empty:
            return Paragraph("Sem dados disponíveis.", sNormal)
        df_show = df.head(max_rows).copy()
        # Formata floats
        for c in df_show.select_dtypes(include="float").columns:
            df_show[c] = df_show[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        headers = [Paragraph(str(c), sTH) for c in df_show.columns]
        rows = [[Paragraph(str(v), sTDL if i == 0 else sTD) for i, v in enumerate(row)] for row in df_show.itertuples(index=False)]
        data = [headers] + rows
        if col_widths is None:
            col_widths = [W / len(df_show.columns)] * len(df_show.columns)
        ts = TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LGRAY, WHITE]),
            ("GRID",        (0, 0), (-1, -1),  0.3, colors.HexColor("#D0DAFF")),
            ("TOPPADDING",  (0, 0), (-1, -1),  3),
            ("BOTTOMPADDING",(0,0), (-1, -1),  3),
            ("LEFTPADDING", (0, 0), (-1, -1),  4),
            ("RIGHTPADDING",(0, 0), (-1, -1),  4),
            ("VALIGN",      (0, 0), (-1, -1),  "MIDDLE"),
        ])
        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(ts)
        return t

    # ══════════════════════════════════════════════════════
    # CAPA
    # ══════════════════════════════════════════════════════
    # Faixa de cabeçalho
    header_data = [[
        Paragraph("Relatório de Alocação de Veículos por Cluster", sTitle),
        Paragraph(datetime.now().strftime("%d/%m/%Y %H:%M"), sDate),
    ]]
    header_table = Table(header_data, colWidths=[W * 0.75, W * 0.25])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("LEFTPADDING",  (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING",   (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(header_table)

    # Subtítulo
    sub_data = [[Paragraph("Plano de Rotas × ISs do Dia — alocação automática por cluster e HUB", sSub)]]
    sub_table = Table(sub_data, colWidths=[W])
    sub_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), TEAL),
        ("LEFTPADDING",  (0, 0), (-1, -1), 14),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    story.append(sub_table)
    story.append(Spacer(1, 0.4*cm))

    # ══════════════════════════════════════════════════════
    # MÉTRICAS EXECUTIVAS
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Resumo Executivo", sH2))
    story.append(HRFlowable(width=W, thickness=2, color=TEAL, spaceAfter=6))

    resumo_frota = analyses.get("Resumo_Frota", pd.DataFrame())
    faltas_df    = analyses.get("Faltas_Resumo_Cluster", pd.DataFrame())
    sinergia_df  = analyses.get("Sinergia_Emprestimos", pd.DataFrame())

    total_oferta   = int(resumo_frota["Oferta"].sum())  if not resumo_frota.empty and "Oferta"  in resumo_frota.columns else 0
    total_usado    = int(resumo_frota["Usado"].sum())   if not resumo_frota.empty and "Usado"   in resumo_frota.columns else 0
    total_saldo    = int(resumo_frota["Saldo"].sum())   if not resumo_frota.empty and "Saldo"   in resumo_frota.columns else 0
    n_faltas       = len(faltas_df) if not faltas_df.empty else 0
    n_sinergia     = int(sinergia_df["Veiculos"].sum()) if not sinergia_df.empty and "Veiculos" in sinergia_df.columns else 0
    util_pct       = f"{total_usado/total_oferta*100:.1f}%" if total_oferta > 0 else "—"
    n_clusters     = output_final["Cluster"].nunique() if not output_final.empty else 0
    n_hubs         = output_final["HUB"].nunique()     if not output_final.empty else 0

    metrics = [
        (str(total_alocado := total_usado), "Veículos alocados"),
        (util_pct,                          "Utilização geral"),
        (str(total_oferta),                 "Oferta total"),
        (str(total_saldo),                  "Saldo restante"),
        (str(n_clusters),                   "Clusters atendidos"),
        (str(n_hubs),                       "HUBs envolvidos"),
        (str(n_sinergia),                   "Veículos em sinergia"),
        (str(n_faltas),                     "Clusters com falta"),
    ]
    metric_cells = []
    for val, label in metrics:
        cell = Table(
            [[Paragraph(val, sMetricV)], [Paragraph(label, sMetricL)]],
            colWidths=[W/8]
        )
        cell.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), LGRAY),
            ("TOPPADDING",   (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ("LEFTPADDING",  (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("BOX",          (0, 0), (-1, -1), 1, colors.HexColor("#D0DAFF")),
        ]))
        metric_cells.append(cell)

    metrics_row = Table([metric_cells], colWidths=[W/8]*8)
    metrics_row.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),2), ("RIGHTPADDING",(0,0),(-1,-1),2)]))
    story.append(metrics_row)
    story.append(Spacer(1, 0.3*cm))

    # Config usada
    config_text = f"Configuração: Ocupação m³ = {occupancy_m3:.0%} | Ocupação kg = {occupancy_kg:.0%} | MIN_MEDIO oversize ≥ {MIN_MEDIO_OVERSIZE_M3} m³ ou ≥ {MIN_MEDIO_OVERSIZE_KG} kg"
    story.append(Paragraph(config_text, sNormal))

    # Alerta de faltas
    if n_faltas > 0:
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(f"⚠ ATENÇÃO: {n_faltas} cluster(s) com falta de oferta — verifique a aba Diagnóstico.", sAlert))
    else:
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("✓ Todos os clusters foram atendidos sem falta de oferta.", sOk))

    story.append(Spacer(1, 0.4*cm))

    # ══════════════════════════════════════════════════════
    # GRÁFICO 1 — Veículos alocados por HUB
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Distribuição por HUB", sH2))
    story.append(HRFlowable(width=W, thickness=2, color=TEAL, spaceAfter=6))

    uso_hub = analyses.get("Uso_HUB_Frota", pd.DataFrame())
    if not uso_hub.empty:
        hub_tot = uso_hub.groupby("HUB", as_index=False)["Veiculos"].sum().sort_values("Veiculos", ascending=False)
        story.append(bar_chart_image(
            hub_tot["HUB"].tolist(), hub_tot["Veiculos"].tolist(),
            "Veículos alocados por HUB", color="#028090", width_cm=16, height_cm=7,
        ))
        story.append(Spacer(1, 0.2*cm))

        # Top 3 HUBs
        story.append(Paragraph("Top HUBs por volume alocado:", sH3))
        top3 = hub_tot.head(3)
        for _, row in top3.iterrows():
            pct = row["Veiculos"] / hub_tot["Veiculos"].sum() * 100
            story.append(Paragraph(f"  • <b>{row['HUB']}</b> — {int(row['Veiculos'])} veículos ({pct:.1f}% do total)", sNormal))
        story.append(Spacer(1, 0.3*cm))

    # ══════════════════════════════════════════════════════
    # GRÁFICO 2 — Utilização por Tipo de Frota
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Utilização por Tipo de Frota", sH2))
    story.append(HRFlowable(width=W, thickness=2, color=TEAL, spaceAfter=6))

    if not resumo_frota.empty and "Oferta" in resumo_frota.columns:
        df_melt = resumo_frota[["Tipo Frota", "Oferta", "Usado"]].melt(
            id_vars="Tipo Frota", var_name="Métrica", value_name="Qtd"
        )
        story.append(grouped_bar_image(
            df_melt, x_col="Tipo Frota", y_col="Qtd", color_col="Métrica",
            title="Oferta vs Utilizado por Tipo de Frota",
            color_map={"Oferta": "#2E4DA3", "Usado": "#02C39A"},
            width_cm=16, height_cm=7,
        ))
        story.append(Spacer(1, 0.2*cm))

        # Tabela resumo frota
        story.append(Paragraph("Resumo por Tipo de Frota", sH3))
        cols_frota = ["Tipo Frota", "Oferta", "Usado", "Saldo", "Utilizacao_%"]
        df_frota_show = resumo_frota[[c for c in cols_frota if c in resumo_frota.columns]].copy()
        if "Utilizacao_%" in df_frota_show.columns:
            df_frota_show["Utilizacao_%"] = (df_frota_show["Utilizacao_%"] * 100).round(1).astype(str) + "%"
        cw_frota = [W*0.3, W*0.17, W*0.17, W*0.17, W*0.19][:len(df_frota_show.columns)]
        story.append(make_table(df_frota_show, col_widths=cw_frota))
        story.append(Spacer(1, 0.3*cm))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # GRÁFICO 3 — Distribuição por Classe de Veículo
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Distribuição por Classe de Veículo", sH2))
    story.append(HRFlowable(width=W, thickness=2, color=TEAL, spaceAfter=6))

    resumo_cls = analyses.get("Resumo_Classe", pd.DataFrame())
    if not resumo_cls.empty and "Usado" in resumo_cls.columns:
        cls_tot = resumo_cls.groupby("vehicle_class", as_index=False)["Usado"].sum()
        cls_tot = cls_tot[cls_tot["Usado"] > 0].sort_values("Usado", ascending=False)
        colors_cls = ["#028090","#2E4DA3","#F59E0B","#EF4444","#10B981","#534AB7","#64748B","#02C39A"]
        story.append(bar_chart_image(
            cls_tot["vehicle_class"].tolist(), cls_tot["Usado"].tolist(),
            "Veículos alocados por Classe",
            color=colors_cls[0], width_cm=16, height_cm=6.5,
        ))
        story.append(Spacer(1, 0.3*cm))

    # ══════════════════════════════════════════════════════
    # TABELA — Distribuição por Transportadora
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Distribuição por Transportadora", sH2))
    story.append(HRFlowable(width=W, thickness=2, color=TEAL, spaceAfter=6))

    dist_car = analyses.get("Distribuicao_Transportadora", pd.DataFrame())
    if not dist_car.empty:
        cols_car = ["Transportadora", "Tipo Frota", "Oferta", "Usado", "Saldo", "Oferta_%", "Uso_%", "Delta_pp"]
        df_car_show = dist_car[[c for c in cols_car if c in dist_car.columns]].copy()
        for c in ["Oferta_%", "Uso_%"]:
            if c in df_car_show.columns:
                df_car_show[c] = (df_car_show[c] * 100).round(1).astype(str) + "%"
        if "Delta_pp" in df_car_show.columns:
            df_car_show["Delta_pp"] = df_car_show["Delta_pp"].round(1).astype(str) + "pp"
        n_cols = len(df_car_show.columns)
        cw = [W / n_cols] * n_cols
        cw[0] = W * 0.22
        story.append(make_table(df_car_show, col_widths=cw, max_rows=30))
        story.append(Spacer(1, 0.3*cm))

    # ══════════════════════════════════════════════════════
    # TABELA — Sinergia entre Clusters
    # ══════════════════════════════════════════════════════
    if not sinergia_df.empty:
        story.append(Paragraph("Sinergia — Empréstimos entre Clusters", sH2))
        story.append(HRFlowable(width=W, thickness=2, color=AMBER, spaceAfter=6))
        story.append(Paragraph("Veículos alocados de clusters vizinhos (mesmo grupo de sinergia):", sNormal))
        story.append(Spacer(1, 0.15*cm))
        cols_sin = ["Grupo_Sinergia", "Cluster", "Cluster_Oferta", "Tipo Frota", "vehicle_class", "Veiculos"]
        df_sin = sinergia_df[[c for c in cols_sin if c in sinergia_df.columns]]
        cw_sin = [W*0.18, W*0.16, W*0.18, W*0.14, W*0.14, W*0.14][:len(df_sin.columns)]
        story.append(make_table(df_sin, col_widths=cw_sin, max_rows=25))
        story.append(Spacer(1, 0.3*cm))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # TABELA — Diagnóstico de Faltas
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Diagnóstico de Faltas de Oferta", sH2))
    story.append(HRFlowable(width=W, thickness=2, color=CORAL, spaceAfter=6))

    if n_faltas == 0:
        story.append(Paragraph("✓ Nenhum cluster com falta de oferta nesta rodada.", sOk))
    else:
        story.append(Paragraph(f"Os {n_faltas} cluster(s) abaixo tiveram demanda superior à oferta disponível:", sAlert))
        story.append(Spacer(1, 0.15*cm))
        cols_falt = ["Cluster", "Demanda_m3", "Demanda_kg", "Capacidade_Aloc_m3", "Capacidade_Aloc_kg", "Veiculos_SemOferta", "Gap_m3", "Gap_kg"]
        df_falt = faltas_df[[c for c in cols_falt if c in faltas_df.columns]]
        cw_falt = [W*0.18] + [W*0.82/(len(df_falt.columns)-1)]*(len(df_falt.columns)-1)
        story.append(make_table(df_falt, col_widths=cw_falt))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Gap positivo indica que a capacidade alocada foi insuficiente para cobrir a demanda.", sNormal))

    story.append(Spacer(1, 0.4*cm))

    # ══════════════════════════════════════════════════════
    # TABELA — Output Consolidado (primeiras 40 linhas)
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Output Consolidado (primeiras 40 linhas)", sH2))
    story.append(HRFlowable(width=W, thickness=2, color=TEAL, spaceAfter=6))
    story.append(Paragraph("Para o output completo, utilize o download em Excel ou CSV.", sNormal))
    story.append(Spacer(1, 0.15*cm))

    if not output_final.empty:
        cols_out = ["Cluster", "HUB", "Transportadora", "Tipo Frota", "Modal", "Veiculos"]
        df_out = output_final[[c for c in cols_out if c in output_final.columns]]
        cw_out = [W*0.16, W*0.10, W*0.22, W*0.14, W*0.26, W*0.12][:len(df_out.columns)]
        story.append(make_table(df_out, col_widths=cw_out, max_rows=40))

    # ══════════════════════════════════════════════════════
    # RODAPÉ
    # ══════════════════════════════════════════════════════
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width=W, thickness=1, color=TEAL, spaceAfter=4))
    footer_data = [[
        Paragraph("Gerado automaticamente pelo App de Alocação de Veículos por Cluster", sNormal),
        Paragraph(datetime.now().strftime("Gerado em %d/%m/%Y às %H:%M"), ParagraphStyle("r", fontSize=7, textColor=GRAY, alignment=TA_RIGHT, fontName="Helvetica")),
    ]]
    footer_table = Table(footer_data, colWidths=[W*0.65, W*0.35])
    footer_table.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
    story.append(footer_table)

    doc.build(story)
    buf.seek(0)
    return buf.read()


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Alocação por Cluster", layout="wide", page_icon="🚛")

st.markdown("""
<style>
    .block-container { padding-top: 1.4rem !important; }
    .stDownloadButton > button { width: 100%; border-radius: 8px; font-weight: 500; }
    div[data-testid="metric-container"] {
        background: #F0F4FF;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        border-left: 4px solid #028090;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🚛 Alocação de Veículos por Cluster")
st.caption("Plano de Rotas × ISs do Dia — alocação automática por cluster e HUB")


@st.cache_data
def load_excel(file) -> pd.DataFrame:
    return pd.read_excel(file)


def preview_columns(df: pd.DataFrame, label: str, col_map: dict):
    """Mostra na sidebar quais colunas foram detectadas no arquivo."""
    st.caption(f"**{label} — colunas detectadas:**")
    for name, col in col_map.items():
        if col:
            st.caption(f"✅ {name} → `{col}`")
        else:
            st.caption(f"❌ {name} → não encontrada")


with st.sidebar:
    st.header("Upload dos arquivos")
    plan_file = st.file_uploader("PlanoRotas (Excel)", type=["xlsx"])
    is_file = st.file_uploader("ISsDia (Excel)", type=["xlsx"])

    st.divider()

    # Sliders de ocupação editáveis
    st.subheader("Configuração")
    occupancy_m3 = st.slider("Ocupação m³", 0.50, 1.00, OCCUPANCY_M3, 0.05, format="%.2f")
    occupancy_kg = st.slider("Ocupação kg", 0.50, 1.00, OCCUPANCY_KG, 0.05, format="%.2f")

    enable_synergy = st.checkbox(
        "Ativar sinergia entre clusters com mesmo prefixo",
        value=True,
    )

    st.divider()
    st.caption(f"MIN_MEDIO OVERSIZE: >= {MIN_MEDIO_OVERSIZE_M3} m³ ou >= {MIN_MEDIO_OVERSIZE_KG} kg")
    st.caption(f"HUB Elétrico: {ELECTRIC_HUB}")

    # Prévia das colunas detectadas após upload
    if plan_file:
        try:
            _prev_plan = load_excel(plan_file)
            preview_columns(_prev_plan, "PlanoRotas", {
                "Cluster": find_col(_prev_plan, ["Cluster"]),
                "Transportadora": find_col(_prev_plan, ["Transportadora", "Carrier", "Transporter"]),
                "Modal": find_col(_prev_plan, ["Modal", "Perfil"]),
                "Tipo Frota": find_col(_prev_plan, ["Tipo Frota", "Frota", "Fleet Type"]),
                "Disponibilidade": find_col(_prev_plan, ["Disponibilidade de Modais", "Disponibilidade", "Qtd", "Quantidade"]),
            })
        except Exception:
            pass

    if is_file:
        try:
            _prev_is = load_excel(is_file)
            preview_columns(_prev_is, "ISsDia", {
                "Cluster": find_col(_prev_is, ["CLUSTER", "Cluster"]),
                "HUB": find_col(_prev_is, ["HUB", "Warehouse", "WH", "WAREHOUSE_ID"]),
                "Peso": find_col(_prev_is, ["Peso(kg)", "Peso", "KG", "WEIGHT"]),
                "Volume": find_col(_prev_is, ["Volume(m³)", "Volume", "M3", "M³", "CUBAGEM"]),
            })
        except Exception:
            pass

run = st.button("Rodar alocação", type="primary", disabled=not (plan_file and is_file))

if run:
    try:
        # Aplica ocupação dos sliders sobrescrevendo as constantes no módulo
        import sys
        mod = sys.modules[__name__]
        mod.OCCUPANCY_M3 = occupancy_m3
        mod.OCCUPANCY_KG = occupancy_kg
        mod.VUC_BASE_M3_EFF = 16 * occupancy_m3
        mod.VUC_BASE_KG_EFF = 1800 * occupancy_kg
        mod.MEDIO_BASE_M3_EFF = 37 * occupancy_m3
        mod.MEDIO_BASE_KG_EFF = 3500 * occupancy_kg

        progress = st.progress(0, text="Lendo arquivos...")
        plan_df = load_excel(plan_file)
        is_df = load_excel(is_file)
        progress.progress(20, text="Processando alocação...")

        output_consolidado, saldo_plano, debug_alloc, saldo_debug, plan_common, isdata_norm = run_allocation(
            plan_df, is_df, enable_synergy=enable_synergy, return_debug=True
        )
        progress.progress(70, text="Montando análises...")

        analyses = build_analyses(output_consolidado, saldo_plano, debug_alloc, plan_common)
        demand_checks = build_demand_vs_output_vs_plan(
            isdata=isdata_norm,
            output_final=output_consolidado,
            plan_common=plan_common,
            saldo_final=saldo_plano,
        )
        analyses.update(demand_checks)
        progress.progress(90, text="Gerando relatório PDF...")
        pdf_bytes = generate_pdf_report(
            output_final=output_consolidado,
            saldo_final=saldo_plano,
            analyses=analyses,
            isdata=isdata_norm,
            occupancy_m3=occupancy_m3,
            occupancy_kg=occupancy_kg,
        )
        progress.progress(100, text="Concluído!")
        progress.empty()

        st.success("✅ Processamento concluído!")

        # =========================
        # MÉTRICAS EXECUTIVAS
        # =========================
        faltas_df = analyses.get("Faltas_Resumo_Cluster", pd.DataFrame())
        sinergia_df = analyses.get("Sinergia_Emprestimos", pd.DataFrame())
        resumo_frota_df = analyses.get("Resumo_Frota", pd.DataFrame())

        n_faltas = len(faltas_df) if not faltas_df.empty else 0
        n_sinergia = int(sinergia_df["Veiculos"].sum()) if not sinergia_df.empty and "Veiculos" in sinergia_df.columns else 0
        util_geral = 0.0
        total_alocados = 0
        if not resumo_frota_df.empty and "Oferta" in resumo_frota_df.columns:
            total_oferta = resumo_frota_df["Oferta"].sum()
            total_usado = resumo_frota_df["Usado"].sum() if "Usado" in resumo_frota_df.columns else 0
            total_alocados = int(total_usado)
            util_geral = (total_usado / total_oferta * 100) if total_oferta > 0 else 0.0

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("🚛 Veículos alocados", total_alocados)
        with m2:
            st.metric("📊 Utilização geral", f"{util_geral:.1f}%")
        with m3:
            st.metric("🔄 Em sinergia", n_sinergia)
        with m4:
            if n_faltas > 0:
                st.metric("⚠️ Clusters com falta", n_faltas)
            else:
                st.metric("✅ Clusters com falta", 0)

        st.divider()

        # =========================
        # DOWNLOADS NO TOPO
        # =========================
        sheets = {"output_consolidado": output_consolidado, "saldo_plano": saldo_plano, **analyses}
        excel_bytes = to_excel_bytes_multi(sheets)

        st.markdown("#### ⬇️ Downloads")
        dl1, dl2, dl3, dl4 = st.columns(4)
        with dl1:
            st.download_button(
                "📑 Relatório PDF",
                data=pdf_bytes,
                file_name="relatorio_alocacao.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        with dl2:
            st.download_button(
                "📥 Excel completo (todas as abas)",
                data=excel_bytes,
                file_name="output_alocacao_por_cluster.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with dl3:
            st.download_button(
                "📄 Output consolidado (CSV)",
                data=to_csv_bytes(output_consolidado),
                file_name="output_consolidado.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl4:
            st.download_button(
                "📄 Saldo do plano (CSV)",
                data=to_csv_bytes(saldo_plano),
                file_name="saldo_plano.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.divider()

        # =========================
        # ABAS PRINCIPAIS
        # =========================
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Resultado",
            "📈 Gráficos",
            "📊 Análises detalhadas",
            "⚠️ Diagnóstico de faltas",
        ])

        # ── ABA 1: RESULTADO ──────────────────────────────
        with tab1:
            st.markdown("##### Filtros")
            all_clusters = sorted(output_consolidado["Cluster"].astype(str).unique().tolist()) if not output_consolidado.empty else []
            all_hubs     = sorted(output_consolidado["HUB"].astype(str).unique().tolist()) if not output_consolidado.empty else []
            all_frotas   = sorted(output_consolidado["Tipo Frota"].astype(str).unique().tolist()) if not output_consolidado.empty else []

            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                sel_clusters = st.multiselect("Cluster", all_clusters, placeholder="Todos")
            with fc2:
                sel_hubs = st.multiselect("HUB", all_hubs, placeholder="Todos")
            with fc3:
                sel_frotas = st.multiselect("Tipo Frota", all_frotas, placeholder="Todas")

            out_filtrado = output_consolidado.copy() if not output_consolidado.empty else output_consolidado
            if sel_clusters:
                out_filtrado = out_filtrado[out_filtrado["Cluster"].astype(str).isin(sel_clusters)]
            if sel_hubs:
                out_filtrado = out_filtrado[out_filtrado["HUB"].astype(str).isin(sel_hubs)]
            if sel_frotas:
                out_filtrado = out_filtrado[out_filtrado["Tipo Frota"].astype(str).isin(sel_frotas)]

            r1, r2 = st.columns(2)
            with r1:
                label = "Output consolidado" + (" (filtrado)" if any([sel_clusters, sel_hubs, sel_frotas]) else "")
                st.markdown(f"##### {label}")
                st.dataframe(out_filtrado, use_container_width=True, hide_index=True, height=420)
            with r2:
                st.markdown("##### Saldo do plano (disponibilidade restante ≥ 1)")
                st.dataframe(saldo_plano, use_container_width=True, hide_index=True, height=420)

        # ── ABA 2: GRÁFICOS ───────────────────────────────
        with tab2:
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Oferta vs Usado por Tipo Frota**")
                if not resumo_frota_df.empty and "Oferta" in resumo_frota_df.columns:
                    df_melt = resumo_frota_df[["Tipo Frota", "Oferta", "Usado"]].melt(
                        id_vars="Tipo Frota", var_name="Métrica", value_name="Qtd"
                    )
                    fig = px.bar(df_melt, x="Tipo Frota", y="Qtd", color="Métrica", barmode="group",
                                 color_discrete_map={"Oferta": "#378ADD", "Usado": "#1D9E75"})
                    fig.update_layout(margin=dict(t=10, b=10), height=320, legend_title_text="")
                    st.plotly_chart(fig, use_container_width=True)

            with g2:
                st.markdown("**Utilização % por Tipo Frota**")
                if not resumo_frota_df.empty and "Utilizacao_%" in resumo_frota_df.columns:
                    fig2 = px.bar(resumo_frota_df, x="Tipo Frota", y="Utilizacao_%",
                                  color="Utilizacao_%",
                                  color_continuous_scale=["#E24B4A", "#EF9F27", "#1D9E75"],
                                  range_color=[0, 1])
                    fig2.update_layout(margin=dict(t=10, b=10), height=320, showlegend=False)
                    fig2.update_traces(texttemplate="%{y:.0%}", textposition="outside")
                    st.plotly_chart(fig2, use_container_width=True)

            g3, g4 = st.columns(2)
            with g3:
                st.markdown("**Veículos alocados por HUB**")
                uso_hub = analyses.get("Uso_HUB_Frota", pd.DataFrame())
                if not uso_hub.empty:
                    hub_tot = uso_hub.groupby("HUB", as_index=False)["Veiculos"].sum()
                    fig3 = px.bar(hub_tot, x="HUB", y="Veiculos", color_discrete_sequence=["#534AB7"])
                    fig3.update_layout(margin=dict(t=10, b=10), height=300)
                    st.plotly_chart(fig3, use_container_width=True)

            with g4:
                st.markdown("**Distribuição por Classe de Veículo**")
                resumo_cls_df = analyses.get("Resumo_Classe", pd.DataFrame())
                if not resumo_cls_df.empty and "Usado" in resumo_cls_df.columns:
                    cls_tot = resumo_cls_df.groupby("vehicle_class", as_index=False)["Usado"].sum()
                    cls_tot = cls_tot[cls_tot["Usado"] > 0]
                    fig4 = px.pie(cls_tot, names="vehicle_class", values="Usado",
                                  color_discrete_sequence=px.colors.qualitative.Safe)
                    fig4.update_layout(margin=dict(t=10, b=10), height=300)
                    st.plotly_chart(fig4, use_container_width=True)

        # ── ABA 3: ANÁLISES DETALHADAS ────────────────────
        with tab3:
            st.markdown("##### Resumo por Tipo de Frota")
            st.dataframe(analyses.get("Resumo_Frota"), use_container_width=True, hide_index=True)

            st.markdown("##### Resumo por Classe de Veículo")
            st.dataframe(analyses.get("Resumo_Classe"), use_container_width=True, hide_index=True)

            cA, cB = st.columns(2)
            with cA:
                st.markdown("##### Uso por Cluster × Tipo Frota")
                st.dataframe(analyses.get("Uso_Cluster_Frota"), use_container_width=True, hide_index=True)
            with cB:
                st.markdown("##### Uso por HUB × Tipo Frota")
                st.dataframe(analyses.get("Uso_HUB_Frota"), use_container_width=True, hide_index=True)

            st.markdown("##### Distribuição por Transportadora")
            st.dataframe(analyses.get("Distribuicao_Transportadora"), use_container_width=True, hide_index=True)

            st.markdown("##### Uso por Cluster × Transportadora")
            st.dataframe(analyses.get("Uso_Cluster_Transportadora"), use_container_width=True, hide_index=True)

            st.markdown("##### Sinergia — empréstimos entre clusters")
            st.dataframe(analyses.get("Sinergia_Emprestimos"), use_container_width=True, hide_index=True)

            st.markdown("##### Proporcionalidade por bucket")
            st.dataframe(analyses.get("Proporcionalidade_Bucket"), use_container_width=True, hide_index=True)

            st.markdown("##### Demanda × Capacidade Alocada × Plano — por Cluster")
            st.dataframe(analyses.get("Demanda_vs_Capacidade_Cluster"), use_container_width=True, hide_index=True)

            st.markdown("##### Demanda × Capacidade Alocada — por HUB")
            st.dataframe(analyses.get("Demanda_vs_Capacidade_HUB"), use_container_width=True, hide_index=True)

        # ── ABA 4: DIAGNÓSTICO DE FALTAS ──────────────────
        with tab4:
            if n_faltas == 0:
                st.success("✅ Nenhum cluster com falta de oferta nesta rodada.")
            else:
                st.error(f"⚠️ {n_faltas} cluster(s) com indicativo de falta de oferta ou gap de capacidade.")
                st.dataframe(analyses.get("Faltas_Resumo_Cluster"), use_container_width=True, hide_index=True)
                st.markdown("""
> **O que significa cada coluna:**
> - **Veiculos_SemOferta** — entregas registradas como `(SEM OFERTA)`, ou seja, sem veículo disponível no pool
> - **Gap_m3 / Gap_kg** — diferença entre a demanda e a capacidade efetivamente alocada
> - **Falta_Disponibilidade** — flag indicando que houve déficit nesse cluster
                """)


    except Exception as e:
        st.error("Erro ao processar. Veja detalhes abaixo:")
        st.exception(e)
else:
    st.info("Faça upload dos 2 arquivos na barra lateral e clique em **Rodar alocação**.")
