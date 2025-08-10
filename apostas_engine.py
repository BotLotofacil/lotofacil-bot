# -*- coding: utf-8 -*-
"""
apostas_engine.py
Núcleo de geração e avaliação de apostas Lotofácil com:
- Score multi-critério (hot/cold com decaimento, composição, mod 3/5, finais, grid 5x5, anti-viés do último sorteio)
- Filtros rígidos (sequências >=4, repetição excessiva do último resultado, extremos par/ímpar e soma)
- GRASP (construção gulosa randomizada) + melhoria local por trocas
- Seleção do conjunto final com diversidade (penalização por Jaccard alto) e cobertura de pares/trincas quentes
API: gerar_apostas(...) e avaliar_apostas(...).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Iterable
from collections import Counter
import random

NUMS = list(range(1, 26))  # 1..25
TAMANHO_APOSTA = 15

# -----------------------------
# Configuração
# -----------------------------
@dataclass
class Config:
    # Hot/cold
    janela_hot: int = 120
    meia_vida: int = 30

    # Composição
    alvo_par_min: int = 7
    alvo_par_max: int = 8
    soma_min: int = 170
    soma_max: int = 210

    # Grid 5x5
    linha_min: int = 2
    linha_max: int = 4
    coluna_min: int = 2
    coluna_max: int = 4

    # Finais
    max_por_final: int = 3

    # Distribuições mod
    peso_mod3: float = 1.0
    peso_mod5: float = 1.0

    # Anti-viés último sorteio
    penaliza_overlap_maior_que: int = 7
    overlap_maximo_ultimo: int = 9

    # Filtros rígidos
    max_sequencia: int = 3
    max_par: int = 10
    max_impar: int = 10

    # Pesos de score
    w_hot: float = 1.0
    w_comp_par: float = 2.0
    w_comp_soma: float = 1.5
    w_grid: float = 1.2
    w_finais: float = 1.0
    w_mod: float = 1.0
    w_overlap: float = 2.0
    w_sequencia: float = 1.8
    w_pares_quentes: float = 0.35
    w_trincas_quentes: float = 0.6

    # GRASP
    alpha: float = 0.30
    candidatos_por_aposta: int = 48
    max_trocas_melhoria: int = 200

    # Diversidade e cobertura
    jaccard_penal_limite_alto: float = 0.60
    jaccard_penal_limite_medio: float = 0.50
    penal_jaccard_alto: float = 8.0
    penal_jaccard_medio: float = 3.0
    top_pares: int = 40
    top_trincas: int = 30

# -----------------------------
# Utilidades
# -----------------------------
def to_set(aposta: Iterable[int]) -> Set[int]:
    return set(int(x) for x in aposta)

def conta_sequencias(nums: Set[int]) -> int:
    s = sorted(nums)
    maior = atual = 1 if s else 0
    for i in range(1, len(s)):
        if s[i] == s[i-1] + 1:
            atual += 1
            maior = max(maior, atual)
        else:
            atual = 1
    return maior

def grid_pos(n: int) -> Tuple[int, int]:
    return ((n - 1) // 5 + 1, (n - 1) % 5 + 1)

def jaccard(a: Set[int], b: Set[int]) -> float:
    inter = len(a & b)
    un = len(a | b)
    return inter / un if un else 0.0

def pares_da_aposta(ap: Set[int]) -> Set[Tuple[int, int]]:
    s = sorted(ap)
    res = set()
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            res.add((s[i], s[j]))
    return res

def trincas_da_aposta(ap: Set[int]) -> Set[Tuple[int, int, int]]:
    s = sorted(ap)
    res = set()
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            for k in range(j+1, len(s)):
                res.add((s[i], s[j], s[k]))
    return res

# -----------------------------
# Features do histórico
# -----------------------------
def _expo_weight(idx: int, meia_vida: int) -> float:
    if meia_vida <= 0:
        return 1.0
    return 0.5 ** (idx / float(meia_vida))

def features_historico(historico: List[List[int]], cfg: Config) -> Dict:
    if not historico or any(len(set(h)) != 15 for h in historico):
        raise ValueError("Histórico inválido: cada concurso deve ter 15 números distintos.")

    janela = cfg.janela_hot
    recortes = historico[-janela:] if len(historico) > janela else historico[:]
    rec_rev = list(reversed(recortes))  # 0 = mais recente

    # Hotness com decaimento
    pesos = [_expo_weight(i, cfg.meia_vida) for i in range(len(rec_rev))]
    score_num = {n: 0.0 for n in NUMS}
    for idx, concurso in enumerate(rec_rev):
        w = pesos[idx]
        for n in concurso:
            score_num[n] += w
    maxv = max(score_num.values()) or 1.0
    hotness = {n: (score_num[n] / maxv) for n in NUMS}

    # Pares/trincas quentes (contagem simples)
    c_pares = Counter()
    c_trincas = Counter()
    for concurso in recortes:
        s = sorted(concurso)
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                c_pares[(s[i], s[j])] += 1
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                for k in range(j+1, len(s)):
                    c_trincas[(s[i], s[j], s[k])] += 1

    pares_quentes = set([p for p, _ in c_pares.most_common(cfg.top_pares)])
    trincas_quentes = set([t for t, _ in c_trincas.most_common(cfg.top_trincas)])

    ultimo = set(historico[-1])

    return {
        "hotness": hotness,
        "pares_quentes": pares_quentes,
        "trincas_quentes": trincas_quentes,
        "ultimo": ultimo
    }

# -----------------------------
# Score
# -----------------------------
def _penalizacao_intervalo(valor: int, minimo: int, maximo: int) -> float:
    if minimo <= valor <= maximo:
        return 0.0
    if valor < minimo:
        return float((minimo - valor) ** 2)
    return float((valor - maximo) ** 2)

def _penaliza_distribuicao(contagens: Dict[int, int], alvo: float) -> float:
    total = 0.0
    for _, v in contagens.items():
        total += (v - alvo) ** 2
    return total

def score_aposta(aposta: Set[int], feats: Dict, cfg: Config) -> float:
    if len(aposta) != TAMANHO_APOSTA:
        return float("-inf")

    # Filtros rígidos
    pares = sum(1 for n in aposta if n % 2 == 0)
    impares = TAMANHO_APOSTA - pares
    if pares > cfg.max_par or impares > cfg.max_impar:
        return float("-inf")

    maior_seq = conta_sequencias(aposta)
    if maior_seq >= (cfg.max_sequencia + 1):
        return float("-inf")

    overlap_ult = len(aposta & feats["ultimo"])
    if overlap_ult > cfg.overlap_maximo_ultimo:
        return float("-inf")

    # Componentes
    hot = feats["hotness"]
    soma_hot = sum(hot[n] for n in aposta)

    pen_par = 0.0
    if pares < cfg.alvo_par_min:
        pen_par = (cfg.alvo_par_min - pares) ** 2
    elif pares > cfg.alvo_par_max:
        pen_par = (pares - cfg.alvo_par_max) ** 2

    soma_tot = sum(aposta)
    pen_soma = _penalizacao_intervalo(soma_tot, cfg.soma_min, cfg.soma_max)

    # Grid
    linhas = Counter()
    colunas = Counter()
    for n in aposta:
        li, co = grid_pos(n)
        linhas[li] += 1
        colunas[co] += 1
    pen_grid = 0.0
    for li in range(1, 6):
        pen_grid += _penalizacao_intervalo(linhas[li], cfg.linha_min, cfg.linha_max)
    for co in range(1, 6):
        pen_grid += _penalizacao_intervalo(colunas[co], cfg.coluna_min, cfg.coluna_max)

    # Finais
    finais = Counter(n % 10 for n in aposta)
    pen_finais = 0.0
    for _, qtd in finais.items():
        if qtd > cfg.max_por_final:
            pen_finais += (qtd - cfg.max_por_final) ** 2

    # Mod 3 e 5
    mod3 = Counter(n % 3 for n in aposta)
    mod5 = Counter(n % 5 for n in aposta)
    alvo_mod3 = TAMANHO_APOSTA / 3.0
    alvo_mod5 = TAMANHO_APOSTA / 5.0
    pen_mod = cfg.peso_mod3 * _penaliza_distribuicao({k: mod3.get(k, 0) for k in (0,1,2)}, alvo_mod3) \
            + cfg.peso_mod5 * _penaliza_distribuicao({k: mod5.get(k, 0) for k in (0,1,2,3,4)}, alvo_mod5)

    # Overlap com último (penal acima do limiar)
    pen_overlap = 0.0
    if overlap_ult > cfg.penaliza_overlap_maior_que:
        pen_overlap = (overlap_ult - cfg.penaliza_overlap_maior_que) ** 2

    # Sequências (leve penal para 3)
    pen_seq = 1.0 if maior_seq == cfg.max_sequencia else 0.0

    # Cobertura pares/trincas quentes
    ap_pares = pares_da_aposta(aposta)
    ap_trincas = trincas_da_aposta(aposta)
    bonus_pares = len(ap_pares & feats["pares_quentes"])
    bonus_trincas = len(ap_trincas & feats["trincas_quentes"])

    score = 0.0
    score += cfg.w_hot * soma_hot
    score -= cfg.w_comp_par * pen_par
    score -= cfg.w_comp_soma * pen_soma
    score -= cfg.w_grid * pen_grid
    score -= cfg.w_finais * pen_finais
    score -= cfg.w_mod * pen_mod
    score -= cfg.w_overlap * pen_overlap
    score -= cfg.w_sequencia * pen_seq
    score += cfg.w_pares_quentes * bonus_pares
    score += cfg.w_trincas_quentes * bonus_trincas

    return float(score)

# -----------------------------
# GRASP + melhoria local
# -----------------------------
def _score_candidato_incr(partial: Set[int], candidato: int, feats: Dict, cfg: Config) -> float:
    base = feats["hotness"][candidato]

    # par/ímpar
    pares = sum(1 for n in partial if n % 2 == 0) + (1 if candidato % 2 == 0 else 0)
    if pares < cfg.alvo_par_min:
        comp_bonus = 0.3
    elif pares > cfg.alvo_par_max:
        comp_bonus = -0.3
    else:
        comp_bonus = 0.2

    # sequências
    seq_pen = 0.0
    if (candidato-1 in partial and candidato-2 in partial) or (candidato+1 in partial and candidato+2 in partial) \
       or (candidato-1 in partial and candidato+1 in partial):
        seq_pen = 0.15
    if (candidato-1 in partial and candidato-2 in partial and candidato-3 in partial) or \
       (candidato+1 in partial and candidato+2 in partial and candidato+3 in partial):
        seq_pen = 0.60

    # grid
    linhas = Counter()
    colunas = Counter()
    for n in partial:
        li, co = grid_pos(n)
        linhas[li] += 1
        colunas[co] += 1
    li, co = grid_pos(candidato)
    linhas[li] += 1
    colunas[co] += 1
    grid_pen = 0.0
    if linhas[li] > cfg.linha_max: grid_pen += 0.4
    if colunas[co] > cfg.coluna_max: grid_pen += 0.4

    # finais
    finais = Counter(n % 10 for n in partial)
    finais[candidato % 10] += 1
    fin_pen = 0.4 if finais[candidato % 10] > cfg.max_por_final else 0.0

    # mod 3/5 (leve)
    mod3 = Counter(n % 3 for n in partial)
    mod5 = Counter(n % 5 for n in partial)
    mod3[(candidato % 3)] += 1
    mod5[(candidato % 5)] += 1
    alvo3 = (len(partial)+1) / 3.0
    alvo5 = (len(partial)+1) / 5.0
    mod_pen = 0.03 * (sum((mod3[k] - alvo3)**2 for k in (0,1,2))
                      + sum((mod5[k] - alvo5)**2 for k in (0,1,2,3,4)))

    # overlap com último
    overlap_pre = len(partial & feats["ultimo"])
    ov_pen = 0.15 if (candidato in feats["ultimo"] and overlap_pre >= cfg.penaliza_overlap_maior_que) \
            else (0.05 if candidato in feats["ultimo"] else 0.0)

    return base + comp_bonus - seq_pen - grid_pen - fin_pen - mod_pen - ov_pen

def _construir_aposta_grasp(feats: Dict, cfg: Config, rng: random.Random) -> Set[int]:
    partial: Set[int] = set()
    disponiveis = set(NUMS)

    while len(partial) < TAMANHO_APOSTA:
        cand_scores = [(n, _score_candidato_incr(partial, n, feats, cfg)) for n in disponiveis]
        cand_scores.sort(key=lambda x: x[1], reverse=True)
        best = cand_scores[0][1]
        worst = cand_scores[-1][1]
        limite = best - cfg.alpha * (best - worst)
        rcl = [n for n, sc in cand_scores if sc >= limite]
        escolhido = rng.choice(rcl)
        partial.add(escolhido)
        disponiveis.remove(escolhido)

    # melhoria local
    aposta = set(partial)
    melhor = score_aposta(aposta, feats, cfg)
    for _ in range(cfg.max_trocas_melhoria):
        out_n = rng.choice(tuple(aposta))
        in_n = rng.choice(tuple(set(NUMS) - aposta))
        nova = set(aposta)
        nova.remove(out_n)
        nova.add(in_n)
        sc = score_aposta(nova, feats, cfg)
        if sc > melhor:
            aposta = nova
            melhor = sc
    return aposta

def _gera_candidatos(feats: Dict, cfg: Config, rng: random.Random, quantidade: int):
    alvo = quantidade * 5
    candidatos = []
    while len(candidatos) < alvo:
        melhor_local = None
        melhor_score = float("-inf")
        for _ in range(cfg.candidatos_por_aposta):
            ap = _construir_aposta_grasp(feats, cfg, rng)
            sc = score_aposta(ap, feats, cfg)
            if sc > melhor_score:
                melhor_local, melhor_score = ap, sc
        if melhor_local is not None and melhor_score > float("-inf"):
            candidatos.append((melhor_local, melhor_score))
    candidatos.sort(key=lambda x: x[1], reverse=True)
    return candidatos

def _seleciona_diversificado(cands, feats: Dict, cfg: Config, quantidade: int):
    selecionadas = []
    cobertos_pares = set()
    cobertos_trincas = set()

    for ap, base_sc in cands:
        if len(selecionadas) >= quantidade:
            break

        # penal similaridade
        penal = 0.0
        for ap_sel in selecionadas:
            j = jaccard(ap, ap_sel)
            if j >= cfg.jaccard_penal_limite_alto:
                penal += cfg.penal_jaccard_alto
            elif j >= cfg.jaccard_penal_limite_medio:
                penal += cfg.penal_jaccard_medio

        # bônus cobertura
        ap_pairs = pares_da_aposta(ap)
        ap_tris = trincas_da_aposta(ap)
        novos_pares = len((ap_pairs & feats["pares_quentes"]) - cobertos_pares)
        novos_tris  = len((ap_tris  & feats["trincas_quentes"]) - cobertos_trincas)
        bonus_cov = cfg.w_pares_quentes * novos_pares + cfg.w_trincas_quentes * novos_tris

        score_final = base_sc - penal + bonus_cov
        selecionadas.append(ap)
        cobertos_pares |= (ap_pairs & feats["pares_quentes"])
        cobertos_trincas |= (ap_tris  & feats["trincas_quentes"])

    # fallback se faltar
    if len(selecionadas) < quantidade:
        for ap, _ in cands:
            if ap not in selecionadas:
                selecionadas.append(ap)
                if len(selecionadas) >= quantidade:
                    break
    return selecionadas

# -----------------------------
# API pública
# -----------------------------
def gerar_apostas(historico: List[List[int]], quantidade: int, seed: int | None = None, cfg: Config | None = None) -> List[List[int]]:
    cfg = cfg or Config()
    rng = random.Random(seed)
    feats = features_historico(historico, cfg)
    candidatos = _gera_candidatos(feats, cfg, rng, quantidade)
    selecionadas = _seleciona_diversificado(candidatos, feats, cfg, quantidade)
    return [sorted(list(ap)) for ap in selecionadas]

def avaliar_apostas(apostas: List[List[int]], resultado: List[int]) -> List[Dict]:
    res = to_set(resultado)
    out = []
    for idx, ap in enumerate(apostas, start=1):
        s = to_set(ap)
        acertos = len(s & res)
        out.append({"indice": idx, "aposta": sorted(list(s)), "acertos": acertos, "premiada": (acertos >= 11)})
    return out
