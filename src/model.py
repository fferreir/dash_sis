"""
=============================================================================
Módulo Matemático e Numérico: Modelo SIS (Suscetível-Infectado-Suscetível)
=============================================================================
Contém as Equações Diferenciais Ordinárias (EDOs), rotinas de integração
numérica via solve_ivp (método Radau para sistemas rígidos/stiff) e o cálculo
de métricas epidemiológicas fundamentais (R0, pontos de equilíbrio e indicadores).
"""

import numpy as np
from scipy.integrate import solve_ivp


def ode_sys(t, state, beta, gamma):
    """
    Sistema compartimental SIS sem dinâmica vital (população total constante).

    Estado:
      s: Número ou fração de indivíduos suscetíveis
      i: Número ou fração de indivíduos infectados

    Parâmetros:
      t: Tempo em anos
      beta: Taxa de transmissão / contatos potencialmente infectantes (ano^-1)
      gamma: Taxa de recuperação = inverso do período infeccioso (ano^-1)

    Retorna:
      [ds_dt, di_dt]
    """
    s = max(0.0, float(state[0]))
    i = max(0.0, float(state[1]))

    ds_dt = -beta * s * i + gamma * i
    di_dt = beta * s * i - gamma * i

    return [ds_dt, di_dt]


def calcular_metricas_epidemiologicas(beta, gamma, s_init, i_init):
    """
    Calcula métricas analíticas e limiares clássicos do modelo epidemiológico SIS.

    Retorna um dicionário com:
      - n_0: População total inicial (s_init + i_init)
      - r0: Número básico de reprodução (beta * n_0 / gamma)
      - periodo_infeccioso: Período infeccioso médio em anos (1 / gamma) e em dias
      - s_eq: Nível de suscetíveis no equilíbrio
      - i_eq: Nível de infectados no equilíbrio
      - status_epidemia: 'Potencial Epidêmico / Equilíbrio Endêmico' ou 'Extinção / Livre de Infecção'
    """
    beta = max(0.0, float(beta))
    gamma = max(1e-6, float(gamma))
    s_0 = max(0.0, float(s_init))
    i_0 = max(0.0, float(i_init))
    n_0 = max(s_0 + i_0, 1e-6)

    # R0 para o modelo SIS clássico com ação de massas
    r0 = (beta * n_0) / gamma
    periodo_infeccioso_anos = 1.0 / gamma
    periodo_infeccioso_dias = periodo_infeccioso_anos * 365.0

    # Determinação do ponto de equilíbrio estável
    if r0 > 1.0 and beta > 0:
        s_eq = gamma / beta
        i_eq = n_0 - s_eq
        status_epidemia = "Potencial Epidêmico / Equilíbrio Endêmico"
    else:
        s_eq = n_0
        i_eq = 0.0
        status_epidemia = "Extinção da Infecção (Subcrítico)"

    return {
        "n_0": n_0,
        "r0": r0,
        "periodo_infeccioso_anos": periodo_infeccioso_anos,
        "periodo_infeccioso_dias": periodo_infeccioso_dias,
        "s_eq": s_eq,
        "i_eq": i_eq,
        "prop_s_eq": (s_eq / n_0) * 100.0,
        "prop_i_eq": (i_eq / n_0) * 100.0,
        "status_epidemia": status_epidemia
    }


def simular_sis(
    s_init=100.0,
    i_init=1.0,
    beta=0.009,
    gamma=0.5,
    t_end=100.0,
    n_samples=1000
):
    """
    Executa a simulação numérica temporal completa do modelo SIS.

    Retorna um dicionário com trajetórias temporais, indicadores acumulados,
    níveis de equilíbrio analíticos e numéricos, e estatísticas de pico.
    """
    s_0 = float(s_init) if s_init is not None else 100.0
    i_0 = float(i_init) if i_init is not None else 1.0
    beta = float(beta) if beta is not None else 0.009
    gamma = float(gamma) if gamma is not None else 0.5

    n_0 = max(s_0 + i_0, 1e-6)
    t_begin = 0.0
    t_end = max(1.0, float(t_end))
    t_span = (t_begin, t_end)
    t_eval = np.linspace(t_begin, t_end, int(n_samples))

    # Integração numérica com método implícito Radau (alta estabilidade)
    sol = solve_ivp(
        fun=ode_sys,
        t_span=t_span,
        y0=[s_0, i_0],
        args=(beta, gamma),
        t_eval=t_eval,
        method='Radau'
    )

    s_arr = np.maximum(0.0, sol.y[0])
    i_arr = np.maximum(0.0, sol.y[1])

    # Estatísticas de pico
    idx_pico = int(np.argmax(i_arr))
    pico_infectados = float(i_arr[idx_pico])
    tempo_pico = float(sol.t[idx_pico])

    # Valores finais atingidos na simulação
    s_final = float(s_arr[-1])
    i_final = float(i_arr[-1])

    # Proporções (%) da população inicial N0
    prop_s = (s_arr / n_0) * 100.0
    prop_i = (i_arr / n_0) * 100.0

    # Métricas analíticas teóricas
    metricas = calcular_metricas_epidemiologicas(beta, gamma, s_0, i_0)

    # Estimativa do tempo característico de relaxamento até o equilíbrio
    taxa_liquida = abs(beta * n_0 - gamma)
    tempo_caracteristico = (1.0 / taxa_liquida) if taxa_liquida > 1e-5 else float('inf')

    return {
        "t": sol.t,
        "s": s_arr,
        "i": i_arr,
        "prop_s": prop_s,
        "prop_i": prop_i,
        "n_0": n_0,
        "pico_infectados": pico_infectados,
        "tempo_pico": tempo_pico,
        "s_final": s_final,
        "i_final": i_final,
        "r0": metricas["r0"],
        "s_eq": metricas["s_eq"],
        "i_eq": metricas["i_eq"],
        "prop_s_eq": metricas["prop_s_eq"],
        "prop_i_eq": metricas["prop_i_eq"],
        "status_epidemia": metricas["status_epidemia"],
        "tempo_caracteristico": tempo_caracteristico,
        "beta": beta,
        "gamma": gamma
    }
