"""
Testes unitários para o módulo matemático e epidemiológico (src/model.py).
"""

import pytest
import numpy as np
from src.model import ode_sys, calcular_metricas_epidemiologicas, simular_sis


def test_ode_sys_extinction_zeros():
    """Se todas as populações forem zero, as taxas de variação devem ser zero."""
    state = [0.0, 0.0]
    derivs = ode_sys(t=0.0, state=state, beta=0.009, gamma=0.5)
    assert derivs == [0.0, 0.0]


def test_ode_sys_conservation_law():
    """
    Verifica se a soma das taxas de variação d/dt(S + I) é identicamente nula,
    garantindo a conservação estrita da população total N = S + I.
    """
    state = [75.0, 25.0]
    derivs = ode_sys(t=5.0, state=state, beta=0.009, gamma=0.5)
    ds_dt, di_dt = derivs
    assert pytest.approx(ds_dt + di_dt, abs=1e-9) == 0.0


def test_metricas_epidemiologicas_calculo():
    """Valida o cálculo analítico de R0, períodos e pontos de equilíbrio endêmico."""
    beta = 0.009
    gamma = 0.5
    s_0 = 100.0
    i_0 = 1.0
    m = calcular_metricas_epidemiologicas(beta=beta, gamma=gamma, s_init=s_0, i_init=i_0)

    # N0 = 101
    assert m["n_0"] == 101.0
    # R0 = (0.009 * 101) / 0.5 = 0.909 / 0.5 = 1.818
    assert pytest.approx(m["r0"], 1e-3) == 1.818
    # Período infeccioso médio = 1 / 0.5 = 2.0 anos
    assert pytest.approx(m["periodo_infeccioso_anos"], 1e-4) == 2.0
    # S* = gamma / beta = 0.5 / 0.009 = 55.5555...
    assert pytest.approx(m["s_eq"], 1e-3) == 55.556
    # I* = N0 - S* = 101 - 55.5555... = 45.4444...
    assert pytest.approx(m["i_eq"], 1e-3) == 45.444
    assert m["status_epidemia"] == "Potencial Epidêmico / Equilíbrio Endêmico"


def test_metricas_extincao_subcritico():
    """Valida o comportamento analítico em regime subcrítico (R0 <= 1)."""
    beta = 0.009
    gamma = 1.0
    s_0 = 100.0
    i_0 = 1.0
    m = calcular_metricas_epidemiologicas(beta=beta, gamma=gamma, s_init=s_0, i_init=i_0)

    # R0 = 0.909 <= 1
    assert m["r0"] < 1.0
    assert m["i_eq"] == 0.0
    assert m["s_eq"] == 101.0
    assert m["status_epidemia"] == "Extinção da Infecção (Subcrítico)"


def test_simular_sis_convergencia_equilibrio():
    """
    Verifica se a integração numérica de longo prazo converge exatamente para
    o ponto de equilíbrio analítico (S*, I*).
    """
    res = simular_sis(s_init=100.0, i_init=1.0, beta=0.009, gamma=0.5, t_end=150.0, n_samples=2000)

    # Ao final de 150 anos (muitos tempos de relaxamento), S e I devem estar no equilíbrio analítico
    assert pytest.approx(res["s_final"], rel=1e-2) == res["s_eq"]
    assert pytest.approx(res["i_final"], rel=1e-2) == res["i_eq"]


def test_simular_sis_conservacao_populacao():
    """Verifica a conservação de S(t) + I(t) == N0 em toda a trajetória temporal."""
    res = simular_sis(s_init=80.0, i_init=20.0, beta=0.015, gamma=0.3, t_end=50.0, n_samples=500)
    pop_total_ao_longo_tempo = res["s"] + res["i"]
    assert np.allclose(pop_total_ao_longo_tempo, 100.0, atol=1e-6)
