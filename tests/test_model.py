"""
Testes unitários para o módulo matemático e epidemiológico (src/model.py).
"""

import pytest
import numpy as np
from src.model import ode_sys, calcular_metricas_epidemiologicas, simular_sis


def test_ode_sys_extinction_zeros():
    """Se todas as populações forem zero, as taxas de variação devem ser zero."""
    state = [0.0, 0.0]
    derivs = ode_sys(t=0.0, state=state, beta=0.90, gamma=0.50)
    assert derivs == [0.0, 0.0]


def test_ode_sys_conservation_law():
    """
    Verifica se a soma das taxas de variação d/dt(S + I) com termo beta*S*I/N
    é identicamente nula, garantindo a conservação estrita da população total N = S + I.
    """
    state = [75.0, 25.0]
    derivs = ode_sys(t=5.0, state=state, beta=0.90, gamma=0.50)
    ds_dt, di_dt = derivs
    assert pytest.approx(ds_dt + di_dt, abs=1e-9) == 0.0


def test_metricas_epidemiologicas_calculo():
    """Valida o cálculo analítico de R0 = beta/gamma, períodos e pontos de equilíbrio endêmico."""
    beta = 0.90
    gamma = 0.50
    s_0 = 100.0
    i_0 = 1.0
    m = calcular_metricas_epidemiologicas(beta=beta, gamma=gamma, s_init=s_0, i_init=i_0)

    # N0 = 101
    assert m["n_0"] == 101.0
    # R0 = beta / gamma = 0.90 / 0.50 = 1.80
    assert pytest.approx(m["r0"], 1e-4) == 1.80
    # Período infeccioso médio = 1 / 0.50 = 2.0 anos
    assert pytest.approx(m["periodo_infeccioso_anos"], 1e-4) == 2.0
    # S* = N0 / R0 = 101 / 1.80 = 56.1111...
    assert pytest.approx(m["s_eq"], 1e-3) == 56.111
    # I* = N0 * (1 - 1/R0) = 101 * (0.8 / 1.8) = 44.8888...
    assert pytest.approx(m["i_eq"], 1e-3) == 44.889
    # Proporções % no equilíbrio
    assert pytest.approx(m["prop_s_eq"], 1e-2) == 55.56
    assert pytest.approx(m["prop_i_eq"], 1e-2) == 44.44
    assert m["status_epidemia"] == "Potencial Epidêmico / Equilíbrio Endêmico"


def test_metricas_invariancia_escala_populacao():
    """
    Comprova que a formulação com termo beta*S*I/N torna R0 e as proporções
    de equilíbrio (prop_s_eq, prop_i_eq) invariantes à escala do tamanho populacional N.
    """
    m1 = calcular_metricas_epidemiologicas(beta=0.90, gamma=0.50, s_init=99.0, i_init=1.0)
    m2 = calcular_metricas_epidemiologicas(beta=0.90, gamma=0.50, s_init=999.0, i_init=1.0)

    assert pytest.approx(m1["r0"], 1e-4) == m2["r0"]
    assert pytest.approx(m1["prop_s_eq"], 1e-3) == m2["prop_s_eq"]
    assert pytest.approx(m1["prop_i_eq"], 1e-3) == m2["prop_i_eq"]


def test_metricas_extincao_subcritico():
    """Valida o comportamento analítico em regime subcrítico (R0 <= 1)."""
    beta = 0.90
    gamma = 1.20
    s_0 = 100.0
    i_0 = 1.0
    m = calcular_metricas_epidemiologicas(beta=beta, gamma=gamma, s_init=s_0, i_init=i_0)

    # R0 = 0.90 / 1.20 = 0.75 <= 1
    assert m["r0"] == 0.75
    assert m["i_eq"] == 0.0
    assert m["s_eq"] == 101.0
    assert m["status_epidemia"] == "Extinção da Infecção (Subcrítico)"


def test_simular_sis_convergencia_equilibrio():
    """
    Verifica se a integração numérica de longo prazo converge exatamente para
    o ponto de equilíbrio analítico (S*, I*).
    """
    res = simular_sis(s_init=100.0, i_init=1.0, beta=0.90, gamma=0.50, t_end=100.0, n_samples=2000)

    assert pytest.approx(res["s_final"], rel=1e-2) == res["s_eq"]
    assert pytest.approx(res["i_final"], rel=1e-2) == res["i_eq"]


def test_simular_sis_conservacao_populacao():
    """Verifica a conservação de S(t) + I(t) == N0 em toda a trajetória temporal."""
    res = simular_sis(s_init=80.0, i_init=20.0, beta=1.20, gamma=0.40, t_end=50.0, n_samples=500)
    pop_total_ao_longo_tempo = res["s"] + res["i"]
    assert np.allclose(pop_total_ao_longo_tempo, 100.0, atol=1e-6)
