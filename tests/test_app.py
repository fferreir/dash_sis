"""
Testes de integração para a interface e callbacks do Dash SIS.
"""

import pytest
import numpy as np
from app import app
from src.model import simular_sis, calcular_metricas_epidemiologicas
from src.layout import rodape


def test_app_layout_structure():
    """Verifica se o layout do Dash é criado com sucesso e possui estrutura de Container."""
    layout = app.layout
    assert layout is not None
    assert "Container" in type(layout).__name__


def test_footer_github_link():
    """Verifica se o rodapé institucional possui o link correto para o repositório GitHub."""
    assert rodape is not None
    footer_repr = str(rodape)
    assert "https://github.com/fferreir/dash_sis" in footer_repr
    assert "Código Aberto no GitHub" in footer_repr


def test_simulacao_parametros_extremos():
    """Verifica estabilidade numérica sob parâmetros extremos e limites de escala."""
    res = simular_sis(
        s_init=500.0,
        i_init=100.0,
        beta=0.05,
        gamma=0.01,
        t_end=50.0,
        n_samples=500
    )
    assert not np.isnan(res['s']).any()
    assert not np.isnan(res['i']).any()
    assert (res['s'] >= 0.0).all()
    assert (res['i'] >= 0.0).all()


def test_metricas_limites():
    """Verifica robustez das funções analíticas com parâmetros limite."""
    m = calcular_metricas_epidemiologicas(beta=0.0, gamma=0.5, s_init=100.0, i_init=1.0)
    assert m['r0'] == 0.0
    assert m['status_epidemia'] == "Extinção da Infecção (Subcrítico)"
