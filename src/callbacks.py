"""
=============================================================================
Callbacks da Aplicação: Modelo SIS (Suscetível-Infectado-Suscetível)
=============================================================================
Gerencia a reatividade da interface:
1. Carregamento instantâneo de cenários didáticos.
2. Atualização dos badges numéricos dos controles.
3. Integração numérica e renderização gráfica interativa com Plotly e KPIs.
"""

from dash import Input, Output, callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from src.model import simular_sis


def register_callbacks(app):
    """
    Registra todos os callbacks reativos na instância da aplicação Dash.
    """

    # =========================================================================
    # CALLBACK 1: CARREGAMENTO DE CENÁRIOS RÁPIDOS / QUESTÕES DIDÁTICAS
    # =========================================================================
    @app.callback(
        [
            Output('beta', 'value'),
            Output('gamma', 'value'),
            Output('t_end', 'value'),
            Output('s_init', 'value'),
            Output('i_init', 'value'),
        ],
        [
            Input('btn_padrao', 'n_clicks'),
            Input('btn_q1', 'n_clicks'),
            Input('btn_q2a', 'n_clicks'),
            Input('btn_q2b', 'n_clicks'),
            Input('btn_q2c', 'n_clicks'),
            Input('btn_q2d', 'n_clicks'),
            Input('btn_extincao', 'n_clicks'),
        ],
        prevent_initial_call=True
    )
    def carregar_cenario(btn_padrao, btn_q1, btn_q2a, btn_q2b, btn_q2c, btn_q2d, btn_extincao):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        botao_acionado = ctx.triggered[0]['prop_id'].split('.')[0]

        # Retorna: beta, gamma, t_end, s_init, i_init
        if botao_acionado in ('btn_padrao', 'btn_q1'):
            # Cenário de Referência (Q1: beta = 0.90, gamma = 0.50 -> R0 = 1.80)
            return 0.90, 0.50, 100, 100, 1

        elif botao_acionado == 'btn_q2a':
            # Q2a: gamma = 0.20 (R0 = 4.50, alta endemicidade)
            return 0.90, 0.20, 100, 100, 1

        elif botao_acionado == 'btn_q2b':
            # Q2b: gamma = 0.40 (R0 = 2.25)
            return 0.90, 0.40, 100, 100, 1

        elif botao_acionado == 'btn_q2c':
            # Q2c: gamma = 0.80 (R0 = 1.125)
            return 0.90, 0.80, 100, 100, 1

        elif botao_acionado == 'btn_q2d':
            # Q2d: gamma = 0.90 (R0 = 1.00, limiar crítico exato beta = gamma)
            return 0.90, 0.90, 150, 100, 1

        elif botao_acionado == 'btn_extincao':
            # Extinção: gamma = 1.20 > beta (R0 = 0.75 < 1)
            return 0.90, 1.20, 100, 100, 1

        raise PreventUpdate

    # =========================================================================
    # CALLBACK 2: ATUALIZAÇÃO DOS BADGES DOS SLIDERS
    # =========================================================================
    @app.callback(
        [
            Output("badge_beta", "children"),
            Output("badge_gamma", "children"),
            Output("badge_t_end", "children"),
            Output("badge_s_init", "children"),
            Output("badge_i_init", "children"),
        ],
        [
            Input("beta", "value"),
            Input("gamma", "value"),
            Input("t_end", "value"),
            Input("s_init", "value"),
            Input("i_init", "value"),
        ]
    )
    def update_badges(beta, gamma, t_end, s_init, i_init):
        tau_inf = (1.0 / gamma) if gamma and gamma > 0 else 0.0
        return (
            f"{beta:.2f} ano⁻¹",
            f"{gamma:.2f} ano⁻¹ (1/γ = {tau_inf:.1f} a)",
            f"{t_end} anos",
            f"{s_init} indiv.",
            f"{i_init} indiv.",
        )

    # =========================================================================
    # CALLBACK 3: INTEGRAÇÃO E RENDERIZAÇÃO GRÁFICA + KPIS
    # =========================================================================
    @app.callback(
        [
            Output('population_chart', 'figure'),
            Output('kpi_r0_val', 'children'),
            Output('kpi_r0_sub', 'children'),
            Output('kpi_pico_val', 'children'),
            Output('kpi_pico_sub', 'children'),
            Output('kpi_ieq_val', 'children'),
            Output('kpi_ieq_sub', 'children'),
            Output('kpi_seq_val', 'children'),
            Output('kpi_seq_sub', 'children'),
        ],
        [
            Input('s_init', 'value'),
            Input('i_init', 'value'),
            Input('beta', 'value'),
            Input('gamma', 'value'),
            Input('t_end', 'value'),
            Input('unit_display', 'value'),
        ]
    )
    def simula_e_atualiza(s_init, i_init, beta, gamma, t_end, unit_display):
        # Executa simulação numérica através do módulo matemático
        res = simular_sis(
            s_init=s_init,
            i_init=i_init,
            beta=beta,
            gamma=gamma,
            t_end=t_end,
            n_samples=1000
        )

        t_arr = res["t"]
        n_0 = res["n_0"]
        r0 = res["r0"]
        s_eq = res["s_eq"]
        i_eq = res["i_eq"]
        pico_infectados = res["pico_infectados"]
        tempo_pico = res["tempo_pico"]
        s_final = res["s_final"]
        i_final = res["i_final"]

        # Configura curvas conforme unidade selecionada (proporção % ou contagem absoluta)
        if unit_display == 'pct':
            y_s = res["prop_s"]
            y_i = res["prop_i"]
            eq_s_val = res["prop_s_eq"]
            eq_i_val = res["prop_i_eq"]
            y_axis_label = "Proporção da População Total (%)"
            hover_unit = "%"
            y_max_limit = 105.0
        else:
            y_s = res["s"]
            y_i = res["i"]
            eq_s_val = s_eq
            eq_i_val = i_eq
            y_axis_label = "Número de Indivíduos"
            hover_unit = " indivíduos"
            y_max_limit = max(100.0, n_0 * 1.05)

        # Construção do gráfico com Plotly
        fig = go.Figure()

        # Linhas teóricas tracejadas para os equilíbrios analíticos (se R0 > 1)
        if r0 > 1.0 and i_eq > 0.05:
            fig.add_hline(
                y=eq_i_val,
                line_dash="dot",
                line_color="#f87171",
                line_width=1.5,
                annotation_text=f"I* eq = {eq_i_val:.1f}{hover_unit}",
                annotation_position="bottom right",
                annotation_font=dict(size=10, color="#dc2626")
            )
            fig.add_hline(
                y=eq_s_val,
                line_dash="dot",
                line_color="#86efac",
                line_width=1.5,
                annotation_text=f"S* eq = {eq_s_val:.1f}{hover_unit}",
                annotation_position="top right",
                annotation_font=dict(size=10, color="#16a34a")
            )

        # Curva de Suscetíveis (S)
        fig.add_trace(go.Scatter(
            x=t_arr, y=y_s,
            name='Suscetível (S)',
            line=dict(color='#16a34a', width=3.5),
            hovertemplate="Tempo: %{x:.2f} anos<br>Suscetíveis: %{y:.1f}" + hover_unit + "<extra></extra>"
        ))

        # Curva de Infectados (I)
        fig.add_trace(go.Scatter(
            x=t_arr, y=y_i,
            name='Infectado (I)',
            line=dict(color='#dc2626', width=3.5, dash='dot'),
            hovertemplate="Tempo: %{x:.2f} anos<br>Infectados: %{y:.1f}" + hover_unit + "<extra></extra>"
        ))

        fig.update_layout(
            template='plotly_white',
            title=dict(
                text='<b>Evolução Temporal das Populações (Modelo SIS)</b>',
                font=dict(size=14, color="#1e293b"),
                x=0.02,
                xanchor='left',
                y=0.96,
                yref='container',
                yanchor='top'
            ),
            xaxis=dict(
                title='Tempo (anos)',
                range=[0, t_end],
                gridcolor='#f1f5f9',
                zerolinecolor='#cbd5e1'
            ),
            yaxis=dict(
                title=y_axis_label,
                range=[0, y_max_limit],
                gridcolor='#f1f5f9',
                zerolinecolor='#cbd5e1'
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#e2e8f0',
                borderwidth=1
            ),
            margin=dict(l=55, r=30, t=55, b=85),
            hovermode="x unified"
        )

        # Formatação dos cards de métricas (KPIs)
        # KPI 1: R0 = beta / gamma
        card_r0_val = f"{r0:.2f}"
        card_r0_sub = "Potencial endêmico (R₀ > 1)" if r0 > 1.0 else "Extinção da infecção (R₀ ≤ 1)"

        # KPI 2: Pico de Infectados
        pico_pct = (pico_infectados / n_0) * 100.0
        card_pico_val = f"{pico_pct:.1f}%" if unit_display == 'pct' else f"{pico_infectados:.1f}"
        if tempo_pico > 0:
            card_pico_sub = f"pico atingido no ano {tempo_pico:.1f}"
        else:
            card_pico_sub = "máximo no início t = 0"

        # KPI 3: Infectados no Equilíbrio (I*)
        if r0 > 1.0:
            card_ieq_val = f"{res['prop_i_eq']:.1f}%" if unit_display == 'pct' else f"{i_eq:.1f}"
            card_ieq_sub = f"I* = N·(1 - 1/R₀) (numérico: {i_final:.1f})"
        else:
            card_ieq_val = "0.0%" if unit_display == 'pct' else "0.0"
            card_ieq_sub = "Equilíbrio livre de doença (I* = 0)"

        # KPI 4: Suscetíveis no Equilíbrio (S*)
        if r0 > 1.0:
            card_seq_val = f"{res['prop_s_eq']:.1f}%" if unit_display == 'pct' else f"{s_eq:.1f}"
            card_seq_sub = f"S* = N/R₀ (numérico: {s_final:.1f})"
        else:
            card_seq_val = "100.0%" if unit_display == 'pct' else f"{n_0:.1f}"
            card_seq_sub = f"Toda a população suscetível (N={n_0:.0f})"

        return (
            fig,
            card_r0_val, card_r0_sub,
            card_pico_val, card_pico_sub,
            card_ieq_val, card_ieq_sub,
            card_seq_val, card_seq_sub
        )
