"""
=============================================================================
Layout da Interface Visual: Modelo SIS (Suscetível-Infectado-Suscetível)
=============================================================================
Define os componentes de apresentação: Hero Header, Cards de Métricas (KPIs),
Painéis de Controle com Sliders e Badges, Botões de Cenários Didáticos,
Gráfico Principal Interativo, Acordeom Teórico-Pedagógico e Rodapé Institucional
com Link para o GitHub.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


# =============================================================================
# CABEÇALHO HERO
# =============================================================================
cabecalho = html.Div(
    [
        html.Div(
            [
                html.H1(
                    [
                        html.I(className="fa-solid fa-virus me-2"),
                        "Modelo SIS (Suscetível-Infectado-Suscetível)"
                    ],
                    className="display-6 mb-1"
                ),
                html.P(
                    "Simulador epidemiológico determinístico com incidência padrão (β·S·I/N), perda imediata de imunidade e equilíbrio endêmico.",
                    className="lead fs-6 mb-0 opacity-75"
                ),
            ],
            className="hero-header p-3 mb-3"
        )
    ]
)


# =============================================================================
# CARDS DE MÉTRICAS (KPIS)
# =============================================================================
cards_metricas = dbc.Row(
    [
        dbc.Col(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Número de Reprodução (R₀)", className="kpi-title"),
                            html.Div(id="kpi_r0_val", className="kpi-value"),
                            html.Div(id="kpi_r0_sub", className="kpi-subtitle"),
                            html.I(className="fa-solid fa-arrows-split-up-and-left kpi-icon"),
                        ],
                        className="kpi-card-inner"
                    )
                ],
                className="kpi-card kpi-r0"
            ),
            width=6, sm=6, lg=3, className="mb-2"
        ),
        dbc.Col(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Pico de Infectados", className="kpi-title"),
                            html.Div(id="kpi_pico_val", className="kpi-value"),
                            html.Div(id="kpi_pico_sub", className="kpi-subtitle"),
                            html.I(className="fa-solid fa-chart-line kpi-icon"),
                        ],
                        className="kpi-card-inner"
                    )
                ],
                className="kpi-card kpi-peak"
            ),
            width=6, sm=6, lg=3, className="mb-2"
        ),
        dbc.Col(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Infectados no Equilíbrio (I*)", className="kpi-title"),
                            html.Div(id="kpi_ieq_val", className="kpi-value"),
                            html.Div(id="kpi_ieq_sub", className="kpi-subtitle"),
                            html.I(className="fa-solid fa-virus kpi-icon"),
                        ],
                        className="kpi-card-inner"
                    )
                ],
                className="kpi-card kpi-infected-eq"
            ),
            width=6, sm=6, lg=3, className="mb-2"
        ),
        dbc.Col(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Suscetíveis no Equilíbrio (S*)", className="kpi-title"),
                            html.Div(id="kpi_seq_val", className="kpi-value"),
                            html.Div(id="kpi_seq_sub", className="kpi-subtitle"),
                            html.I(className="fa-solid fa-users kpi-icon"),
                        ],
                        className="kpi-card-inner"
                    )
                ],
                className="kpi-card kpi-susceptible-eq"
            ),
            width=6, sm=6, lg=3, className="mb-2"
        ),
    ],
    className="mb-2"
)


# =============================================================================
# PAINEL DE CENÁRIOS RÁPIDOS E QUESTÕES PEDAGÓGICAS
# =============================================================================
ajuste_cenarios = html.Div(
    [
        html.Div(
            [
                html.I(className="fa-solid fa-bookmark me-2 text-primary"),
                "Cenários Didáticos Rápidos (Questões de Estudo)"
            ],
            className="card-header"
        ),
        html.Div(
            [
                html.P(
                    "Carregue instantaneamente as configurações para explorar cada questão:",
                    className="small text-muted mb-2"
                ),
                html.Div(
                    [
                        dbc.Button(
                            "Padrão (Referência: β=0.90, γ=0.50)",
                            id="btn_padrao",
                            size="sm",
                            color="primary",
                            outline=True,
                            className="btn-scenario"
                        ),
                        dbc.Button(
                            "Q1: Ref (γ=0.50, R₀=1.80)",
                            id="btn_q1",
                            size="sm",
                            color="secondary",
                            outline=True,
                            className="btn-scenario"
                        ),
                        dbc.Button(
                            "Q2a: γ=0.20 (R₀=4.50)",
                            id="btn_q2a",
                            size="sm",
                            color="danger",
                            outline=True,
                            className="btn-scenario"
                        ),
                        dbc.Button(
                            "Q2b: γ=0.40 (R₀=2.25)",
                            id="btn_q2b",
                            size="sm",
                            color="warning",
                            outline=True,
                            className="btn-scenario"
                        ),
                        dbc.Button(
                            "Q2c: γ=0.80 (R₀=1.13)",
                            id="btn_q2c",
                            size="sm",
                            color="info",
                            outline=True,
                            className="btn-scenario"
                        ),
                        dbc.Button(
                            "Q2d: Limiar γ=0.90 (R₀=1.00)",
                            id="btn_q2d",
                            size="sm",
                            color="dark",
                            outline=True,
                            className="btn-scenario"
                        ),
                        dbc.Button(
                            "Extinção (γ=1.20, R₀=0.75)",
                            id="btn_extincao",
                            size="sm",
                            color="success",
                            outline=True,
                            className="btn-scenario"
                        ),
                    ],
                    className="d-flex flex-wrap"
                )
            ],
            className="p-3"
        )
    ],
    className="card-control mb-3"
)


# =============================================================================
# COMPONENTES DE CONTROLE (SLIDERS E BADGES)
# =============================================================================
ajuste_parametros = html.Div(
    [
        html.Div(
            [
                html.I(className="fa-solid fa-sliders me-2 text-primary"),
                "Ajuste dos Parâmetros Epidemiológicos"
            ],
            className="card-header"
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'$$\beta$$: Taxa de transmissão / contatos ($$ano^{-1}$$)', mathjax=True), className="mb-0"),
                    html.Span(id="badge_beta", className="slider-val-badge badge-blue")
                ], className="slider-label"),
                dcc.Slider(
                    id="beta",
                    min=0.10,
                    max=2.50,
                    step=0.05,
                    value=0.90,
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'$$\gamma$$: Taxa de recuperação ($$ano^{-1}$$)', mathjax=True), className="mb-0"),
                    html.Span(id="badge_gamma", className="slider-val-badge badge-blue")
                ], className="slider-label"),
                dcc.Slider(
                    id="gamma",
                    min=0.05,
                    max=1.50,
                    step=0.05,
                    value=0.50,
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label("Horizonte de Tempo (anos)", className="mb-0"),
                    html.Span(id="badge_t_end", className="slider-val-badge")
                ], className="slider-label"),
                dcc.Slider(
                    id="t_end",
                    min=10,
                    max=200,
                    step=10,
                    value=100,
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ],
            className="slider-group",
        ),
    ],
    className="card-control mb-3",
)

ajuste_condicoes_iniciais = html.Div(
    [
        html.Div(
            [
                html.I(className="fa-solid fa-users me-2 text-success"),
                "Ajuste das Condições Iniciais"
            ],
            className="card-header"
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'$$S(0)$$: Suscetíveis iniciais', mathjax=True), className="mb-0"),
                    html.Span(id="badge_s_init", className="slider-val-badge badge-green")
                ], className="slider-label"),
                dcc.Slider(
                    id="s_init",
                    min=0,
                    max=500,
                    step=5,
                    value=100,
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'$$I(0)$$: Infectados iniciais', mathjax=True), className="mb-0"),
                    html.Span(id="badge_i_init", className="slider-val-badge badge-red")
                ], className="slider-label"),
                dcc.Slider(
                    id="i_init",
                    min=0,
                    max=100,
                    step=1,
                    value=1,
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ],
            className="slider-group",
        ),
    ],
    className="card-control mb-3",
)


# =============================================================================
# DOCUMENTAÇÃO TEÓRICA E ACORDEOM DIDÁTICO
# =============================================================================
def _criar_secao_descricao(image_url):
    return html.Div([
        dcc.Markdown(
            r'''
            É apresentado um modelo determinístico compartimental do tipo **SIS** (*Suscetível-Infectado-Suscetível*) com **incidência padrão** (frequência-dependente).

            Neste modelo, consideramos infecções para as quais o indivíduo, algum tempo após ter passado pelo processo infeccioso, **volta imediatamente a ser suscetível**, sem adquirir imunidade duradoura.
            
            Exemplos clássicos de dinâmica SIS na infectologia incluem:
            * **Doenças Sexualmente Transmissíveis (DSTs):** Infecções bacterianas como gonorreia e clamídia, em que episódios de cura clínica não geram proteção imunológica subsequente.
            * **Infecções bacterianas de mucosas:** Otites recorrentes, faringites e infecções urinárias.

            A população total é fechada e constante: $$N = S + I$$.
            ''', mathjax=True
        ),
        html.Div([
            html.Img(src=image_url, className="sis-diagram my-3 d-block mx-auto", alt="Diagrama do Modelo SIS com beta*S*I/N"),
        ]),
        dcc.Markdown(
            r'''
            ### Equações Diferenciais Ordinárias (EDOs) com Termo $$\frac{\beta S I}{N}$$:
            $$\begin{aligned}
            \frac{dS}{dt} &= -\frac{\beta S I}{N} + \gamma I \\
            \frac{dI}{dt} &= \frac{\beta S I}{N} - \gamma I = \left(\frac{\beta S}{N} - \gamma\right) I
            \end{aligned}$$

            Somando as duas equações, observamos a conservação estrita da população total:
            $$\frac{d(S + I)}{dt} = \frac{dN}{dt} = 0 \implies S(t) + I(t) = N = S(0) + I(0)$$

            Em termos de proporções da população ($$s = S/N$$ e $$i = I/N$$, com $$s + i = 1$$):
            $$\frac{di}{dt} = \beta (1 - i) i - \gamma i = (\beta - \gamma) i \left(1 - \frac{\beta}{\beta - \gamma} i\right)$$
            que é formalmente equivalente à equação logística com taxa intrínseca $$r = \beta - \gamma$$.
            ''', mathjax=True
        )
    ])


parametros = html.Div(
    [
        dcc.Markdown(
            r'''
            * **$$\beta$$ (Taxa de transmissão per capita / contatos infectantes):** Número médio de contatos efetivos com potencial de transmissão que um indivíduo realiza por unidade de tempo ($$ano^{-1}$$).
            * **$$\gamma$$ (Taxa de recuperação):** Velocidade de recuperação clínica ($$ano^{-1}$$), correspondendo ao inverso do período infeccioso médio ($$\tau_{inf} = 1/\gamma$$).
            * **$$\frac{\beta S I}{N}$$ (Termo de Transmissão por Incidência Padrão):** O fluxo de novas infecções por ano. Note que a probabilidade de um contato ser com um infectado é $$I/N$$, e a população vulnerável é $$S$$.
            * **$$S$$ (Suscetíveis):** Quantidade de indivíduos vulneráveis à infecção.
            * **$$I$$ (Infectados):** Quantidade de indivíduos infectados capazes de transmitir o patógeno.
            * **$$N = S + I$$:** População total sob estudo.
            * **$$R_0 = \frac{\beta}{\gamma}$$:** Número básico de reprodução (adimensional), que define se a infecção se expande ou se extingue.
            ''', mathjax=True
        )
    ]
)

cond_inicial = html.Div(
    [
        dcc.Markdown(
            r'''
            ### Configuração de Referência:
            * Taxa de transmissão: $$\beta = 0.90 \space ano^{-1}$$
            * Taxa de recuperação: $$\gamma = 0.50 \space ano^{-1}$$ (período infeccioso médio de $$2$$ anos)
            * Condições iniciais: $$S(0) = 100$$, $$I(0) = 1$$ ($$N = 101$$)

            ### Análise de Estabilidade e Pontos de Equilíbrio:
            Analisando $$\frac{dI}{dt} = 0$$, obtemos dois estados estacionários:
            1. **Equilíbrio Livre de Doença (DFE - Disease-Free Equilibrium):**
               $$I^* = 0, \quad S^* = N$$
               * **Estabilidade:** Globalmente assintoticamente estável se $$R_0 = \frac{\beta}{\gamma} \le 1$$.
            2. **Equilíbrio Endêmico (EE - Endemic Equilibrium):**
               $$S^* = N \frac{\gamma}{\beta} = \frac{N}{R_0}, \quad I^* = N \left(1 - \frac{\gamma}{\beta}\right) = N \left(1 - \frac{1}{R_0}\right)$$
               Em frações populacionais:
               $$s^* = \frac{1}{R_0}, \quad i^* = 1 - \frac{1}{R_0}$$
               * **Estabilidade:** Fisicamente admissível e assintoticamente estável sempre que $$R_0 > 1$$.

            ### Tempo Característico de Transição:
            O tempo característico de relaxamento da dinâmica até o equilíbrio é dado por:
            $$\tau = \frac{1}{|\beta - \gamma|}$$
            ''', mathjax=True
        )
    ]
)

perguntas = html.Div(
    [
        html.Div(
            [
                html.Div("1. Dinâmica de Equilíbrio de Referência (β = 0.90, γ = 0.50)", className="study-question-title"),
                dcc.Markdown(
                    r'''
                    Considere a situação de referência com $$\beta = 0.90 \space ano^{-1}$$ e $$\gamma = 0.50 \space ano^{-1}$$, partindo de $$S(0) = 100$$ e $$I(0) = 1$$ ($$N = 101$$):
                    * Calcule o número básico de reprodução: $$R_0 = \frac{\beta}{\gamma} = \frac{0.90}{0.50} = 1.80$$.
                    * Qual a proporção teórica de infectados no equilíbrio? $$(i^* = 1 - 1/R_0 = 1 - 1/1.80 \approx 44.4\%)$$.
                    * Quantos indivíduos suscetíveis ($$S^*$$) e infectados ($$I^*$$) estarão presentes no equilíbrio?
                    * Em quanto tempo o sistema atinge aproximadamente o estado de equilíbrio? *(Dica: avalie o tempo característico $$\tau = \frac{1}{|\beta - \gamma|} = \frac{1}{0.40} = 2.5 \space anos$$ e confira na curva gráfica entre $$3\tau$$ e $$5\tau$$)*.
                    ''',
                    mathjax=True
                )
            ],
            className="study-question-box"
        ),
        html.Div(
            [
                html.Div("2. Sensibilidade à Taxa de Recuperação (γ = 0.2, 0.4, 0.8 e 0.9)", className="study-question-title"),
                dcc.Markdown(
                    r'''
                    Mantendo a taxa de transmissão fixa em $$\beta = 0.90 \space ano^{-1}$$, utilize os botões de cenários rápidos para analisar os seguintes valores de $$\gamma$$:
                    * **$$\gamma = 0.20 \space ano^{-1}$$:** Período infeccioso longo ($$5$$ anos). Verifique que $$R_0 = 4.50$$ e $$i^* \approx 77.8\%$$. A convergência é mais rápida ou mais lenta?
                    * **$$\gamma = 0.40 \space ano^{-1}$$:** $$R_0 = 2.25$$, com $$i^* \approx 55.6\%$$.
                    * **$$\gamma = 0.80 \space ano^{-1}$$:** $$R_0 = 1.125$$, com $$i^* \approx 11.1\%$$.
                    * **$$\gamma = 0.90 \space ano^{-1}$$:** $$R_0 = 1.00$$. O que ocorre no limiar exato $$\beta = \gamma$$?
                    Observe como o aumento de $$\gamma$$ (recuperação mais rápida) comprime o reservatório endêmico de infectados e como o tempo para atingir o equilíbrio aumenta drasticamente quando $$\gamma \to \beta$$ (*desaceleração crítica*).
                    ''',
                    mathjax=True
                )
            ],
            className="study-question-box"
        ),
        html.Div(
            [
                html.Div("3. Limiar Crítico de Bifurcação e Extinção (R₀ ≤ 1)", className="study-question-title"),
                dcc.Markdown(
                    r'''
                    Teste o cenário de **Extinção** com $$\gamma = 1.20 \space ano^{-1}$$ ($$R_0 = \frac{0.90}{1.20} = 0.75 < 1$$):
                    * O que acontece com o único indivíduo inicialmente infectado ($$I(0) = 1$$)? A infecção cresce em algum momento?
                    * Por que quando $$\gamma > \beta$$, o número médio de casos secundários gerados por um infectado é estritamente menor que 1, tornando impossível a sustentação da doença na população?
                    ''',
                    mathjax=True
                )
            ],
            className="study-question-box"
        ),
        html.Div(
            [
                html.Div("4. Invariância em Relação ao Tamanho da População (N)", className="study-question-title"),
                dcc.Markdown(
                    r'''
                    No modelo com transmissão proporcional $$\frac{\beta S I}{N}$$, o limiar $$R_0 = \beta / \gamma$$ independe do tamanho absoluto da população:
                    * Ajuste o slider de $$S(0)$$ para $$200$$ ou $$400$$ indivíduos (com $$I(0) = 1$$).
                    * Mude o seletor de escala no topo do gráfico para **Proporção (%)**.
                    * Verifique se as curvas normalizadas e a fração endêmica ($$i^* = 44.4\%$$) se alteram. Por que essa formulação é mais realista para comunidades humanas agregadas do que a incidência pseudo-ação-de-massas pura ($$\beta S I$$)?
                    ''',
                    mathjax=True
                )
            ],
            className="study-question-box"
        ),
    ]
)


def _criar_acordeom(image_url):
    return html.Div(
        dbc.Accordion(
            [
                dbc.AccordionItem(_criar_secao_descricao(image_url), title="📘 Descrição e Formulação do Modelo"),
                dbc.AccordionItem(parametros, title="📊 Parâmetros do Modelo"),
                dbc.AccordionItem(cond_inicial, title="⚙️ Condições Iniciais e Pontos de Equilíbrio"),
                dbc.AccordionItem(perguntas, title="❓ Questões para Estudo e Exploração"),
            ],
            start_collapsed=True,
            flush=False,
        )
    )


# =============================================================================
# RODAPÉ INSTITUCIONAL (FOOTER COM LINK PARA O GITHUB)
# =============================================================================
rodape = html.Footer(
    html.Div(
        [
            html.Small(
                [
                    "Simulador do Modelo SIS • Desenvolvido com Dash, Plotly, SciPy e Bootstrap • ",
                    html.A(
                        [
                            html.I(className="fa-brands fa-github me-1"),
                            "Código Aberto no GitHub"
                        ],
                        href="https://github.com/fferreir/dash_sis",
                        target="_blank",
                        className="text-decoration-none text-muted fw-semibold"
                    )
                ],
                className="text-muted"
            )
        ],
        className="text-center py-3 mt-4 border-top"
    )
)


# =============================================================================
# FUNÇÃO PRINCIPAL DE CRIAÇÃO DO LAYOUT
# =============================================================================
def create_layout(app=None):
    """
    Constrói a árvore de componentes da aplicação Dash.
    """
    if app is not None and hasattr(app, 'get_asset_url'):
        image_url = app.get_asset_url('SIS.png')
    else:
        image_url = "assets/SIS.png"

    textos_descricao = _criar_acordeom(image_url)

    return dbc.Container(
        [
            cabecalho,
            cards_metricas,
            dbc.Row(
                [
                    # Coluna lateral de controles (4 colunas em telas grandes)
                    dbc.Col(
                        [
                            ajuste_cenarios,
                            ajuste_parametros,
                            ajuste_condicoes_iniciais,
                        ],
                        width=12, lg=4
                    ),

                    # Coluna principal do gráfico e documentação (8 colunas em telas grandes)
                    dbc.Col(
                        [
                            # Card do Gráfico com Seletor de Escala
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.I(className="fa-solid fa-chart-line me-2 text-primary"),
                                                    html.Span(
                                                        "Dinâmica Temporal do Modelo SIS",
                                                        className="fw-bold text-dark me-3"
                                                    ),
                                                ],
                                                className="d-flex align-items-center mb-1 mb-md-0"
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Escala:",
                                                        className="me-2 mb-0 small text-muted fw-semibold"
                                                    ),
                                                    dbc.RadioItems(
                                                        id="unit_display",
                                                        options=[
                                                            {"label": " Proporção (%)", "value": "pct"},
                                                            {"label": " Indivíduos", "value": "abs"},
                                                        ],
                                                        value="abs",
                                                        inline=True,
                                                        className="small"
                                                    ),
                                                ],
                                                className="d-flex align-items-center"
                                            )
                                        ],
                                        className="d-flex flex-wrap justify-content-between align-items-center mb-2 px-1"
                                    ),
                                    dcc.Loading(
                                        dcc.Graph(
                                            id='population_chart',
                                            style={'height': '540px'},
                                            config={
                                                "displayModeBar": True,
                                                "responsive": True,
                                                "modeBarButtonsToRemove": ["lasso2d", "select2d"]
                                            }
                                        ),
                                        type="circle"
                                    )
                                ],
                                className="graph-card mb-3"
                            ),

                            # Acordeom Pedagógico e Teórico logo abaixo do gráfico
                            html.Div(textos_descricao, className="mb-4")
                        ],
                        width=12, lg=8
                    ),
                ],
                className="g-3"
            ),
            rodape,
        ],
        fluid=True,
        className="py-2 px-3"
    )
