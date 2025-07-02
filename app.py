import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy import heaviside
from textwrap import dedent
import plotly.graph_objects as go


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN, dbc.icons.FONT_AWESOME], requests_pathname_prefix='/dash_sis/')
server = app.server

cabecalho = html.H1("Modelo SIS",className="bg-primary text-white p-2 mb-4")

descricao = dcc.Markdown(
    '''
    É apresentado um modelo determinístico do tipo *SIS*. Neste exercício, consideramos infecções para as quais o indivíduo,
        algum tempo após ter passado pelo processo de infecção, volta a ser suscetível.
        É o caso de algumas doenças de transmissão sexual.
        Também é o caso de algumas infecções bacterianas.
        Este é um modelo simples, que não considera alterações na população por natalidade
        e mortalidade. O que é equivalente a considerarmos uma população fixa.
    ''', mathjax=True
)

parametros = dcc.Markdown(
    '''
    * taxa de contatos potencialmente infectantes ($$\\beta$$)
    * taxa de recuperação = inverso do período infeccioso ($$\\gamma$$)
    * $$S$$: número de indivíduos suscetíveis
    * $$I$$: número de indivíduos infectados
    ''', mathjax=True
)
cond_inicial = dcc.Markdown(
    '''
    * taxa de contatos: $$\\beta=0.009 \\text{ ano}^{-1}$$
    * taxa de recuperação: $$\\gamma=0.5 \\text{ ano}^{-1}$$
    * condições iniciais: $$S(0)=100$$, $$I(0)=1$$
    ''', mathjax=True
)

perguntas = dcc.Markdown(
    '''
    1. Considere uma situação em que a taxa de contatos seja baixa (por exemplo, $$\\beta=0.009$$, com um número inicial
    de $$100$$ suscetíveis e $$1$$ infectado. Para $$\\gamma=0.5$$, qual o número de suscetíveis e infectados no equilíbrio?
    Qual o tempo necessário para se atingir o equilíbrio?
    2. Repita o item 1, para $$\\gamma=0.2$$, $$0.4$$, $$0.8$$ e $$0.9$$.
    ''', mathjax=True
)

textos_descricao = html.Div(
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    descricao, title="Descrição do modelo"
                ),
                dbc.AccordionItem(
                    parametros, title="Parâmetros do modelo"
                ),
                dbc.AccordionItem(
                    cond_inicial, title="Condições iniciais"
                ),
                dbc.AccordionItem(
                    perguntas, title="Perguntas"
                ),
            ],
            start_collapsed=True,
        )
    )

ajuste_condicoes_iniciais = html.Div(
        [
            html.P("Ajuste das condições iniciais", className="card-header border-dark mb-3"),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''$$S$$ total de suscetíveis''', mathjax=True), html_for="s_init"),
                    dcc.Slider(id="s_init", min=0, max=100, value=100, tooltip={"placement": "bottom", "always_visible": False}),
                ],
                className="m-2",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''$$I$$ total de infectados ''', mathjax=True), html_for="i_init"),
                    dcc.Slider(id="i_init", min=0, max=100, value=1, tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),

        ],
        className="card border-dark mb-3",
    )

ajuste_parametros = html.Div(
        [
            html.P("Ajuste dos parâmetros", className="card-header border-dark mb-3"),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''$$\\beta$$ Taxa de contatos potencialmente infectantes:''', mathjax=True), html_for="beta"),
                    dcc.Slider(id="beta", min=0.001, max=0.05, value=0.009, tooltip={"placement": "bottom", "always_visible": False}),
                ],
                className="m-2",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''$$\\gamma$$ Taxa de recuperação ''', mathjax=True), html_for="gamma"),
                    dcc.Slider(id="gamma", min=0.1, max=0.9, value=0.5, tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),
        ],
        className="card border-dark mb-3",
    )

def ode_sys(t, state, beta, gamma):
    s, i=state
    ds_dt=-beta*s*i+gamma*i
    di_dt=beta*s*i-gamma*i
    return [ds_dt, di_dt]

@app.callback(Output('population_chart', 'figure'),
              [Input('s_init', 'value'),
              Input('i_init', 'value'),
              Input('beta', 'value'),
              Input('gamma', 'value')])
def gera_grafico(s_init, i_init, beta, gamma):
    t_begin = 0.
    t_end = 100.
    t_span = (t_begin, t_end)
    t_nsamples = 10000
    t_eval = np.linspace(t_begin, t_end, t_nsamples)
    sol = solve_ivp(fun=ode_sys,
                    t_span=t_span, 
                    y0=[s_init, i_init],
                    args=(beta, gamma),
                    t_eval=t_eval,
                    method='Radau')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name='Suscetível',
                             line=dict(color='#00b400', width=4)))
    fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name ='Infectado',
                             line=dict(color='#ff0000', width=4, dash='dot')))
    fig.update_layout(title='Dinâmica Modelo SIS',
                       xaxis_title='Tempo (anos)',
                       yaxis_title='Proporção da população')
    return fig

app.layout = dbc.Container([
                cabecalho,
                dbc.Row([
                        dbc.Col(html.Div(ajuste_parametros), width=3),
                        dbc.Col(html.Div([ajuste_condicoes_iniciais,html.Div(textos_descricao)]), width=3),
                        dbc.Col(dcc.Graph(id='population_chart', className="shadow-sm rounded-3 border-primary",
                                style={'height': '500px'}), width=6),
                ]),
              ], fluid=True),


if __name__ == '__main__':
    app.run(debug=False)
