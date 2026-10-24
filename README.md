# 🦠 dash_sis: Simulador do Modelo SIS (Suscetível-Infectado-Suscetível)

Aplicação web interativa desenvolvida com **Dash**, **Plotly**, **Bootstrap Components** e **SciPy** para simulação, análise e visualização de dinâmicas epidemiológicas através do modelo compartimental determinístico **SIS** (*Suscetível – Infectado – Suscetível*) com **incidência padrão** ($\frac{\beta S I}{N}$), no qual indivíduos recuperados retornam imediatamente ao estado suscetível por ausência de imunidade duradoura.

---

## 📌 Sumário
- [Visão Geral](#-visão-geral)
- [Formulação Matemática](#-formulação-matemática)
- [Análise de Equilíbrio e $R_0$](#-análise-de-equilíbrio-e-r_0)
- [Parâmetros do Modelo](#-parâmetros-do-modelo)
- [Principais Funcionalidades](#-principais-funcionalidades)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Instalação e Execução](#-instalação-e-execução)
- [Testes Automatizados](#-testes-automatizados)
- [Roteiro de Estudo e Perguntas](#-roteiro-de-estudo-e-perguntas)
- [Licença](#-licença)

---

## 🔬 Visão Geral

O simulador modela infecções em que o hospedeiro, após a cura clínica, volta a ser suscetível sem adquirir imunidade protetora duradoura. Esse comportamento é típico de:
* **Infecções bacterianas recorrentes** (por exemplo, estreptococos de vias aéreas, otites e infecções urinárias);
* **Doenças Sexualmente Transmissíveis (DSTs)** causadas por bactérias, como gonorreia e clamídia;
* **Certas infecções parasitárias ou superficiais**.

### Destaques da Implementação:
1. **Termo de Transmissão por Incidência Padrão ($\frac{\beta S I}{N}$):** Formulação baseada na probabilidade de encontro com indivíduos infectados ($I/N$), tornando a taxa de reprodução $R_0 = \frac{\beta}{\gamma}$ e as frações de equilíbrio invariantes ao tamanho populacional total $N$.
2. **Design System Moderno:** Interface responsiva com Hero Header em gradiente, cards de métricas (KPIs), controles organizados com badges reativos e paleta de cores harmoniosa baseada no [dash-sir](https://github.com/fferreir/dash-sir).
3. **Cenários Didáticos Instantâneos:** Botões de predefinição para carregar em um clique as configurações exatas das questões de estudo (Q1, Q2a, Q2b, Q2c, Q2d e Extinção).
4. **Indicadores em Tempo Real (KPIs):**
   - Número Básico de Reprodução ($R_0$);
   - Prevalência de pico e tempo de ocorrência;
   - Nível endêmico de infectados no equilíbrio ($I^*$ e $i^*\%$);
   - Nível de suscetíveis no equilíbrio ($S^*$ e $s^*\%$).
5. **Visualização Flexível:** Alternância dinâmica entre proporção percentual da população inicial (%) e contagem absoluta de indivíduos.
6. **Integração Numérica e Analítica:** Cálculo numérico preciso via solver implícito `Radau` (`scipy.integrate.solve_ivp`), combinado com linhas de referência teóricas para os pontos de equilíbrio.
7. **Rodapé Institucional:** Rodapé estilizado com identificação e link direto para o repositório aberto no GitHub.

---

## 📐 Formulação Matemática

O modelo SIS considera uma população fechada de tamanho constante $N = S + I$, desconsiderando natalidade e mortalidade demográfica.

$$\begin{aligned}
\frac{dS}{dt} &= -\frac{\beta S I}{N} + \gamma I \\
\frac{dI}{dt} &= \frac{\beta S I}{N} - \gamma I = \left(\frac{\beta S}{N} - \gamma\right) I
\end{aligned}$$

### Conservação da População Total:
Somando as duas equações diferenciais:

$$\frac{d(S + I)}{dt} = \frac{dN}{dt} = 0 \implies S(t) + I(t) = N = S(0) + I(0)$$

Em frações da população ($s = S/N$ e $i = I/N$, com $s + i = 1$):

$$\frac{di}{dt} = \beta (1 - i) i - \gamma i = (\beta - \gamma) i \left(1 - \frac{\beta}{\beta - \gamma} i\right)$$

que corresponde formalmente à equação logística de crescimento populacional com taxa intrínseca $r = \beta - \gamma$ e capacidade de suporte fracionária $K = 1 - \frac{\gamma}{\beta} = 1 - \frac{1}{R_0}$.

---

## ⚖️ Análise de Equilíbrio e $R_0$

A condição para a expansão inicial da epidemia a partir de $I(0) \approx 0$ ($S(0) \approx N$) é:

$$\left.\frac{dI}{dt}\right|_{t=0} = (\beta - \gamma) I_0 > 0 \iff R_0 = \frac{\beta}{\gamma} > 1$$

Fazendo $\frac{dI}{dt} = 0$, encontramos dois estados estacionários:

1. **Equilíbrio Livre de Doença (Disease-Free Equilibrium - DFE):**
   $$I^* = 0, \quad S^* = N$$
   * É globalmente assintoticamente estável se $R_0 = \frac{\beta}{\gamma} \le 1$.

2. **Equilíbrio Endêmico (Endemic Equilibrium - EE):**
   $$S^* = N \frac{\gamma}{\beta} = \frac{N}{R_0}, \quad I^* = N \left(1 - \frac{\gamma}{\beta}\right) = N \left(1 - \frac{1}{R_0}\right)$$
   Em termos fracionários:
   $$s^* = \frac{1}{R_0}, \quad i^* = 1 - \frac{1}{R_0}$$
   * É fisicamente admissível e assintoticamente estável sempre que $R_0 > 1$.

O tempo característico de relaxamento até o estado estacionário é dado por:

$$\tau = \frac{1}{|\beta - \gamma|}$$

---

## ⚙️ Parâmetros do Modelo

| Símbolo | Nome do Parâmetro | Unidade | Descrição | Valor Padrão |
| :---: | :--- | :---: | :--- | :---: |
| **$\beta$** | Taxa de Contatos Infectantes | $\text{ano}^{-1}$ | Frequência per capita de contatos eficazes com potencial de transmissão | $0.90$ |
| **$\gamma$** | Taxa de Recuperação | $\text{ano}^{-1}$ | Inverso do período infeccioso médio ($\tau_{inf} = 1/\gamma$) | $0.50$ |
| **$S(0)$** | Suscetíveis Iniciais | indivíduos | Tamanho inicial da população suscetível | $100$ |
| **$I(0)$** | Infectados Iniciais | indivíduos | População infectada no início da epidemia | $1$ |
| **$t_{fim}$** | Horizonte Temporal | anos | Tempo total de simulação numérica | $100$ |

---

## 🌟 Principais Funcionalidades

- **Sliders com Badges Dinâmicos:** Controle fino dos parâmetros com badges de valores em tempo real.
- **Cenários Pré-configurados:** Carregamento rápido dos exercícios propostos com um clique.
- **Gráfico Interativo Plotly:** Linhas de evolução temporal suaves, linhas de equilíbrio assintótico pontilhadas ($S^*, I^*$) e modo hover unificado.
- **Acordeom Didático Completo:**
  - 📘 Descrição conceitual e diagrama visual compartimental ([assets/SIS.png](assets/SIS.png)).
  - 📊 Detalhamento de grandezas, variáveis e unidades.
  - ⚙️ Demonstração da estabilidade dos equilíbrios e do limiar $R_0$.
  - ❓ Questões para exploração guiada.
- **Compatibilidade Local e Deploy:** Roteamento configurado para `/dash_sis/` com redirecionamento amigável da rota raiz `/`.
- **Rodapé com Link para GitHub:** Link oficial para [https://github.com/fferreir/dash_sis](https://github.com/fferreir/dash_sis).

---

## 📁 Estrutura do Projeto

O projeto segue a arquitetura modular estabelecida em [dash-sir](https://github.com/fferreir/dash-sir):

```text
dash_sis/
├── app.py                 # Ponto de entrada da aplicação Dash e configuração de rotas
├── assets/
│   ├── estilos.css        # Folha de estilos customizada (Design system, KPIs, Hero)
│   └── SIS.png            # Diagrama do modelo compartimental SIS com termo beta*S*I/N
├── pytest.ini             # Configuração do pytest (pythonpath = .)
├── requirements.txt       # Dependências do projeto (Dash, Bootstrap, SciPy, Plotly, pytest)
├── LICENSE                # Licença GNU GPLv3
├── README.md              # Documentação completa e guia de uso
├── src/
│   ├── __init__.py        # Inicialização do pacote src
│   ├── model.py           # EDOs, solver Radau e cálculo de métricas analíticas
│   ├── layout.py          # Interface visual, KPIs, acordeom pedagógico e rodapé institucional
│   └── callbacks.py       # Callbacks reativos (cenários rápidos, badges e gráfico)
└── tests/
    ├── __init__.py        # Inicialização do módulo de testes
    ├── test_app.py        # Testes de integração de interface, rodapé e limites numéricos
    └── test_model.py      # Testes unitários das EDOs, conservação S+I=N e equilíbrios
```

---

## 🚀 Instalação e Execução

### Pré-requisitos
- Python 3.10 ou superior.

### 1. Clonar o Repositório
```bash
git clone https://github.com/fferreir/dash_sis.git
cd dash_sis
```

### 2. Criar e Ativar Ambiente Virtual

Com `venv` padrão:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ou com [`uv`](https://docs.astral.sh/uv/):
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Executar a Aplicação
```bash
python app.py
```

Abra no navegador:
- [http://127.0.0.1:8050/](http://127.0.0.1:8050/) (redireciona para `/dash_sis/`) ou diretamente em [http://127.0.0.1:8050/dash_sis/](http://127.0.0.1:8050/dash_sis/).

---

## 🧪 Testes Automatizados

Execute a suíte de testes com `pytest`:

```bash
pytest -v
```

Todos os testes validam:
- Conservação estrita de população $S(t) + I(t) = N_0$;
- Convergência numérica ao equilíbrio analítico $(S^*, I^*)$;
- Invariância de escala populacional com o termo $\frac{\beta S I}{N}$;
- Cálculo correto de $R_0 = \beta / \gamma$ e status epidemiológico;
- Estrutura dos componentes de interface e integridade do link do rodapé para o GitHub.

---

## 📚 Roteiro de Estudo e Perguntas

1. **Dinâmica de Equilíbrio de Referência ($\beta=0.90$, $\gamma=0.50$):**
   - Com $S(0) = 100$ e $I(0) = 1$ ($N = 101$), observe a evolução temporal.
   - *Calcule $R_0 = \beta / \gamma = 0.90 / 0.50 = 1.80$.*
   - *Qual a fração teórica de infectados no equilíbrio? ($i^* = 1 - 1/R_0 = 1 - 1/1.80 \approx 44.4\%$).*
   - *Quantos indivíduos infectados ($I^*$) e suscetíveis ($S^*$) estarão presentes no equilíbrio?*
   - *Estime o tempo de estabilização usando o tempo característico $\tau = \frac{1}{|\beta - \gamma|} = \frac{1}{0.40} = 2.5 \space \text{anos}$ (verifique que o equilíbrio é alcançado entre $3\tau$ e $5\tau$, ou seja, $8$ a $12$ anos).*

2. **Sensibilidade à Taxa de Recuperação ($\gamma = 0.20, 0.40, 0.80 \text{ e } 0.90 \space \text{ano}^{-1}$):**
   - Mantendo $\beta = 0.90 \space \text{ano}^{-1}$, compare os casos $\gamma = 0.20, 0.40, 0.80$ e $0.90$.
   - *Como a duração média da infecção ($1/\gamma$) afeta a proporção de infectados no equilíbrio ($i^*$)?*
   - *O que acontece com a velocidade de convergência à medida que $\gamma \to \beta = 0.90$ (desaceleração crítica)?*

3. **Limiar Crítico de Bifurcação e Extinção ($R_0 \le 1$):**
   - Teste o cenário com $\gamma = 0.90$ ($R_0 = 1.0$) e com $\gamma = 1.20$ ($R_0 = 0.75 < 1$).
   - *Por que a infecção não consegue se estabelecer na população quando $\gamma > \beta$?*
   - *O que ocorre com o caso inicial $I(0) = 1$?*

4. **Invariância em Relação ao Tamanho da População ($N$):**
   - Ajuste o slider de $S(0)$ para $200$ ou $400$ indivíduos (mantendo $I(0) = 1$).
   - Mude a escala do gráfico para **Proporção (%)**.
   - *Por que as proporções de equilíbrio ($s^* = 55.6\%$, $i^* = 44.4\%$) e a taxa $R_0$ permanecem exatamente as mesmas, independentemente do valor de $N$?*

---

## 📄 Licença

Este projeto é distribuído sob a licença **GNU General Public License v3.0**. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
