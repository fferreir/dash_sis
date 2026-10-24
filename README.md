# 🦠 dash_sis: Simulador do Modelo SIS (Suscetível-Infectado-Suscetível)

Aplicação web interativa desenvolvida com **Dash**, **Plotly**, **Bootstrap Components** e **SciPy** para simulação, análise e visualização de dinâmicas epidemiológicas através do modelo compartimental determinístico **SIS** (*Suscetível – Infectado – Suscetível*), no qual indivíduos recuperados retornam imediatamente ao estado suscetível por ausência de imunidade duradoura.

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

O simulador modela infecções em que o hospedeiro, após a cura clínica, volta a ser suscetível sem adquirir imunidade duradoura. Esse comportamento é típico de:
* **Infecções bacterianas recorrentes** (por exemplo, estreptococos de vias aéreas, otites e infecções urinárias);
* **Doenças Sexualmente Transmissíveis (DSTs)** causadas por bactérias, como gonorreia e clamídia;
* **Certas infecções parasitárias ou superficiais**.

### Destaques da Implementação:
1. **Design System Moderno:** Interface responsiva com Hero Header em gradiente, cards de métricas (KPIs), controles organizados com badges reativos e paleta de cores harmoniosa.
2. **Cenários Didáticos Instantâneos:** Botões de predefinição para carregar em um clique as configurações exatas das questões de estudo (Q1, Q2a, Q2b, Q2c, Q2d e Extinção).
3. **Indicadores em Tempo Real (KPIs):**
   - Número Básico de Reprodução ($R_0$);
   - Pico máximo de infectados e tempo de ocorrência;
   - Nível endêmico de infectados no equilíbrio ($I^*$);
   - Nível de suscetíveis no equilíbrio ($S^*$).
4. **Visualização Flexível:** Alternância dinâmica entre proporção percentual da população inicial (%) e contagem absoluta de indivíduos.
5. **Integração Numérica e Analítica:** Cálculo numérico preciso via solver implícito `Radau` (`scipy.integrate.solve_ivp`), combinado com linhas de referência teóricas para os pontos de equilíbrio.
6. **Rodapé Institucional:** Rodapé estilizado com identificação e link direto para o repositório aberto no GitHub.

---

## 📐 Formulação Matemática

O modelo SIS considera uma população fechada de tamanho constante $N = S + I$, desconsiderando natalidade e mortalidade demográfica.

$$\begin{aligned}
\frac{dS}{dt} &= -\beta S I + \gamma I \\
\frac{dI}{dt} &= \beta S I - \gamma I = (\beta S - \gamma) I
\end{aligned}$$

### Conservação da População Total:
Somando as duas equações diferenciais:

$$\frac{d(S + I)}{dt} = \frac{dN}{dt} = 0 \implies S(t) + I(t) = N = S(0) + I(0)$$

---

## ⚖️ Análise de Equilíbrio e $R_0$

A taxa de crescimento inicial dos infectados é positiva se e somente se:

$$\left.\frac{dI}{dt}\right|_{t=0} = (\beta S_0 - \gamma) I_0 > 0 \iff \beta N > \gamma \iff R_0 = \frac{\beta N}{\gamma} > 1$$

Fazendo $\frac{dI}{dt} = 0$, encontramos dois estados estacionários:

1. **Equilíbrio Livre de Doença (Disease-Free Equilibrium - DFE):**
   $$I^* = 0, \quad S^* = N$$
   * É globalmente assintoticamente estável quando $R_0 \le 1$.

2. **Equilíbrio Endêmico (Endemic Equilibrium - EE):**
   $$S^* = \frac{\gamma}{\beta}, \quad I^* = N - \frac{\gamma}{\beta} = N \left(1 - \frac{1}{R_0}\right)$$
   * É fisicamente admissível e assintoticamente estável quando $R_0 > 1$.

O tempo característico de relaxamento até o estado estacionário é dado por:

$$\tau = \frac{1}{|\beta N - \gamma|}$$

---

## ⚙️ Parâmetros do Modelo

| Símbolo | Nome do Parâmetro | Unidade | Descrição | Valor Padrão |
| :---: | :--- | :---: | :--- | :---: |
| **$\beta$** | Taxa de Contatos Infectantes | $\text{ano}^{-1}$ | Frequência de contatos com potencial de transmissão | $0.009$ |
| **$\gamma$** | Taxa de Recuperação | $\text{ano}^{-1}$ | Inverso do período infeccioso médio ($\tau_{inf} = 1/\gamma$) | $0.50$ |
| **$S(0)$** | Suscetíveis Iniciais | indivíduos | Tamanho inicial da população suscetível | $100$ |
| **$I(0)$** | Infectados Iniciais | indivíduos | População infectada no início da epidemia | $1$ |
| **$t_{fim}$** | Horizonte Temporal | anos | Tempo total de simulação numérica | $100$ |

---

## 🌟 Principais Funcionalidades

- **Sliders com Badges Dinâmicos:** Controle fino dos parâmetros com badges de valores em tempo real.
- **Cenários Pré-configurados:** Carregamento rápido dos exercícios propostos com um clique.
- **Gráfico Interativo Plotly:** Linhas de tendência temporal suaves, linhas de equilíbrio assintótico pontilhadas e modo hover unificado.
- **Acordeom Didático Completo:**
  - 📘 Descrição conceitual e diagrama visual compartimental (`assets/SIS.png`).
  - 📊 Detalhamento de grandezas, variáveis e unidades.
  - ⚙️ Demonstração da estabilidade dos equilíbrios e do limiar $R_0$.
  - ❓ Questões para exploração guiada.
- **Compatibilidade Local e Deploy:** Roteamento configurado para `/dash_sis/` com redirecionamento amigável da rota raiz `/`.
- **Rodapé com Link para GitHub:** Link oficial para [https://github.com/fferreir/dash_sis](https://github.com/fferreir/dash_sis).

---

## 📁 Estrutura do Projeto

O projeto segue a arquitetura modular de boas práticas do ecossistema Dash (idêntica ao [dash-sir](https://github.com/fferreir/dash-sir)):

```text
dash_sis/
├── app.py                 # Ponto de entrada da aplicação Dash e configuração de rotas
├── assets/
│   ├── estilos.css        # Folha de estilos customizada (Design system, KPIs, Hero)
│   └── SIS.png            # Diagrama do modelo compartimental SIS
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
    └── test_model.py      # Testes unitários das EDOs, lei de conservação e equilíbrios
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

> **Nota de Deploy:** Para configurar subcaminhos em servidores com proxy reverso (Nginx/Traefik), utilize:
> ```bash
> export DASH_REQUESTS_PATHNAME_PREFIX="/dash_sis/"
> ```

---

## 🧪 Testes Automatizados

Execute a suíte de testes com `pytest`:

```bash
pytest -v
```

Todos os testes validam:
- Conservação de população $S(t) + I(t) = N_0$;
- Convergência numérica ao equilíbrio analítico $(S^*, I^*)$;
- Cálculo correto de $R_0$ e status epidemiológico;
- Estrutura dos componentes de interface e integridade do link do rodapé para o GitHub.

---

## 📚 Roteiro de Estudo e Perguntas

1. **Equilíbrio de Referência ($\beta=0.009$, $\gamma=0.5$):**
   - Com $S(0) = 100$ e $I(0) = 1$ ($N = 101$), observe a evolução temporal.
   - *Qual o valor de $R_0$? Quais os valores finais de $S^*$ e $I^*$ no equilíbrio? Em quantos anos a estabilização ocorre?*
   - *(Dica: $S^* = 0.5/0.009 \approx 55.56$, $I^* \approx 45.44$).*

2. **Sensibilidade à Taxa de Recuperação ($\gamma$):**
   - Varie $\gamma$ para $0.2$, $0.4$, $0.8$ e $0.9 \space \text{ano}^{-1}$ usando os botões de cenários rápidos.
   - *Como o tempo de permanência no estado infeccioso ($\tau = 1/\gamma$) altera o tamanho da população endêmica de infectados?*

3. **Limiar Crítico de Extinção:**
   - Calcule o valor crítico $\gamma_c = \beta N = 0.009 \times 101 \approx 0.909 \space \text{ano}^{-1}$.
   - *O que acontece quando $\gamma \ge 0.909$ ($R_0 \le 1$)? A doença consegue se sustentar na população?*

---

## 📄 Licença

Este projeto é distribuído sob a licença **GNU General Public License v3.0**. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
