# Plano de Upgrade do Dashboard — 18 Fases

> Documento de planejamento para expandir o dashboard Streamlit do Titanium
> Alpha com gráficos "LinkedIn quant grade". Cada fase é independente,
> pode ser implementada isoladamente e entregar valor visual imediato.

---

## 1. Contexto e Objetivo

### Estado atual (sessão 42)
O dashboard atual (`src/dashboard/app.py`, ~2150 linhas) já cobre o básico:

- **Aba Benchmark:** equity curve (linear/log), drawdown, 16 métricas de
  performance, rolling Sharpe com janela configurável, heatmap de pesos
  no tempo, link para PDF report.
- **Aba Performance:** cards de métricas, tabela de decisões, donut de
  concentração, comparação raw vs tilted, confidence histogram, action
  pie, deltas de peso vs último rebalance.
- **Aba War Room:** replay animado do debate, live debate com streaming
  por nó LangGraph, cards de report por agente, decisão final.
- **Aba Microstructure:** fan chart de quantis do PatchTST, histórico de
  pesos por ticker, cumulative return vs SPY.

O dashboard é funcional e coerente, mas está **fortemente centrado em
séries temporais simples**. Falta o ferramental que caracteriza posts
quant de alto engajamento: **distribuições, path-dependency, decomposições
de retorno e diagnósticos estatísticos.**

### Objetivo deste plano
Posicionar o dashboard como **peça central de portfolio no LinkedIn**,
adicionando 18 visualizações organizadas em dois tiers:

- **Tier 1 (Fases 1-9):** clássicos LinkedIn — o que todo post quant tem
  e o Titanium ainda não.
- **Tier 2 (Fases 10-18):** diferenciadores únicos — o que expõe a
  sofisticação real do pipeline (CPCV, LangGraph, PatchTST, RAG).

Cada fase é projetada para ~30-90 minutos de implementação e pode ser
entregue num commit próprio (`feat: dashboard phase N — <chart>`).

---

## 2. Padrões Técnicos Globais

Toda fase deve respeitar os padrões já estabelecidos:

### Cores e estilo
- Background: `#0E1117` (tema dark Streamlit padrão)
- Accent blue: `_ACCENT_BLUE = "#2E86DE"` (já exposto em `app.py`)
- Accent amber (warning): `#FFC837`
- Green (positivo): `#2ECC71`
- Red (negativo): `#E74C3C`
- Neutro/texto secundário: `#AAAAAA`, `#888888`
- Divisores: `#333333`
- Cores por ticker: usar `_ticker_color(idx)` que já existe em `app.py:95`

### Convenções de código
- Funções de chart prefixadas com `_chart_*` (ex: `_chart_calendar_heatmap`)
- Funções de render com efeito colateral Streamlit prefixadas com `_render_*`
- Type hints obrigatórios (ver `CLAUDE.md`)
- Docstrings Google Style em função pública
- Polars (nunca Pandas) para transformação de dados
- Plotly (nunca Matplotlib) para renderização
- `st.plotly_chart(fig, width="stretch")` — padrão do projeto
- Logging com loguru se houver erro recuperável

### Dados disponíveis
Arquivos que já existem em `data/outputs/`:

| Arquivo | Conteúdo | Loader |
|---|---|---|
| `benchmark_equity.parquet` | `date, portfolio_value, benchmark_value` (walk-forward) | `load_benchmark_equity()` |
| `benchmark_weights.parquet` | `date × ticker → weight` | `load_benchmark_weights()` |
| `benchmark_metrics.json` | 16 métricas agregadas | `load_benchmark_metrics()` |
| `decisions.json` | Decisão agêntica atual + metadata | `load_decisions()` |
| `debate_history.json` | Reports dos 4 agentes LangGraph | `load_debate_history()` |
| `predictions.json` | PatchTST quantis mais recentes | `load_predictions()` |
| `forecast.json` | Fan chart por ticker (quantis t+1…t+H) | `load_forecast()` |
| `benchmark_report.pdf` | Report de 6 páginas | — |

**Dados que ainda não existem e podem precisar ser gerados** (flagged em
cada fase onde forem necessários):

- Histórico de decisões agênticas (apenas o último run persiste hoje)
- CPCV-OOS path-level equity curves (só Sharpe agregado é salvo)
- Logs de retrieval RAG (citações por ticker/sessão)
- Série temporal de leverage/realized vol do walk-forward
- Histórico de predições PatchTST alinhado com outcomes realizados

Quando uma fase depende de dado inexistente, a **etapa 0** dela é
persistir esse dado (geralmente editando `src/backtest/walk_forward.py`,
`src/backtest/cpcv_oos.py` ou `src/agents/pipeline.py`).

### Testes
- Criar `tests/unit/dashboard/test_phase_N_<chart>.py` (diretório criado a
  partir da Fase 1; contém `__init__.py` vazio para consistência com
  `tests/__init__.py`).
- Mock de parquet/json com dados sintéticos (10-20 linhas)
- Teste mínimo: função retorna `go.Figure` sem exception, com ≥1 trace
- Para helpers de cálculo (ex: up-capture, effective N): testes
  numéricos com valores conhecidos
- Rodar `poetry run pytest tests/unit/dashboard/ -q` antes de commitar

### Critério de aceite genérico (todas as fases)
1. Gráfico renderiza sem warning em dados reais.
2. Label de eixo, título e hover tooltip em inglês (consistência com
   resto do dashboard — decisões, metrics, etc. estão todos em inglês).
3. Funciona em dark mode sem cores hardcoded claras.
4. Degrada graciosamente quando dados estão ausentes (usa `st.info`
   com mensagem acionável, não silencia nem quebra).
5. Pelo menos 1 teste unitário novo.
6. Screenshot anexado ao commit ou ao PR (ajuda review).

---

## 3. Sequenciamento Recomendado

| Ordem | Fase | Tier | Esforço | Impacto visual |
|---|---|---|---|---|
| 1 | Fase 1 — Calendar heatmap | 1 | S | Muito alto |
| 2 | Fase 10 — CPCV path spaghetti | 2 | M | Muito alto |
| 3 | Fase 12 — Agent vote matrix | 2 | M | Muito alto |
| 4 | Fase 3 — Distribution + QQ | 1 | M | Alto |
| 5 | Fase 5 — Up/Down capture | 1 | S | Alto |
| 6 | Fase 2 — Top-10 drawdowns | 1 | S | Alto |
| 7 | Fase 11 — Sharpe violin CPCV | 2 | S | Alto |
| 8 | Fase 15 — PatchTST calibration | 2 | M | Alto |
| 9 | Fase 4 — Rolling beta/alpha | 1 | S | Médio |
| 10-18 | Resto | — | — | — |

**Entrega mínima viável para LinkedIn (MVP):** fases 1, 10, 12 +
screenshot composto. Daria 3 gráficos impactantes que capturam os
três pilares do projeto (retornos, validação estatística, camada
agêntica).

---

---

# TIER 1 — Clássicos LinkedIn

---

## Fase 1 — Calendar Heatmap de Retornos Mensais

### Objetivo
Produzir o gráfico quant mais compartilhado do LinkedIn: matriz 12 meses
× N anos colorida por retorno mensal (verde/vermelho, escala divergente
em 0). Mostra sazonalidade, consistência e "anos ruins" de relance.

### Aba de destino
Aba **Benchmark**, logo após o drawdown chart, antes das métricas.

### Dados necessários
- `benchmark_equity.parquet` (`date`, `portfolio_value`, `benchmark_value`)
  — já existe.

### Sketch de implementação

```python
def _compute_monthly_returns(equity_df: pl.DataFrame) -> pl.DataFrame:
    """Return (year, month, port_ret, spy_ret) from daily equity series."""
    # Derive daily simple returns from portfolio_value / benchmark_value,
    # then compound inside each (year, month) bucket as prod(1+r) - 1.
    # This is robust for partial first/last months (first daily return is
    # dropped so no uncovered month becomes 0% by accident).

def _chart_calendar_heatmap(equity_df: pl.DataFrame) -> go.Figure:
    """12-column heatmap: rows=years, cols=Jan..Dec, color=ret."""
    # go.Heatmap with colorscale="RdYlGn" (green=positive, red=negative),
    # zmid=0, zmin=-0.15, zmax=0.15. Do NOT use RdYlGn_r — with zmid=0 the
    # non-reversed scale already maps positive→green / negative→red.
    # Annotate each cell with "+2.3%" / "-4.1%" text in white.
    # Append an "Annual" column with full-year compounded return and a
    # "Mean" row at the bottom (seasonality).
```

### Cuidados
- **Anos parciais:** o primeiro e o último ano do backtest são parciais.
  Marcar meses não cobertos como `None` (cinza) e não `0` (verde errado).
- **Escala divergente:** `zmid=0` é mandatório ou o contraste fica errado.
- **Ordem de linhas:** ano mais recente no topo (convenção CTA / Two
  Sigma posts).

### Critério de aceite
- Cada célula tem texto "+X.X%" legível.
- Coluna "Annual" à direita somando retorno do ano.
- Linha de "Média mensal" no rodapé (bônus).
- Tooltip mostra `year=2023, month=Mar, return=+2.3%`.

### Esforço
S (1-2h). Matemática trivial, só resample + pivot + heatmap.

### Referências visuais
- `quantstats.plots.monthly_heatmap` — estilo de referência.

---

## Fase 2 — Top-10 Drawdowns Ranked

### Objetivo
Barra horizontal dos 10 piores drawdowns, ordenados por magnitude, com
cada barra mostrando `start_date → end_date → recovery_date` e duração em
dias. Complementa o drawdown chart atual (que mostra só a série contínua).

### Aba de destino
Aba **Benchmark**, após o drawdown chart atual.

### Dados necessários
- `benchmark_equity.parquet` — já existe.
- Função de detecção de drawdowns (ver abaixo).

### Sketch de implementação

```python
def _detect_drawdown_periods(
    equity: list[float],
    dates: list[str],
) -> list[dict]:
    """Return drawdown events sorted by max depth.

    Each event: {start, trough, end, depth, duration_days, recovery_days}
    An event ends when equity reclaims prior peak (or series ends).
    """
    # Walk series, track running max. When equity < max, we're in DD.
    # When equity returns to prior max, close event.

def _chart_top_drawdowns(equity_df: pl.DataFrame, n: int = 10) -> go.Figure:
    """Horizontal bar: DD1 | DD2 | ... sorted by depth."""
    # x-axis: depth (negative)
    # y-axis: labeled "2022-01 → 2023-03 (148d)"
    # Color: depth (darker = deeper)
```

### Cuidados
- **Drawdowns não-recuperados:** se a série termina em DD (caso atual do
  walk-forward), marcar com asterisco "ongoing".
- **Mínimo de 1% de profundidade:** evitar spam de micro-drawdowns.
- **Overlap:** usar apenas peak-to-trough único (definição padrão, ver
  `src/backtest/benchmark_metrics.py` se já houver).
- **Trading days, não calendar days:** medir `duration_days` e
  `recovery_days` como diferença de índice no array de equity (a série é
  em business-day grid). Calcular via `(later - earlier).days` puxa o
  gap de fim de semana/feriado e gera números inconsistentes com o resto
  do dashboard, que opera em dias úteis.

### Critério de aceite
- 10 barras ordenadas por magnitude.
- Cada barra: label com datas + duração.
- Tooltip com depth %, duration days, recovery days.
- Bônus: tabela abaixo com as mesmas 10 linhas detalhadas.

### Esforço
S (2h). Lógica de detecção é simples; cuidado com edge case de série
terminando em DD.

---

## Fase 3 — Return Distribution + QQ Plot

### Objetivo
Dois sub-plots lado a lado:
1. Histograma de retornos diários com overlay de normal de mesma
   média/desvio, VaR 5% e CVaR 5% sombreados na cauda esquerda.
2. QQ-plot (quantis empíricos vs teóricos de normal) para visualizar
   fat tails / skew.

Mostra explicitamente que o pipeline **não assume retornos normais** e
quantifica o risco de cauda.

### Aba de destino
Aba **Benchmark**, seção nova "Return Distribution" após métricas.

### Dados necessários
- `benchmark_equity.parquet` → derivar retornos diários do portfólio.
- `scipy.stats` para normal theoretical quantiles (já é dependência).

### Sketch de implementação

```python
def _chart_return_distribution(equity_df: pl.DataFrame) -> go.Figure:
    """2-column subplot: histogram + QQ."""
    from plotly.subplots import make_subplots
    import scipy.stats as stats

    rets = _to_daily_returns(equity_df)
    mu, sigma = rets.mean(), rets.std()
    var5 = np.quantile(rets, 0.05)
    cvar5 = rets[rets <= var5].mean()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Return Distribution", "Q-Q vs Normal"))

    # Col 1: histogram + normal PDF + VaR shading
    fig.add_trace(go.Histogram(x=rets, histnorm="probability density",
                               name="Empirical"), row=1, col=1)
    xs = np.linspace(rets.min(), rets.max(), 200)
    fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, mu, sigma),
                             mode="lines", name="Normal fit"), row=1, col=1)
    # Shade tail < var5
    fig.add_vrect(x0=rets.min(), x1=var5, fillcolor="red", opacity=0.2,
                  row=1, col=1)

    # Col 2: QQ
    theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(rets)))
    empirical = np.sort(rets)
    fig.add_trace(go.Scatter(x=theoretical, y=empirical, mode="markers",
                             name="Empirical"), row=1, col=2)
    # 45-degree line
    lo, hi = theoretical.min(), theoretical.max()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo*sigma+mu, hi*sigma+mu],
                             mode="lines", name="Normal line"), row=1, col=2)
    return fig
```

### Cuidados
- **Skew/kurtosis anotados:** mostrar `Skew: X.XX | Excess Kurt: X.XX` no
  título (usar kurtosis **de excesso** / Fisher, que é 0 para a normal —
  evita o valor 3 de base que confunde o leitor).
- **VaR/CVaR rotulados como diários:** o eixo X está em retorno diário,
  então o título e as vlines devem dizer "Daily VaR 5%" / "Daily CVaR 5%"
  explicitamente.
- **QQ empírico vs teórico:** não confundir ordem dos eixos. Usar
  plotting positions Blom-like `(i-0.5)/n` em vez de `linspace(0.01, 0.99)`.

### Critério de aceite
- Histograma + fit normal visível.
- Cauda < VaR sombreada em vermelho claro.
- QQ com pontos fugindo da diagonal nas pontas (prova visual de fat tail).
- Anotação textual: `Skew: -0.58 | Excess Kurt: +10.66 | Daily VaR 5%: -1.02% | Daily CVaR 5%: -1.71%`
  (valores do champion 10y OOS como referência).

### Esforço
M (3-4h). Subplot + scipy + anotações. Cuidado com NaN de primeiro dia.

---

## Fase 4 — Rolling Beta, Alpha e Correlation vs SPY

### Objetivo
Três séries temporais em painel único mostrando como a relação
portfolio vs SPY evolui. Essencial para argumentar "meu alpha é real,
não beta leverage".

### Aba de destino
Aba **Benchmark**, seção nova "Market Relationship" após rolling Sharpe.

### Dados necessários
- `benchmark_equity.parquet` (`equity`, `spy_equity`) — já existe.

### Sketch de implementação

```python
def _rolling_regression(
    port_ret: np.ndarray,
    spy_ret: np.ndarray,
    window: int = 126,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (beta, alpha_annual, correlation) rolling series."""
    # For each window: OLS port ~ alpha + beta*spy.
    # Alpha annualized = alpha_daily * 252.
    # Correlation = np.corrcoef.

def _chart_rolling_market_relationship(equity_df, window: int) -> go.Figure:
    """3 subplots stacked: beta, alpha, correlation."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Rolling Beta", "Rolling Alpha (ann.)",
                                        "Rolling Correlation"))
    # Horizontal reference line at beta=1, alpha=0, corr=0.
```

### Cuidados
- **Janela padrão 126 dias (6 meses)** — compatível com rolling Sharpe.
- **Alpha anualizado** não diário. Multiplicar por 252.
- **Beta > 1 não é ruim por si só** — só precisa ser acompanhado de
  alpha positivo. Anotar média ao lado do título.

### Critério de aceite
- 3 séries alinhadas no mesmo eixo X.
- Linhas de referência (beta=1, alpha=0, corr=0).
- Slider de janela (60-252), reusar componente da fase Rolling Sharpe.
- Hover mostra os 3 valores na mesma data.

### Esforço
S (2-3h). OLS rolante trivial com numpy; subplot stack.

---

## Fase 5 — Up-Capture / Down-Capture Chart

### Objetivo
4 barras agrupadas:
- Up-Capture: portfolio return médio em meses up-SPY ÷ SPY return médio
  em meses up
- Down-Capture: idem para meses down
- Up vs Down ratio
- Positive month % (portfolio vs SPY)

Gráfico institucional clássico de pitch deck.

### Aba de destino
Aba **Benchmark**, ao lado da seção "Performance Metrics".

### Dados necessários
- `benchmark_equity.parquet` — já existe.

### Sketch de implementação

```python
def _compute_capture_ratios(equity_df: pl.DataFrame) -> dict[str, float]:
    """Returns up_capture, down_capture, up_pct, dn_pct."""
    monthly = _to_monthly_returns(equity_df)  # from Fase 1 helper
    up_mask = monthly["spy_ret"] > 0
    up_capture = (monthly.filter(up_mask)["port_ret"].mean() /
                  monthly.filter(up_mask)["spy_ret"].mean())
    down_capture = (monthly.filter(~up_mask)["port_ret"].mean() /
                    monthly.filter(~up_mask)["spy_ret"].mean())
    return {...}

def _chart_capture_ratios(equity_df: pl.DataFrame) -> go.Figure:
    """Grouped bar: 4 metrics × 2 series (portfolio, SPY=baseline=100%)."""
```

### Cuidados
- **Expressar em %:** up_capture = 85% é mais legível que 0.85.
- **Baseline 100%** para visualizar "melhor ou pior que SPY".
- **Interpretação no subtítulo:** "Up-capture > 100%: portfolio captures
  more upside than SPY. Down-capture < 100%: captures less downside."

### Critério de aceite
- 4 barras anotadas com o valor no topo.
- Linha horizontal em 100% como referência.
- Subtítulo explicando.

### Esforço
S (1-2h). Só matemática + bar chart.

---

## Fase 6 — CAPM Scatter com Regressão

### Objetivo
Scatter: cada ponto é um dia, eixo X = retorno SPY, eixo Y = retorno
portfolio. Linha de regressão com slope (beta) e intercept (alpha
anualizado) anotados. Visualização direta da equação `R_p = α + β·R_m`.

### Aba de destino
Aba **Benchmark**, junto com o rolling beta/alpha da Fase 4 (fazem par).

### Dados necessários
- `benchmark_equity.parquet` — já existe.

### Sketch de implementação

```python
def _chart_capm_scatter(equity_df: pl.DataFrame) -> go.Figure:
    port_ret = _to_daily_returns(equity_df["equity"])
    spy_ret  = _to_daily_returns(equity_df["spy_equity"])

    slope, intercept = np.polyfit(spy_ret, port_ret, 1)
    r_squared = np.corrcoef(spy_ret, port_ret)[0, 1] ** 2

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spy_ret, y=port_ret, mode="markers",
                             marker=dict(size=4, opacity=0.5)))
    xs = np.linspace(spy_ret.min(), spy_ret.max(), 100)
    fig.add_trace(go.Scatter(x=xs, y=slope*xs + intercept, mode="lines",
                             name=f"β={slope:.2f}, α_ann={intercept*252*100:.2f}%"))
    fig.add_annotation(text=f"R² = {r_squared:.3f}", ...)
    return fig
```

### Cuidados
- **Densidade visual:** com 2500 dias, usar opacity baixa (0.3-0.5) ou
  2D histogram density.
- **Outliers:** se COVID (2020-03) estiver no set, vai puxar a regressão.
  Considerar mostrar regressão com e sem outliers como toggle.

### Critério de aceite
- Scatter denso + linha de regressão clara.
- Anotação com β, α (anualizado), R².
- Eixos em % (multiplicar por 100).

### Esforço
S (1-2h).

---

## Fase 7 — Turnover e Custo de Transação Acumulado

### Objetivo
Dois subplots:
1. Turnover por rebalance (|Δw| somado sobre tickers).
2. Custo de transação acumulado em $ e diferença gross vs net equity.

Justifica numericamente por que os 15bps importam e por que
`min_rebalance_delta=0.02` está certo.

### Aba de destino
Aba **Benchmark**, seção nova "Trading Costs" após weight heatmap.

### Dados necessários
- `benchmark_weights.parquet` — já existe (→ turnover).
- `benchmark_equity.parquet` — já existe (equity pós-custo).
- **Novo:** `gross_equity` (sem custos) — ver "Etapa 0" abaixo.

### Etapa 0 — persistência do gross
Editar `src/backtest/walk_forward.py` para além de `equity` também gravar
`equity_gross` (sem deduzir custos) em `benchmark_equity.parquet`.
Testar que `equity_gross >= equity` sempre.

### Sketch de implementação

```python
def _compute_turnover(weights_df: pl.DataFrame) -> pl.DataFrame:
    """Return (date, turnover) where turnover = sum |w_t - w_{t-1}|."""
    # Shift, subtract, abs, sum across ticker columns.

def _chart_turnover_and_costs(
    weights_df: pl.DataFrame,
    equity_df: pl.DataFrame,  # with gross + net
) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Turnover per Rebalance",
                                        "Gross vs Net Equity"))
    # Row 1: bar chart of turnover
    # Row 2: two lines (gross, net) + filled area between them
```

### Cuidados
- **Turnover em %:** `0.35 = 35%` é mais legível.
- **Escala dual:** equity em index (base 100) no row 2.
- **Gap gross-net** como annotation "Total cost drag: X.X%".

### Critério de aceite
- Bar chart de turnover por data de rebalance.
- Gross vs net sobrepostos com área preenchida.
- Texto final: "Total transaction cost drag over 10y: +XX bps/year".

### Esforço
M (3-4h). Principalmente Etapa 0 (adicionar gross ao walk-forward).

---

## Fase 8 — Effective N (1/HHI) e Gini no Tempo

### Objetivo
Duas séries sobrepostas mostrando a **diversificação efetiva** do
portfólio ao longo do tempo:

- **Effective N = 1 / Σ w²** — "número equivalente de posições iguais"
- **Gini coefficient** — inequality dos pesos (0 = perfeita igualdade,
  1 = totalmente concentrado)

Hoje o dashboard mostra só a foto atual (concentração no tab Performance).

### Aba de destino
Aba **Benchmark**, seção nova "Portfolio Concentration Over Time" após
o weight heatmap.

### Dados necessários
- `benchmark_weights.parquet` — já existe.

### Sketch de implementação

```python
def _compute_effective_n(weights_row: np.ndarray) -> float:
    """1 / sum(w_i^2). Returns n for uniform, 1 for single-asset."""
    w = weights_row[weights_row > 0]  # long-only, ignore cash/zero
    return 1.0 / np.sum(w ** 2)

def _compute_gini(weights_row: np.ndarray) -> float:
    """Standard Gini. 0 = equal, 1 = fully concentrated."""
    w = np.sort(weights_row[weights_row > 0])
    n = len(w)
    cum = np.cumsum(w)
    return (2 * np.sum((np.arange(1, n+1)) * w) - (n+1) * cum[-1]) / (n * cum[-1])

def _chart_concentration_evolution(weights_df) -> go.Figure:
    """Two-axis: left=Effective N, right=Gini."""
```

### Cuidados
- **Eixo dual:** Effective N vai de ~1 a 52 (universo), Gini de 0 a 1.
- **Referência:** linha horizontal em Effective N = 16.67 (valor que
  corresponde a `max_weight=0.06` uniforme).
- **Cash:** quando o portfólio está 50% cash, effective N deve
  refletir isso (contar cash como 1 ativo? Decidir e documentar).

### Critério de aceite
- Duas séries temporais sobrepostas (escala dual).
- Linha de referência "Max diversified (HRP constraint)" em Effective N.
- Tooltip mostra os dois valores na mesma data.

### Esforço
S (2h).

---

## Fase 9 — Contribuição por Ticker ao Retorno Total (Waterfall)

### Objetivo
Waterfall bar chart: cada ticker contribui X pontos percentuais ao
retorno total. Ordenado do maior contribuidor positivo ao maior
detrator. Mostra quem de fato gerou alpha.

### Aba de destino
Aba **Benchmark**, seção nova "Return Attribution" no final.

### Dados necessários
- `benchmark_weights.parquet` — já existe.
- **Novo ou derivado:** retornos diários de cada ticker do universo
  (atualmente apenas os pesos estão persistidos). Ver Etapa 0.

### Etapa 0 — persistência dos ticker returns
Opções:

1. **Recalcular on-demand no dashboard** a partir dos OHLCV do
   PostgreSQL (melhor para desenvolvimento, pior para deploy).
2. **Persistir `ticker_returns.parquet`** no run do benchmark
   (`src/backtest/walk_forward.py`). Formato: `date × ticker → daily_ret`.

Recomendado: **opção 2**. Arquivo ~1-5 MB e remove dependência de DB do
dashboard.

### Sketch de implementação

```python
def _compute_contribution_per_ticker(
    weights_df: pl.DataFrame,
    returns_df: pl.DataFrame,
) -> dict[str, float]:
    """contribution_i = sum_t w_{i,t-1} * r_{i,t} (multiplicative or additive)."""
    # Align weights and returns; lag weights by 1 day.
    # Sum per ticker.

def _chart_contribution_waterfall(contributions: dict) -> go.Figure:
    """Waterfall: base + each ticker's contribution + total."""
    sorted_items = sorted(contributions.items(), key=lambda x: -x[1])
    fig = go.Figure(go.Waterfall(
        x=["Start"] + [t for t, _ in sorted_items] + ["Total"],
        measure=["absolute"] + ["relative"]*len(sorted_items) + ["total"],
        y=[0] + [c for _, c in sorted_items] + [None],
    ))
```

### Cuidados
- **Soma deve bater com o retorno total** (dentro de tolerância de
  0.5% devido a custos e rebalance).
- **Unidade:** percentage points (não retorno %). Fazer a anotação.
- **Muitos zeros:** tickers nunca selecionados vão ter contribuição 0.
  Filtrar ou agrupar em "Others".

### Critério de aceite
- Waterfall com top 15 + "Others" + Total.
- Cores: verde para positivo, vermelho para negativo.
- Soma final bate com CAGR * n_years dentro de 1pp.

### Esforço
M (3-5h). Etapa 0 é o trabalho principal.

---

---

# TIER 2 — Diferenciadores Únicos

---

## Fase 10 — CPCV Path Spaghetti Chart

### Objetivo
Plotar as **15 equity curves dos paths CPCV-OOS** sobrepostas, com
mediana e banda IQR (25-75%) sombreadas. É *o* gráfico que separa
"backtest sério" de "backtest overfit" — prova visual de que o Sharpe
não veio de um path de sorte.

### Aba de destino
Aba **Benchmark**, seção nova "CPCV-OOS Path Distribution" após o
expander de validation.

### Dados necessários
Atualmente **não existe** — o código CPCV-OOS só agrega Sharpe final.

### Etapa 0 — persistência dos paths
Editar `src/backtest/cpcv_oos.py` para, ao final de cada path, gravar:

```python
# data/outputs/cpcv_paths.parquet
# columns: config_name, path_id (0-14), date, equity
```

- Tamanho esperado: ~15 paths × ~2500 dias × 1 config = 37500 rows ~= 2 MB.
- Se salvar todos os 547 configs: inviável. Salvar só o champion.

### Sketch de implementação

```python
def _chart_cpcv_spaghetti(paths_df: pl.DataFrame) -> go.Figure:
    """15 thin lines + median + IQR band."""
    fig = go.Figure()
    # 15 individual paths (low alpha, thin)
    for pid in range(15):
        path = paths_df.filter(pl.col("path_id") == pid)
        fig.add_trace(go.Scatter(x=path["date"], y=path["equity"],
                                 mode="lines", line=dict(width=1),
                                 opacity=0.3, showlegend=False))
    # Aggregate median + IQR
    agg = paths_df.group_by("date").agg(
        pl.col("equity").quantile(0.25).alias("q25"),
        pl.col("equity").quantile(0.50).alias("med"),
        pl.col("equity").quantile(0.75).alias("q75"),
    )
    # Shaded IQR
    fig.add_trace(go.Scatter(x=agg["date"], y=agg["q75"], ...))
    fig.add_trace(go.Scatter(x=agg["date"], y=agg["q25"], fill="tonexty", ...))
    # Bold median
    fig.add_trace(go.Scatter(x=agg["date"], y=agg["med"],
                             mode="lines", line=dict(width=3)))
```

### Cuidados
- **Alinhamento temporal:** cada path cobre folds diferentes. Usar
  base=1.0 em data comum de início para visualização justa.
- **Não usar CPCV com retrain pesado:** usar a config champion
  (`NaiveModelFactory lookback=5`) pelo custo computacional.
- **Título explicativo:** "Each line = one of 15 combinatorial paths.
  Tight cluster = low path-dependency (good)."

### Critério de aceite
- 15 linhas finas + banda IQR + mediana grossa.
- Anotação: "Sharpe range across paths: X.XX — X.XX".
- Tooltip mostra path_id e equity.

### Esforço
M (4-6h). Etapa 0 é metade do esforço.

---

## Fase 11 — Violin / Box Plot de Sharpe por Path CPCV

### Objetivo
Complementa a Fase 10. Violin plot mostrando a distribuição dos 15
Sharpes individuais, com:

- Mediana e IQR
- Linha horizontal do **Deflated Sharpe threshold**
- Linha horizontal do **Sharpe realizado no walk-forward OOS**
- Anotação de **Probabilistic Sharpe Ratio (PSR)**

Validação estatística explícita.

### Aba de destino
Aba **Benchmark**, logo depois da Fase 10 (formam par).

### Dados necessários
- `cpcv_paths.parquet` da Fase 10 (derivar Sharpe por path_id).
- `benchmark_metrics.json` (Sharpe OOS walk-forward) — já existe.

### Sketch de implementação

```python
def _chart_sharpe_violin(paths_df, oos_sharpe: float,
                         dsr_threshold: float) -> go.Figure:
    """Violin of per-path Sharpes with reference lines."""
    sharpes = _compute_sharpe_per_path(paths_df)  # list of 15 floats
    fig = go.Figure()
    fig.add_trace(go.Violin(y=sharpes, box_visible=True, meanline_visible=True,
                            points="all", name="CPCV paths"))
    fig.add_hline(y=dsr_threshold, line_dash="dash",
                  annotation_text=f"DSR threshold = {dsr_threshold:.2f}")
    fig.add_hline(y=oos_sharpe, line_dash="solid", line_color="gold",
                  annotation_text=f"Walk-forward OOS = {oos_sharpe:.2f}")
```

### Cuidados
- **DSR threshold** é específico para `n_trials`. Buscar em
  `validation_results.json` ou hardcodar o cálculo.
- **Sharpe OOS vs mediana CPCV:** se muito diferente, é sinal de regime
  change — anotar.

### Critério de aceite
- Violin com 15 pontos.
- 2 hlines referenciais com anotação.
- Estatísticas à direita: mean, std, % positivos.

### Esforço
S (1-2h) após Fase 10 pronta.

---

## Fase 12 — Agent Vote Matrix Heatmap

### Objetivo
Heatmap `ticker × rebalance_date` onde cada célula é colorida pela
**ação final do PM** (BUY=verde, HOLD=amarelo, SELL=vermelho). Deixa
explícito o "cérebro" do sistema e mostra a consistência/instabilidade
das decisões.

### Aba de destino
Aba **War Room**, seção nova "Decision History" no topo ou como
pseudo-tab interna.

### Dados necessários
Atualmente **não existe** — apenas o último `decisions.json` é
persistido.

### Etapa 0 — persistência do histórico
Editar `src/agents/pipeline.py` (ou o orquestrador `src/agents/decide.py`
se existir) para, a cada run, adicionar linha em:

```python
# data/outputs/decisions_history.parquet
# columns: run_date, ticker, action, confidence, weight_final
```

Append-only. Se já existe no caminho legado `decisions.json` arquivado,
considerar migração one-time.

### Sketch de implementação

```python
def _chart_agent_vote_matrix(history_df: pl.DataFrame) -> go.Figure:
    """Heatmap: action encoded as {BUY: 1, HOLD: 0, SELL: -1}."""
    pivot = history_df.pivot(index="ticker", on="run_date", values="action_code")
    # RdYlGn colorscale
    fig = go.Figure(data=go.Heatmap(
        z=pivot.to_numpy(),
        x=pivot.columns,
        y=pivot["ticker"],
        colorscale="RdYlGn",
        zmin=-1, zmid=0, zmax=1,
    ))
```

### Cuidados
- **N runs pode ser pequeno** no começo. Rodar pipeline semanalmente
  para construir histórico; ou simular.
- **Ordem de tickers:** agrupar por setor (precisa mapping) ou por
  frequência de BUY.
- **Tooltip:** mostrar também `confidence` e `weight`.

### Critério de aceite
- Heatmap ticker × data com 3 cores.
- Toggle para filtrar "BUY only" / "SELL only" (bônus).
- Legenda clara dos códigos.

### Esforço
M (4-6h). Etapa 0 depende de arquitetura do pipeline.

---

## Fase 13 — Inter-Agent Agreement / Disagreement Rate

### Objetivo
Visualizar com que frequência os 4 agentes LangGraph concordam. Duas
views possíveis:

- **Stacked bar:** em cada ticker/run, quantos agentes votaram BUY vs
  SELL vs HOLD.
- **Sankey diagram:** fluxo Analyst Technical → PM, Fundamentalist → PM,
  Bear → PM, destacando onde PM diverge do consenso.

Mostra que o debate agrega valor (senão PM sempre concordaria com
maioria).

### Aba de destino
Aba **War Room**, após agent vote matrix.

### Dados necessários
- `debate_history.json` — já existe, mas apenas para último run. Estender
  para histórico (depende de Fase 12 Etapa 0, expandir schema para
  incluir voto de cada agente).

### Etapa 0 — expandir `decisions_history.parquet`
```
# columns: run_date, ticker, technical_view, fundamental_view,
#          bear_view, pm_action, pm_confidence
```

Views dos agentes individuais são derivadas dos reports (precisa parser
do campo `recommendation` de cada report).

### Sketch de implementação

```python
def _chart_agent_agreement_rate(history_df: pl.DataFrame) -> go.Figure:
    """Bar: % de vezes que N/4 agentes concordam (0-4)."""
    # For each (run_date, ticker), count how many agent votes == pm_action.
    # Plot histogram of agreement counts.

def _chart_pm_override_rate(history_df: pl.DataFrame) -> go.Figure:
    """When majority says BUY but PM says HOLD/SELL, count overrides."""
```

### Cuidados
- **Reports podem ter recomendações em formato livre.** Pydantic
  structured output do LangGraph deve garantir formato, mas validar.
- **Confidence threshold MIN_CONFIDENCE_FOR_ACTION=0.3** força HOLD
  mesmo com maioria BUY — isso **deve** aparecer como override.

### Critério de aceite
- Bar chart de agreement distribution.
- Tabela de "top PM overrides" com data/ticker/razão.

### Esforço
M (3-5h).

---

## Fase 14 — RAG Source Frequency

### Objetivo
Top-N fontes mais citadas pelo agente Fundamentalist nos debates, com
latência P95 ao lado. Mostra que a camada RAG é real, usada e
performática. Complementa os números `172 artigos, P95=101ms` já no
CLAUDE.md.

### Aba de destino
Aba **War Room**, ou aba nova "RAG Analytics" se for substancial.

### Dados necessários
Atualmente **não existe** — retrievals não são logados com metadata.

### Etapa 0 — logging RAG
Editar `src/agents/rag.py` (ou equivalente) para emitir log estruturado
em cada retrieval:

```python
# data/outputs/rag_retrieval_log.jsonl
{"timestamp": "...", "ticker": "AAPL", "query": "...",
 "sources": [{"url": "...", "score": 0.82}, ...],
 "latency_ms": 94}
```

### Sketch de implementação

```python
def _chart_rag_top_sources(log_path: Path, top_n: int = 20) -> go.Figure:
    """Horizontal bar: source hostname vs citation count."""
    # Parse JSONL, count by hostname, sort desc, top N.

def _chart_rag_latency_distribution(log_path: Path) -> go.Figure:
    """Histogram of latency_ms with P50, P95, P99 annotated."""
```

### Cuidados
- **Hostname agregation:** `cnbc.com/article/123` e `cnbc.com/article/456`
  devem agregar como `cnbc.com`.
- **Dedupe:** mesma fonte retrieved 2× no mesmo run conta 1× ou 2×?
  Decidir. Sugestão: contar citações (cada aparição em report).

### Critério de aceite
- Top 20 sources como horizontal bar.
- Card com P50/P95/P99 latency.
- Total retrievals e total unique sources.

### Esforço
M (4-5h). Etapa 0 é o grosso.

---

## Fase 15 — PatchTST Calibration Plot

### Objetivo
Plot de calibração: no eixo X, `prob_up` predito pelo PatchTST agrupado
em bins (0.0-0.1, 0.1-0.2, …, 0.9-1.0); no eixo Y, frequência real de
eventos up (i.e., retorno dos próximos N dias > 0). Linha 45° = modelo
perfeitamente calibrado.

Essa é *a* validação que separa modelo brinquedo de modelo sério.

### Aba de destino
Aba **Microstructure**, seção nova "Model Calibration" após fan chart.

### Dados necessários
Atualmente **apenas parcial** — `predictions.json` é snapshot. Precisa
histórico.

### Etapa 0 — histórico de predições + outcomes
Editar `src/models/patchtst_predict.py` (ou equivalente) para persistir:

```python
# data/outputs/patchtst_predictions_history.parquet
# columns: prediction_date, ticker, horizon_days, prob_up, quantile_10, ..., quantile_90
```

Depois, no plot, **joinar com OHLCV** do PostgreSQL para calcular
`realized_up = close[t+h] > close[t]`.

### Sketch de implementação

```python
def _compute_calibration(
    predictions: pl.DataFrame,
    ohlcv: pl.DataFrame,
    n_bins: int = 10,
) -> pl.DataFrame:
    """Join predictions with realized outcomes, group by prob_up bin."""
    # For each (ticker, prediction_date, horizon):
    # realized_up = 1 if close[d+h] > close[d] else 0
    # Bin prob_up, compute mean realized per bin.

def _chart_calibration(calibration_df) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=calibration_df["prob_bin_center"],
                             y=calibration_df["realized_freq"],
                             mode="markers+lines", name="PatchTST"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             line=dict(dash="dash"), name="Perfect"))
    # Size of markers proportional to count in bin
```

### Cuidados
- **Horizon:** calibrar para o horizonte de predição usado em produção
  (geralmente 5-10 dias). Fazer dropdown se houver múltiplos horizontes.
- **Dados insuficientes:** com poucos runs, bins vão estar vazios.
  Filtrar bins com n<10.
- **Calibration metric:** anotar Brier score ou ECE (Expected Calibration
  Error).

### Critério de aceite
- Scatter com linha 45° de referência.
- Brier score anotado.
- Tooltip mostra `bin, n_obs, predicted_mean, realized_freq`.

### Esforço
M (5-7h). Etapa 0 é significativa — reprocessar predições históricas.

---

## Fase 16 — Fan Chart Coverage Test

### Objetivo
Dado que o PatchTST emite quantis (q10, q25, q50, q75, q90), nos últimos
N rebalances, quantas vezes o retorno realizado ficou dentro do
intervalo q10-q90? Se bem calibrado, deve ser ~80%.

Primo do calibration plot (Fase 15), mas para quantis inteiros (não só
prob_up).

### Aba de destino
Aba **Microstructure**, logo depois do calibration plot.

### Dados necessários
Mesmos da Fase 15 (histórico de predições PatchTST).

### Sketch de implementação

```python
def _compute_coverage(
    predictions: pl.DataFrame,
    ohlcv: pl.DataFrame,
) -> dict[str, float]:
    """% of realized returns within each quantile band."""
    # Realized return = (close[t+h] - close[t]) / close[t]
    # Bands: q10-q90 (expected 80%), q25-q75 (expected 50%).

def _chart_coverage_bar(coverage: dict) -> go.Figure:
    """Grouped bar: expected vs realized coverage for each band."""
```

### Cuidados
- **Log vs simple return:** PatchTST emite predições no espaço do target
  escolhido. Verificar se quantis são de log-return ou simple return.
- **CRPS (Continuous Ranked Probability Score)** como métrica
  complementar seria bônus.

### Critério de aceite
- 4 barras (q5-q95, q10-q90, q25-q75, q40-q60) com expected vs realized.
- Anotação CRPS ou Pinball Loss.

### Esforço
S (2-3h) após Fase 15 pronta.

---

## Fase 17 — Decision Flow Sankey

### Objetivo
Diagrama Sankey com 4 colunas:

1. **Universo** (52 tickers)
2. **Debate outcome** (BUY / HOLD / SELL agregado)
3. **HRP output** (weights brutos)
4. **Decision final** (após tier adjust + cash)

Flow thickness = fração do capital. Storytelling em uma imagem.

### Aba de destino
Aba **Performance**, topo (overview do pipeline).

### Dados necessários
- `decisions.json` — já existe.
- Metadata da decisão (hrp_raw_weights, action_counts, cash_fraction) —
  já presentes parcialmente.

### Sketch de implementação

```python
def _chart_decision_flow_sankey(decisions: dict) -> go.Figure:
    """4-column sankey."""
    # Nodes: Universe, BUY, HOLD, SELL, HRP_nonzero, HRP_zero, Final_invested, Cash
    # Links with weight = fraction of capital or count.
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=[...], color=[...]),
        link=dict(source=[...], target=[...], value=[...]),
    )])
```

### Cuidados
- **Unit confusion:** uma camada é contagem (n tickers), outra é capital
  (%). Decidir qual unificar. Sugestão: tudo em % de capital.
- **SELL sempre termina em cash** no 3-tier model — deixar isso explícito
  na cor.

### Critério de aceite
- Sankey com 4-5 colunas.
- Cores semânticas (verde=BUY path, vermelho=SELL path).
- Totais batem com `decisions.json` metadata.

### Esforço
M (3-4h).

---

## Fase 18 — Leverage e Realized vs Target Vol

### Objetivo
Três séries temporais empilhadas:

1. **Leverage** aplicado pelo vol targeting (clamp [0.5, 1.0])
2. **Realized volatility** (janela 63d) do portfólio
3. **Target volatility** (10% ann. — linha horizontal)

Prova que o risk management está ativo e quando exatamente o clamp
saturou.

### Aba de destino
Aba **Benchmark**, seção nova "Risk Management" antes de "Trading Costs"
(Fase 7).

### Dados necessários
Atualmente **não persistido** — leverage é calculado mas não gravado.

### Etapa 0 — persistência
Editar `src/backtest/walk_forward.py` para adicionar colunas em
`benchmark_equity.parquet`:

```
# date, equity, spy_equity, leverage, realized_vol_63d
```

Ou arquivo separado `benchmark_risk_state.parquet` se preferir não
poluir o schema principal.

### Sketch de implementação

```python
def _chart_vol_targeting(equity_df: pl.DataFrame, target_vol: float) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Leverage", "Realized vs Target Vol"))
    fig.add_trace(go.Scatter(x=equity_df["date"], y=equity_df["leverage"]),
                  row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dash", row=1, col=1,
                  annotation_text="min leverage")
    fig.add_hline(y=1.0, line_dash="dash", row=1, col=1,
                  annotation_text="max leverage")
    fig.add_trace(go.Scatter(x=..., y=equity_df["realized_vol_63d"],
                             name="Realized"), row=2, col=1)
    fig.add_hline(y=target_vol, line_dash="solid", line_color="gold",
                  row=2, col=1, annotation_text=f"Target = {target_vol:.0%}")
```

### Cuidados
- **Realized vol:** rolling std anualizado (× √252).
- **Leverage quantization:** se `vol_lookback=63`, os primeiros 63 dias
  têm leverage `min_leverage` (cold start). Anotar warmup period.
- **Saturation markers:** shade áreas onde leverage == 0.5 ou == 1.0.

### Critério de aceite
- 2 subplots alinhados temporalmente.
- 3 hlines de referência (min_leverage, max_leverage, target_vol).
- Tooltip combinado.
- Anotação: "% time at max leverage: X% | % time at min: Y%".

### Esforço
M (3-4h). Etapa 0 é metade do esforço.

---

---

## 4. Apêndice — Ideias Descartadas

Descartadas (não entram em nenhuma fase) e por quê:

- **Candlestick OHLCV por ticker:** já disponível em qualquer broker.
  Baixo valor agregado ao dashboard.
- **Sector exposure stacked area:** exigiria mapping ticker→setor que
  hoje não existe no projeto. Valor marginal; retomar depois.
- **Feature importance heatmap do PatchTST:** PatchTST é fim-a-fim
  sem features engineered; não há importância interpretável.
- **Sharpe gauge / PSR gauge visual:** informação já está como texto em
  múltiplos lugares; gauges são ruim-UX (consomem espaço, pouca info).
- **Heatmap de correlação 52×52 dos tickers:** ruim de ler em qualquer
  size, e o universo é pequeno demais para "fator clustering" ser
  interessante.
- **Performance attribution tipo Brinson:** requer benchmark por setor
  + sector weights históricos. Overkill para portfolio pessoal.

---

## 5. Checklist de Execução por Fase

Ao implementar cada fase, seguir:

1. [ ] Ler o documento de referência da visualização
       (quantstats / ffn / pyfolio fazem muitos destes).
2. [ ] Se Etapa 0 (persistência de dado novo): implementar e rodar
       `make benchmark-fast` ou `make decide` para gerar arquivo.
3. [ ] Criar helper `_chart_*` em `src/dashboard/app.py` ou (se >100
       linhas) em `src/dashboard/charts/<topic>.py`.
4. [ ] Integrar na aba correspondente.
5. [ ] Escrever teste em `tests/unit/dashboard/test_phase_N_*.py`.
6. [ ] Rodar `poetry run pytest tests/unit/dashboard -q`.
7. [ ] Rodar `poetry run streamlit run src/dashboard/app.py` e
       inspecionar visualmente.
8. [ ] Capturar screenshot, adicionar ao commit.
9. [ ] Commit seguindo convenção: `feat: dashboard phase N — <chart>`.
10. [ ] Atualizar este arquivo marcando a fase como concluída.

---

## 6. Status de Implementação

| Fase | Status | Commit | Screenshot |
|---|---|---|---|
| 1 — Calendar heatmap | ✅ | `b5e55c4` (bundled) | — |
| 2 — Top-10 drawdowns | ✅ | `b5e55c4` (bundled) | — |
| 3 — Distribution + QQ | ✅ | `b5e55c4` (bundled) | — |
| 4 — Rolling beta/alpha | ✅ | `b5e55c4` (bundled) | — |
| 5 — Up/Down capture | ✅ | `b5e55c4` (bundled) | — |
| 6 — CAPM scatter | ✅ | `b5e55c4` (bundled) | — |
| 7 — Turnover + costs | ✅ | `b5e55c4` (bundled) | — |
| 8 — Effective N / Gini | ✅ | `b5e55c4` (bundled) | — |
| 9 — Contribution waterfall | ✅ | `b5e55c4` (bundled) | — |
| 10 — CPCV spaghetti | ✅ | `b5e55c4` (bundled) | — |
| 11 — Sharpe violin | ✅ | `b5e55c4` (bundled) | — |
| 12 — Agent vote matrix | ⬜ | — | — |
| 13 — Agent agreement | ⬜ | — | — |
| 14 — RAG source frequency | ⬜ | — | — |
| 15 — PatchTST calibration | ⬜ | — | — |
| 16 — Fan chart coverage | ⬜ | — | — |
| 17 — Decision flow sankey | ⬜ | — | — |
| 18 — Vol targeting trajectory | ⬜ | — | — |

---

*Documento criado em 2026-04-23. Atualizar à medida que fases forem
concluídas.*
