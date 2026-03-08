# Benchmark Implementation Plan — US Market (S&P 500)

> Plano detalhado para implementar o benchmark walk-forward com 50 ativos US,
> rebalanceamento semanal e comparacao contra SPY buy-and-hold.

---

## Resumo Executivo

O sistema atual gera um **snapshot unico** de decisoes. Para validar a
estrategia, precisamos de um **motor de simulacao temporal** (walk-forward)
que conecte todas as pecas existentes (PatchTST, HRP, custos) e produza
uma equity curve comparavel ao S&P 500.

**Escopo:** apenas ativos US. Sem BR. Benchmark = SPY.
**Rebalanceamento:** semanal (~5 dias uteis).
**Retreino PatchTST:** a cada 6 meses (~126 dias uteis).
**Debate LLM:** desligado no backtest ($0). Teste isolado em 1 run separado.

---

## Sessoes de Implementacao

### Sessao 21 — Configuracao de Tickers (P1.1)
**Tempo estimado de implementacao:** ~30 min
**Arquivos novos:** `config/tickers.json`, `src/config.py`
**Arquivos alterados:** `src/data/ingestion.py`, `src/portfolio/decision_engine.py`, `src/models/predict.py`
**Testes novos:** `tests/test_config.py`

**O que fazer:**

1. Criar `config/tickers.json` com os 50 tickers US + benchmark:
```json
{
    "market": "US",
    "benchmark": "SPY",
    "tickers": [
        "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO", "CRM", "AMD",
        "JPM", "BAC", "GS", "MS", "WFC", "BLK", "AXP", "C",
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO",
        "PG", "KO", "PEP", "COST", "WMT", "HD", "MCD",
        "XOM", "CVX", "COP", "SLB",
        "CAT", "HON", "UPS", "BA", "GE", "RTX",
        "NEE", "DUK", "AMT", "PLD",
        "DIS", "NFLX", "CMCSA",
        "LIN", "APD", "NEM"
    ],
    "trading_days_per_year": 252,
    "risk_free_rate": 0.05
}
```

2. Criar `src/config.py`:
   - `load_ticker_config(path) -> dict` — carrega JSON, retorna dict completo
   - `load_tickers(path) -> list[str]` — retorna lista de tickers (sem benchmark)
   - `load_benchmark(path) -> str` — retorna ticker do benchmark
   - Path default: `config/tickers.json` relativo a project root

3. Atualizar `DEFAULT_TICKERS` em `ingestion.py` e `decision_engine.py` para
   usar `src.config.load_tickers()` com fallback para os 4 tickers originais
   (backward compat — se config nao existir, usa hardcoded).

4. Testes:
   - load com arquivo valido
   - load com arquivo inexistente (fallback)
   - validacao de campos obrigatorios (tickers, benchmark, market)

**Criterio de aceite:** `load_tickers()` retorna 52 tickers; `load_benchmark()` retorna "SPY"; testes existentes continuam passando.

---

### Sessao 22 — HRP max_weight Dinamico (P1.2)
**Tempo estimado de implementacao:** ~15 min
**Arquivos alterados:** `src/portfolio/decision_engine.py`
**Testes alterados:** `tests/test_decision_engine.py`

**O que fazer:**

1. No `DecisionEngine.__init__`, calcular `max_weight` dinamicamente:
```python
n = len(self.tickers)
dynamic_max_weight = min(0.25, 2.0 / n)  # max 2x equal weight
```
   - Para 4 tickers: `min(0.25, 0.5)` = 0.25 (igual ao atual)
   - Para 50 tickers: `min(0.25, 0.04)` = 0.04 (4%)

2. Usar `dynamic_max_weight` ao instanciar `HRPConfig` se nenhum `hrp_config`
   foi passado explicitamente.

3. Testes:
   - 4 tickers: max_weight = 0.25 (backward compat)
   - 50 tickers: max_weight = 0.04
   - hrp_config explicito: nao sobreescreve

**Criterio de aceite:** HRP com 50 tickers usa max_weight=0.04; testes existentes continuam passando.

---

### Sessao 23 — Download Paralelo de Dados (P2.1)
**Tempo estimado de implementacao:** ~1h
**Arquivos alterados:** `src/data/ingestion.py`
**Testes alterados:** `tests/test_ingestion.py`

**O que fazer:**

1. Adicionar metodo `_download_batch()` ao `MarketDataIngester`:
   - `ThreadPoolExecutor(max_workers=5)` para downloads simultaneos
   - Rate limit de 0.5s entre submissoes (yfinance rate limit)
   - Retry por ticker individual (nao re-baixa todos em falha)

2. Adicionar parametro `parallel: bool = True` ao `run()`:
   - `parallel=True`: usa `_download_batch()`
   - `parallel=False`: usa loop sequencial existente (backward compat)

3. Adicionar suporte a `start_date` / `end_date` alem de `years`:
   - Necessario para walk-forward (baixar periodo especifico)
   - Se `start_date` fornecido, ignora `years`

4. Testes:
   - Download paralelo com mocks
   - Falha parcial (1 ticker falha, outros OK)

**Criterio de aceite:** 50 tickers baixados em paralelo; testes existentes passam.

---

### Sessao 24 — Walk-Forward Backtester (P1.5 — PECA CENTRAL)
**Tempo estimado de implementacao:** ~6-8h (sessao mais longa)
**Arquivos novos:** `src/backtest/walk_forward.py`
**Testes novos:** `tests/test_walk_forward.py`

**O que fazer:**

Este e o modulo mais importante de todo o benchmark. Sem ele, nao ha
simulacao temporal.

1. **Dataclasses de configuracao:**

```python
@dataclass(frozen=True)
class WalkForwardConfig:
    retrain_every: int = 126       # retreinar PatchTST a cada ~6 meses
    rebalance_every: int = 5       # rebalancear semanalmente
    lookback_days: int = 504       # janela de treino (~2 anos)
    initial_capital: float = 1_000_000.0
    costs: TransactionCosts | None = None
    min_rebalance_delta: float = 0.0  # threshold minimo para rebalancear (0 = sempre)
    trading_days_per_year: int = 252
    rf: float = 0.05               # risk-free rate anual
```

2. **Dataclasses de resultado:**

```python
@dataclass
class RebalanceRecord:
    date: date
    weights: dict[str, float]     # pesos apos rebalanceo
    turnover: float               # soma |delta_w| neste rebalanceo
    costs: float                  # custo total em valor absoluto
    retrained: bool               # True se PatchTST foi retreinado nesta data

@dataclass
class WalkForwardResult:
    equity_curve: pl.DataFrame      # date, portfolio_value, benchmark_value
    daily_returns: pl.DataFrame     # date, portfolio_return, benchmark_return
    rebalance_history: list[RebalanceRecord]
    metrics: dict[str, float]       # sharpe, alpha, beta, max_dd, cagr, etc.
    config: WalkForwardConfig
    metadata: dict[str, Any]        # n_tickers, period, etc.
```

3. **Classe `WalkForwardBacktester`:**

```python
class WalkForwardBacktester:
    def __init__(self, config: WalkForwardConfig | None = None) -> None: ...

    def run(
        self,
        ohlcv: pl.DataFrame,           # OHLCV completo (todos os tickers, long format)
        tickers: list[str],
        benchmark_ticker: str = "SPY",
    ) -> WalkForwardResult: ...
```

4. **Logica do `run()` — Loop principal:**

```
trading_dates = sorted unique dates from ohlcv (after warmup period)

Para cada trading_date[i]:
    a. data_available = ohlcv filtrado ate trading_date[i] (sem look-ahead)

    b. Se e dia de RETREINO (a cada retrain_every dias):
       - Recortar ultimos lookback_days de dados
       - Treinar PatchTST (via model_factory callback)
       - Salvar modelo em memoria (nao em disco)

    c. Se e dia de REBALANCEAMENTO (a cada rebalance_every dias):
       - Gerar previsoes com modelo atual (predict)
       - Computar retornos para HRP (ultimos lookback_days)
       - Rodar HRP com previsoes como proxy de confidence
       - Aplicar threshold de rebalanceamento (min_rebalance_delta)
       - Calcular custos de transacao (delta de pesos)
       - Registrar RebalanceRecord

    d. Calcular retorno do portfolio neste dia:
       - portfolio_return = sum(weight_i * return_i)  # retorno ponderado
       - portfolio_value *= (1 + portfolio_return - costs)

    e. Registrar equity diaria
```

5. **model_factory — interface desacoplada:**

```python
class ModelFactory(Protocol):
    def train(self, train_df: pl.DataFrame) -> None:
        """Treina modelo nos dados fornecidos."""
        ...

    def predict(self, df: pl.DataFrame) -> dict[str, float]:
        """Retorna {ticker: predicted_return} para cada ticker."""
        ...
```

Implementar `PatchTSTModelFactory` que wrapa `TitaniumForecaster`:
- `train()` — fit do PatchTST
- `predict()` — predict_proba, retorna P(up) por ticker como proxy de confidence

Implementar `NaiveModelFactory` para testes rapidos:
- `train()` — noop
- `predict()` — retorna momentum simples (retorno dos ultimos 5 dias)

6. **Calculo de retornos do portfolio:**
   - Entre rebalanceamentos: pesos sao fixos, retorno = weighted sum dos retornos diarios
   - Nos dias de rebalanceamento: custos aplicados sobre |delta_w|
   - Benchmark: buy-and-hold SPY (sem custos)

7. **Protecoes anti-bias:**
   - NUNCA acessar dados futuros (filtro strict por date <= current_date)
   - Pesos somam 1.0 (fully invested, sem alavancagem)
   - Se algum ticker nao tem dados para a data, manter peso anterior
   - Log de warning se dados insuficientes para algum ticker

8. **Testes (~40-50 testes):**
   - WalkForwardConfig validation (defaults, custom)
   - RebalanceRecord, WalkForwardResult (dataclass behavior)
   - Loop principal com dados sinteticos (3 tickers, 500 dias)
   - Retreino ocorre na frequencia correta
   - Rebalanceamento ocorre na frequencia correta
   - Zero look-ahead bias (ticker com dados futuros artificiais nao afeta resultado)
   - Custos de transacao reduzem equity (com vs sem custos)
   - Benchmark e buy-and-hold puro (sem custos)
   - min_rebalance_delta: se delta < threshold, pesos nao mudam
   - NaiveModelFactory como model_factory
   - Edge cases: 1 ticker, datas com gaps, ticker sem dados em alguma data

**Criterio de aceite:** Walk-forward com NaiveModelFactory roda em dados sinteticos, produz equity curve e metricas. Zero look-ahead bias confirmado por teste dedicado.

**Revisao obrigatoria:** quant-reviewer apos implementacao (look-ahead bias, calculo de retornos, custos).

---

### Sessao 25 — Metricas de Portfolio vs Benchmark (P1.6)
**Tempo estimado de implementacao:** ~2-3h
**Arquivos novos:** `src/backtest/benchmark_metrics.py`
**Testes novos:** `tests/test_benchmark_metrics.py`

**O que fazer:**

1. **Funcao principal:**

```python
def compute_benchmark_metrics(
    portfolio_returns: pl.Series,    # retornos diarios do portfolio
    benchmark_returns: pl.Series,    # retornos diarios do benchmark
    rf: float = 0.05,
    trading_days: int = 252,
) -> dict[str, float]: ...
```

2. **Metricas a implementar:**

| Categoria | Metrica | Formula |
|-----------|---------|---------|
| Retorno | `cagr` | (final/initial)^(252/n_days) - 1 |
| Retorno | `total_return` | final/initial - 1 |
| Risco | `annualized_volatility` | std(daily_ret) * sqrt(252) |
| Risco | `max_drawdown` | max peak-to-trough (ja existe em cpcv.py — reutilizar) |
| Risco | `max_drawdown_duration_days` | dias no pior drawdown |
| Risco | `calmar_ratio` | CAGR / |max_drawdown| |
| Risk-adj | `sharpe_ratio` | (ann_ret - rf) / ann_vol |
| Risk-adj | `sortino_ratio` | (ann_ret - rf) / downside_vol |
| Risk-adj | `information_ratio` | mean(port - bench) / std(port - bench) * sqrt(252) |
| vs Bench | `alpha` | Jensen's alpha (CAPM regression) |
| vs Bench | `beta` | cov(port, bench) / var(bench) |
| vs Bench | `tracking_error` | std(port - bench) * sqrt(252) |
| vs Bench | `hit_rate_monthly` | % meses com ret_port > ret_bench |
| Turnover | `avg_annual_turnover` | (sum |delta_w| por ano) media |
| Turnover | `avg_positions` | media de posicoes ativas por rebalanceo |

3. **Funcoes auxiliares:**
   - `_compute_drawdown_series(cumulative_returns)` — serie de drawdown
   - `_compute_monthly_returns(daily_returns)` — agrega para hit_rate
   - `_capm_regression(port_excess, bench_excess)` — OLS simples para alpha/beta

4. **Integracao com WalkForwardResult:**
   - `WalkForwardBacktester.run()` chama `compute_benchmark_metrics()` e preenche `result.metrics`

5. **Testes (~25-30 testes):**
   - Cada metrica isoladamente com valores conhecidos
   - Portfolio identico ao benchmark → alpha=0, beta=1, tracking_error=0
   - Portfolio constante (retorno zero) → sharpe negativo
   - Drawdown com dados sinteticos conhecidos
   - hit_rate com dados construidos (5 de 10 meses positivos = 50%)

**Criterio de aceite:** Metricas calculadas corretamente para casos conhecidos; integrado com walk-forward.

---

### Sessao 26 — Relatorio PDF do Benchmark (P1.6 ext)
**Tempo estimado de implementacao:** ~2-3h
**Arquivos novos:** `src/backtest/benchmark_report.py`
**Testes novos:** `tests/test_benchmark_report.py`

**O que fazer:**

1. **Classe `BenchmarkReport`:**

```python
class BenchmarkReport:
    def __init__(
        self,
        result: WalkForwardResult,
        benchmark_name: str = "S&P 500 (SPY)",
        output_dir: str = "data/outputs",
    ) -> None: ...

    def generate(self) -> Path: ...
```

2. **Paginas do PDF (matplotlib + seaborn, backend Agg):**

| Pagina | Conteudo |
|--------|----------|
| 1 | Equity curve: portfolio vs benchmark (duas linhas, eixo log) |
| 2 | Drawdown chart (area preenchida negativa) |
| 3 | Tabela de metricas (Sharpe, Sortino, alpha, beta, CAGR, max DD, etc.) |
| 4 | Rolling Sharpe (janela 252 dias) — portfolio vs benchmark |
| 5 | Heatmap de pesos HRP ao longo do tempo (top 15 ativos por peso medio) |
| 6 | Turnover por rebalanceo (bar chart) |

3. **Output:** `data/outputs/benchmark_report_US.pdf`

4. **Testes (~10-12 testes):**
   - PDF gerado sem erro com dados sinteticos
   - Todas as paginas presentes
   - Funciona com WalkForwardResult minimo (3 tickers, 100 dias)

**Criterio de aceite:** PDF gerado com 6 paginas, graficos legiveis.

---

### Sessao 27 — Integracao e Pipeline Completo (Orchestration)
**Tempo estimado de implementacao:** ~2h
**Arquivos novos:** `src/backtest/run_benchmark.py`
**Arquivos alterados:** `Makefile`
**Testes novos:** `tests/test_run_benchmark.py`

**O que fazer:**

1. **Script orquestrador `run_benchmark.py`:**

```python
def run_us_benchmark(
    config_path: str = "config/tickers.json",
    output_dir: str = "data/outputs",
    use_patchtst: bool = True,    # False = NaiveModelFactory (rapido)
    n_years: int = 10,
) -> WalkForwardResult: ...
```

Pipeline:
```
1. Carregar config (tickers.json)
2. Carregar OHLCV do PostgreSQL (ou baixar se nao existir)
3. Filtrar periodo OOS (ultimos n_years)
4. Instanciar model_factory (PatchTST ou Naive)
5. Instanciar WalkForwardBacktester com config:
   - rebalance_every=5 (semanal)
   - retrain_every=126 (semestral)
   - costs=TransactionCosts(slippage_bps=5, commission_bps=10)
6. Rodar walk-forward
7. Gerar relatorio PDF
8. Salvar resultados em Parquet (para dashboard)
```

2. **Makefile target:**
```makefile
benchmark:
	python -m src.backtest.run_benchmark

benchmark-fast:
	python -m src.backtest.run_benchmark --naive  # NaiveModelFactory para teste rapido
```

3. **Salvar resultados para consumo do dashboard:**
   - `data/outputs/benchmark_equity.parquet` — equity curve diaria
   - `data/outputs/benchmark_metrics.json` — metricas calculadas
   - `data/outputs/benchmark_weights.parquet` — historico de pesos por rebalanceo

4. **Testes:**
   - Pipeline roda end-to-end com mocks (sem PostgreSQL, sem PatchTST real)
   - Output files gerados corretamente
   - Config carregado do JSON

**Criterio de aceite:** `make benchmark-fast` roda sem erros com dados mockados; `make benchmark` roda com dados reais do PostgreSQL.

---

### Sessao 28 — Dashboard: Aba de Benchmark (Extensao)
**Tempo estimado de implementacao:** ~2h
**Arquivos alterados:** `src/dashboard/app.py`
**Testes alterados:** `tests/test_dashboard.py`

**O que fazer:**

1. **Nova aba "Benchmark" no dashboard (aba 0, antes de Performance):**
   - Equity curve interativa (Plotly): portfolio vs SPY, escala log toggle
   - Drawdown chart (area negativa)
   - Tabela de metricas com formatacao condicional (verde se > benchmark)
   - Rolling Sharpe (252d) com slider de janela
   - Heatmap de pesos (top 15 ativos)

2. **Data loaders:**
   - `_load_benchmark_equity()` — le `benchmark_equity.parquet`
   - `_load_benchmark_metrics()` — le `benchmark_metrics.json`
   - `_load_benchmark_weights()` — le `benchmark_weights.parquet`
   - Todos com `@st.cache_data(ttl=300)` e graceful degradation

3. **Testes:**
   - Loaders retornam None se arquivo nao existe
   - Formatacao condicional de metricas

**Criterio de aceite:** Aba Benchmark renderiza com dados do walk-forward; graceful degradation se arquivos ausentes.

---

## Ordem de Execucao e Dependencias

```
Sessao 21 (Config)        ─┐
Sessao 22 (HRP dinamico)  ─┤─► Sessao 24 (Walk-Forward) ─► Sessao 25 (Metricas)
Sessao 23 (Download //l)  ─┘                                     │
                                                                  v
                                                          Sessao 26 (Report PDF)
                                                                  │
                                                                  v
                                                          Sessao 27 (Orchestration)
                                                                  │
                                                                  v
                                                          Sessao 28 (Dashboard)
```

Sessoes 21, 22 e 23 sao independentes entre si — podem ser feitas em paralelo ou em qualquer ordem.

---

## Decisoes Tomadas

| Questao | Decisao | Justificativa |
|---------|---------|---------------|
| Mercado | US apenas | Dados mais limpos, RAG em ingles, sem ajustes |
| Benchmark | SPY buy-and-hold | ETF mais liquido, proxy do S&P 500 |
| Rebalanceamento | Semanal (5 dias uteis) | Realista para fundo ativo |
| Retreino PatchTST | Semestral (126 dias uteis) | Separa ciclo lento do rapido |
| Debate LLM | OFF no backtest | Nao-deterministico, custo proibitivo ($1000+) |
| model_factory | PatchTST (prod) + Naive (dev) | Naive para validar pipeline rapido |
| Custos | slippage=5bps, commission=10bps | Conservador para US large-cap |
| max_weight HRP | min(0.25, 2/n) | Escala com numero de ativos |
| Periodo OOS | 10 anos (2016-2026) | Inclui bull, bear, covid, rate hikes |
| Capital inicial | $1,000,000 | Padrao institucional para backtest |
| Risk-free rate | 5% (T-bill 2024-2026) | Conservador, favorece benchmark |
| SPY no universo | Sim (operavel + benchmark) | Evita exclusao arbitraria |
| Survivorship bias | Aceito (ativos de hoje) | Mitigado por blue-chip selection |

## Decisoes em Aberto (a resolver durante implementacao)

| # | Questao | Opcoes | Resolver em |
|---|---------|--------|-------------|
| 1 | Threshold de rebalanceamento | 0% (sempre) vs 2% (reduz turnover) | Sessao 24 |
| 2 | Warmup period | 504d (2 anos) vs 252d (1 ano) | Sessao 24 |
| 3 | Handling de ticker sem dados em alguma data | Manter peso anterior vs redistribuir | Sessao 24 |

---

## Estimativas de Tempo

| Sessao | Descricao | Tempo dev | Tempo compute |
|--------|-----------|-----------|---------------|
| 21 | Config tickers | 30 min | — |
| 22 | HRP dinamico | 15 min | — |
| 23 | Download paralelo | 1h | 2-5h (50 tickers, 15 anos) |
| 24 | Walk-Forward | 6-8h | — |
| 25 | Metricas benchmark | 2-3h | — |
| 26 | Relatorio PDF | 2-3h | — |
| 27 | Orquestracao | 2h | 4-6h (benchmark completo) |
| 28 | Dashboard benchmark | 2h | — |
| **Total** | | **~16-20h dev** | **~6-11h compute** |

---

## Checklist de Revisao por Sessao

Cada sessao deve passar por:

- [ ] Testes unitarios passando (pytest)
- [ ] Suite completa sem regressoes
- [ ] quant-reviewer em sessoes com logica financeira (24, 25)
- [ ] Zero look-ahead bias (teste dedicado em sessao 24)
- [ ] Logging com loguru (sem print)
- [ ] Type hints em todos os metodos
- [ ] Docstrings Google Style em funcoes publicas
- [ ] Commit atomico com prefixo correto (feat/fix/test)
