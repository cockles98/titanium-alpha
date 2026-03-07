# Titanium Alpha -- Real Benchmark Plan

> Guia completo para rodar um benchmark realista com 50+ ativos, 5-15 anos OOS,
> comparando contra Ibovespa (BR) ou S&P 500 (US).

---

## 1. Visao Geral

O sistema atual esta configurado para 4 ativos US (SPY, NVDA, AAPL, QQQ) com
~2 anos de lookback. Ele gera um **snapshot unico** de decisoes (decisions.json),
mas **nao simula o portfolio ao longo do tempo**. Nao existe loop temporal,
equity curve, nem comparacao contra benchmark.

Para um benchmark realista precisamos:

| Dimensao | Atual | Alvo |
|----------|-------|------|
| Ativos | 4 | 50+ |
| Janela OOS | ~2 anos | 5-15 anos |
| Simulacao temporal | **snapshot unico** | **walk-forward loop** |
| Equity curve | **nao existe** | acumulada com rebalanceamento |
| Benchmark | nenhum | Ibovespa ou S&P 500 |
| Custos | genericos | por ativo (liquidez) |
| Debate LLM | sequencial | paralelo |
| CPCV | valida modelo, nao portfolio | complementar ao walk-forward |

### O que existe vs o que falta

O projeto tem todas as pecas individuais (previsao, debate, alocacao), mas
falta o **motor de simulacao temporal** que conecta tudo:

| Componente | Existe? | Status |
|---|---|---|
| Ingestao de dados (OHLCV + news) | Sim | Pronto |
| Feature engineering (RSI, BB, vol) | Sim | Pronto |
| PatchTST (previsao 5 dias) | Sim | Pronto |
| Debate LLM (4 agentes) | Sim | Pronto |
| HRP (alocacao de pesos) | Sim | Pronto |
| CPCV (validacao do modelo por ticker) | Sim | Valida PatchTST, nao o portfolio |
| **Walk-forward backtester** | **Nao** | **Peca central que falta** |
| **Equity tracker** | **Nao** | Acumula retornos do portfolio |
| **Benchmark comparison** | **Nao** | Portfolio vs SPY/Ibovespa |
| **Metricas de portfolio** | **Nao** | Sharpe, alpha, beta, drawdown |

**CPCV vs Walk-Forward:** O CPCV valida se o PatchTST preve bem cada ativo
isoladamente. O walk-forward simula o portfolio completo (50 ativos com pesos
HRP) operando ao longo do tempo. Sao complementares — CPCV valida o modelo,
walk-forward valida a estrategia.

---

## 2. Escolha do Universo de Ativos

### Opcao A: Acoes BR (Ibovespa)

**Tickers (sufixo `.SA` para yfinance):**

```python
BR_TICKERS = [
    # Commodities / Energia
    "PETR4.SA", "VALE3.SA", "CSAN3.SA", "PRIO3.SA", "SUZB3.SA",
    # Financeiro
    "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "SANB11.SA", "B3SA3.SA",
    # Varejo / Consumo
    "MGLU3.SA", "LREN3.SA", "ABEV3.SA", "NTCO3.SA", "RADL3.SA",
    # Industria / Infraestrutura
    "WEG3.SA", "RENT3.SA", "EQTL3.SA", "CCRO3.SA", "RAIL3.SA",
    # Saude
    "HAPV3.SA", "RDOR3.SA", "FLRY3.SA",
    # Telecomunicacoes / Tech
    "VIVT3.SA", "TOTS3.SA", "LWSA3.SA",
    # Utilities
    "ELET3.SA", "SBSP3.SA", "CMIG4.SA", "CPFE3.SA", "TAEE11.SA",
    # Siderurgia / Mineracao
    "GGBR4.SA", "CSNA3.SA", "USIM5.SA", "BRAP4.SA",
    # Papel / Celulose
    "KLBN11.SA",
    # Seguros
    "BBSE3.SA", "IRBR3.SA",
    # Educacao
    "YDUQ3.SA", "COGN3.SA",
    # Alimentos
    "JBSS3.SA", "BRFS3.SA", "MRFG3.SA", "BEEF3.SA",
    # Imobiliario
    "CYRE3.SA", "MRVE3.SA", "EZTC3.SA",
    # Benchmark
    "^BVSP",  # Ibovespa index (benchmark, nao opera)
]
# Total: ~48 ativos + 1 benchmark
```

**Vantagens BR:**
- Dados yfinance disponiveis para B3 (`.SA` suffix)
- Mercado menos eficiente que US (mais alpha potencial)
- Benchmark claro: Ibovespa (^BVSP)

**Desafios BR:**
- Calendario B3 (~245 dias/ano vs 252 US) — ajustar anualizacao do Sharpe
- Liquidez variavel (small caps com spread alto)
- Noticias em portugues — RAG precisa de modelo multilingual
- Dados historicos limitados para IPOs recentes (RDOR3, PRIO3)

### Opcao B: Acoes US (S&P 500)

```python
US_TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO", "CRM", "AMD",
    # Financeiro
    "JPM", "BAC", "GS", "MS", "WFC", "BLK", "AXP", "C",
    # Saude
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO",
    # Consumo
    "PG", "KO", "PEP", "COST", "WMT", "HD", "MCD",
    # Energia
    "XOM", "CVX", "COP", "SLB",
    # Industrial
    "CAT", "HON", "UPS", "BA", "GE", "RTX",
    # Utilities / Real Estate
    "NEE", "DUK", "AMT", "PLD",
    # Comunicacao
    "DIS", "NFLX", "CMCSA",
    # Materiais
    "LIN", "APD", "NEM",
    # Benchmark
    "SPY",  # S&P 500 ETF (benchmark)
]
# Total: ~50 ativos (SPY serve como benchmark E ativo operavel)
```

**Vantagens US:**
- Dados de alta qualidade (15+ anos para blue chips)
- Liquidez profunda (custos de transacao baixos)
- Noticias abundantes em ingles (RAG funciona out-of-the-box)
- 252 dias/ano (anualizacao padrao ja implementada)

**Desafios US:**
- Mercado mais eficiente (alpha mais dificil)
- Survivorship bias — selecionar ativos que existiam no inicio do periodo

### Recomendacao

**Para um primeiro benchmark realista, comecar com US (Opcao B):**
- Dados mais limpos, maior historico, RAG funciona sem mudancas
- Depois adaptar para BR como segundo benchmark

---

## 3. Mudancas Necessarias no Codigo

### 3.1 Prioridade 1 — Obrigatorio

#### P1.1: Tickers configuraveis (30 min)

Criar `config/tickers.py` ou arquivo YAML externo:

```python
# src/config.py (novo)
from pathlib import Path
import json

def load_tickers(path: str = "config/tickers.json") -> list[str]:
    with open(path) as f:
        cfg = json.load(f)
    return cfg["tickers"]

def load_benchmark(path: str = "config/tickers.json") -> str:
    with open(path) as f:
        cfg = json.load(f)
    return cfg["benchmark"]
```

```json
// config/tickers.json
{
    "tickers": ["AAPL", "MSFT", "GOOG", ...],
    "benchmark": "SPY",
    "market": "US"
}
```

**Arquivos a alterar:**
- `src/data/ingestion.py` — `DEFAULT_TICKERS`
- `src/portfolio/decision_engine.py` — `DEFAULT_TICKERS`
- `src/models/predict.py` — importa de `ingestion.py`
- `src/agents/graph.py` — importa de `ingestion.py`

#### P1.2: HRP max_weight dinamico (15 min)

```python
# decision_engine.py
n = len(self.tickers)
dynamic_config = HRPConfig(
    max_weight=min(0.25, 2.0 / n),  # max 2x equal weight
    confidence_tilt_cap=0.20,
)
```

Para 50 ativos: `max_weight = 2/50 = 0.04` (4%), permitindo concentracao ate
2x o peso igual.

#### P1.3: Paralelizacao do debate LLM (2-3h)

O gargalo principal com 50 ativos. Atualmente `run_agent_debate()` faz um loop
sequencial por ticker (4 chamadas LLM cada = 200 chamadas sequenciais).

```python
# src/agents/graph.py — modificar run_agent_debate()
import asyncio
from concurrent.futures import ThreadPoolExecutor

def run_agent_debate(
    tickers: list[str] | None = None,
    max_workers: int = 10,  # limite de paralelismo
) -> tuple[list[FinalDecision], dict[str, dict]]:
    graph = build_investment_graph()

    def _run_single(ticker: str) -> tuple[FinalDecision | None, dict]:
        state = make_empty_state(ticker)
        result = graph.invoke(state)
        return result.get("final_decision"), dict(result)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_single, t): t for t in tickers}
        # ... coletar resultados
```

**Ganho esperado:** 200s sequencial → ~20s com 10 workers.

**Custo Claude Sonnet por run (~50 ativos):**
- ~200 chamadas x ~1000 tokens in + ~500 tokens out
- Input: 200k tokens x $3/1M = $0.60
- Output: 100k tokens x $15/1M = $1.50
- **Total: ~$2 por run** (aceitavel)

#### P1.4: CPCV integrado ao benchmark (4-6h)

CPCV atualmente e independente — precisa de um `model_factory` para cada ativo.
Para benchmark realista:

```python
def patchtst_model_factory(train_df: pl.DataFrame, test_df: pl.DataFrame) -> pl.DataFrame:
    """Factory que treina PatchTST no train e prediz no test."""
    forecaster = TitaniumForecaster(h=5, input_size=60)
    features_train = compute_all_features(train_df)
    forecaster.fit(features_train)

    features_test = compute_all_features(test_df)
    predictions = forecaster.predict(features_test)
    return predictions
```

**Tempo estimado por ticker (15 anos, 15 paths CPCV):**
- 15 treinos PatchTST x ~10 min = ~2.5 horas por ticker
- 50 tickers = 125 horas sequencial

**Com paralelismo (10 workers):** ~12-15 horas.

**Alternativa rapida para primeiro benchmark:** usar modelo mais simples
como `model_factory` (regressao linear em features) para validar o pipeline
antes de rodar PatchTST completo:

```python
def linear_model_factory(train_df, test_df):
    """Factory rapida para validacao do pipeline."""
    # Treinar regressao em RSI, BB, vol → retorno
    # 15 seconds por path vs 10 minutos
    ...
```

### 3.2 Prioridade 2 — Importante

#### P2.1: Download paralelo de dados (1h)

```python
# src/data/ingestion.py
from concurrent.futures import ThreadPoolExecutor
import time

def _download_single(ticker: str, period: str) -> pl.DataFrame:
    time.sleep(0.5)  # rate limit yfinance
    return yf.download(ticker, period=period)

with ThreadPoolExecutor(max_workers=5) as pool:
    results = pool.map(_download_single, tickers)
```

**Ganho:** 50 tickers x 15 anos de ~10-25h sequencial para ~2-5h paralelo.

#### P2.2: Custos de transacao por ativo (1h)

```python
# Custos adaptativos baseados em liquidez
def estimate_costs(ticker: str, market: str = "US") -> TransactionCosts:
    if market == "BR":
        # B3: spreads maiores, especialmente small caps
        if ticker in LARGE_CAP_BR:
            return TransactionCosts(slippage_bps=8, commission_bps=15)
        else:
            return TransactionCosts(slippage_bps=20, commission_bps=15, market_impact_bps=10)
    else:
        # US: mercado liquido
        return TransactionCosts(slippage_bps=5, commission_bps=10)
```

#### P2.3: Lookback dinamico para HRP (30 min)

```python
# Para 50 ativos, covariancia precisa de mais dados
lookback = max(504, n_tickers * 10)  # minimo 10x o numero de ativos
# 50 ativos → lookback = 504 (ja suficiente)
# 100 ativos → lookback = 1000
```

#### P2.4: News sources para BR (2-3h, so se usar Opcao A)

Adicionar RSS feeds brasileiros e expandir `TICKER_KEYWORDS`:

```python
BR_RSS_FEEDS = [
    "https://www.valor.com.br/feed",
    "https://www.infomoney.com.br/feed/",
    "https://br.investing.com/rss/news.rss",
]

TICKER_KEYWORDS["PETR4.SA"] = ["Petrobras", "petroleo", "pre-sal"]
TICKER_KEYWORDS["VALE3.SA"] = ["Vale", "minerio", "iron ore"]
# ... etc para cada ticker
```

**Modelo de embedding multilingual:**
```python
# Trocar all-MiniLM-L6-v2 por multilingual
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
```

#### P1.5: Walk-Forward Backtester — PECA CENTRAL (6-8h)

**Este e o modulo mais importante que falta no projeto.** Sem ele, nao existe
benchmark. O sistema atual gera um snapshot unico de decisoes; o walk-forward
simula o portfolio operando ao longo do tempo.

```python
# src/backtest/walk_forward.py (novo modulo)
@dataclass
class WalkForwardConfig:
    retrain_every: int = 126       # retreinar a cada 6 meses (~126 dias)
    rebalance_every: int = 21      # rebalancear mensalmente (~21 dias)
    lookback_days: int = 504       # janela de treino (~2 anos)
    initial_capital: float = 1_000_000.0
    costs: TransactionCosts | None = None
    use_debate: bool = False       # incluir debate LLM? ($2 por rebalanceo)

class WalkForwardBacktester:
    def run(
        self,
        ohlcv: pl.DataFrame,         # OHLCV completo (50 tickers, 10 anos)
        tickers: list[str],
        benchmark_ticker: str = "SPY",
    ) -> WalkForwardResult:
        """Simula o portfolio ao longo do tempo.

        Loop principal:
        1. Para cada dia de rebalanceamento:
           a. Recortar dados ate essa data (sem look-ahead)
           b. Treinar PatchTST nos ultimos lookback_days
           c. (Opcional) Rodar debate LLM
           d. Rodar HRP com confidences → pesos
           e. Calcular retorno do portfolio ate proximo rebalanceo
           f. Aplicar custos de transacao nas mudancas de posicao
           g. Registrar: equity, pesos, retorno, custos

        2. Ao final:
           - Equity curve acumulada
           - Metricas vs benchmark (Sharpe, alpha, beta, drawdown)
           - Historico de pesos por rebalanceo
        """
```

**Otimizacao critica — separar frequencia de treino e rebalanceamento:**

O pipeline completo nao precisa rodar inteiro a cada rebalanceamento.
Treinar PatchTST e caro (~10 min por ticker); gerar previsao com modelo
ja treinado e barato (~1s). O walk-forward separa os dois ciclos:

```
Ciclo lento (a cada 6 meses, ~20 vezes em 10 anos):
  - Retreinar PatchTST nos dados ate essa data
  - Atualizar modelo salvo em checkpoints

Ciclo rapido (semanal, ~521 vezes em 10 anos):
  - Computar features atualizadas (RSI, BB, vol) → ~5s
  - Gerar previsao com modelo JA treinado → ~1s
  - Rodar HRP com novas previsoes → novos pesos → <1s
  - Calcular retorno da semana → acumular equity
  - Aplicar custos de transacao nas mudancas de posicao
```

**Impacto nos custos (50 ativos, 10 anos, rebalanceamento semanal):**

| Cenario | Treinos PatchTST | Chamadas LLM | Custo | Tempo (paralelo) |
|---------|-----------------|-------------|-------|------------------|
| PatchTST + HRP puro | 20 | 0 | **$0** | **4-6h** |
| + debate LLM em cada rebalanceo | 20 | 521 x 200 = 104k | **~$1000** | ~15h |
| + debate LLM apenas no run final | 20 | 200 | **~$2** | 4-6h |

**Recomendacao:** rodar walk-forward com PatchTST + HRP puro ($0, 4-6h).
Depois comparar 1 run com debate vs sem debate para medir se os agentes
agregam valor ao resultado. Se sim, considerar incluir debate no walk-forward
(custo de ~$1000 para o benchmark completo).

**Exemplo de uso:**
```python
backtester = WalkForwardBacktester(
    config=WalkForwardConfig(
        retrain_every=126,      # retreinar PatchTST a cada 6 meses
        rebalance_every=5,      # rebalancear semanalmente (~5 dias uteis)
        use_debate=False,       # sem LLM no backtest ($0)
    )
)
result = backtester.run(ohlcv, tickers, benchmark_ticker="SPY")

# result.equity_curve → Series com valor do portfolio por dia
# result.benchmark_curve → Series com valor do SPY por dia
# result.metrics → Sharpe, alpha, beta, max_drawdown, CAGR, etc.
# result.weight_history → DataFrame com pesos por rebalanceo
# result.trade_log → lista de trades com custos
```

**O que o walk-forward faz que o CPCV nao faz:**
- CPCV valida o **modelo** (PatchTST preve bem?) por ticker isolado
- Walk-forward valida a **estrategia** (o portfolio inteiro gera alpha?)
- Walk-forward produz a equity curve que vai no relatorio final

#### P1.6: Metricas de portfolio vs benchmark (2-3h)

```python
# src/backtest/benchmark_metrics.py (novo modulo)
def compute_portfolio_metrics(
    portfolio_equity: pl.Series,    # valor diario do portfolio
    benchmark_equity: pl.Series,    # valor diario do benchmark
    rf: float = 0.05,               # taxa livre de risco anual
    trading_days: int = 252,
) -> dict[str, float]:
    """Todas as metricas necessarias para o relatorio."""
```

(Ver secao 5 para lista completa de metricas.)

### 3.3 Prioridade 3 — Nice to Have

| Item | Descricao | Esforco |
|------|-----------|---------|
| Survivorship bias | Usar lista de constituintes historica do indice | 2-4h |
| GPU batching | Treinar multiplos PatchTST na mesma GPU em batch | 4h |
| Cache LLM | Cachear respostas identicas entre runs | 1h |
| Dashboard escala | Adaptar graficos para 50+ ativos (scroll, filtros) | 2h |

---

## 4. Estimativas de Tempo e Recursos

### 4.1 Tempo de execucao (por run, 50 ativos, 15 anos)

| Etapa | Sequencial | Paralelo (10 workers) |
|-------|-----------|----------------------|
| Download yfinance | 10-25h | 2-5h |
| Feature engineering | 1 min | 1 min |
| PatchTST treino | 5-12h | 30-45 min |
| Debate LLM | 3-5 min | 10-20s |
| HRP alocacao | <1s | <1s |
| CPCV (se rodar) | 125h+ | 12-15h |
| Walk-forward (10 anos, mensal) | 60-120h | 6-12h |
| **Total sem CPCV** | **75-157h** | **9-18h** |
| **Total com CPCV** | **200h+** | **20-30h** |

**Nota:** O download so precisa rodar 1x. Depois disso, dados ficam no
PostgreSQL e os re-runs usam o cache local.

### 4.2 Recursos computacionais

| Recurso | Requisito |
|---------|-----------|
| RAM | 4 GB (pico durante features com 50 tickers) |
| Disco | ~12 GB (PostgreSQL + checkpoints PatchTST) |
| GPU | Opcional mas recomendado para PatchTST (reduz treino de 10min para 2min) |
| CPU | 8+ cores para paralelismo efetivo |
| Rede | Estavel para download yfinance (rate limited) |

### 4.3 Custo por run

| Componente | Custo |
|------------|-------|
| Claude Sonnet (50 tickers x 4 calls) | ~$2.00 |
| Infraestrutura (local) | $0 |
| yfinance API | Gratuito |
| NewsAPI (se usar) | Gratuito (tier basico) |
| **Total** | **~$2 por run** |

Para CPCV com debate LLM em cada fold: 15 paths x $2 = **~$30 por ticker**,
$1500 para 50 tickers. **Recomendacao:** rodar CPCV sem debate LLM (usar
apenas PatchTST + HRP) e reservar debate para o pipeline final.

---

## 5. Metricas do Benchmark

### 5.1 Metricas obrigatorias

```python
# Implementar em src/backtest/benchmark_metrics.py
def compute_benchmark_metrics(
    portfolio_returns: pl.Series,
    benchmark_returns: pl.Series,
    rf: float = 0.05,  # taxa livre de risco (SELIC para BR, T-bill para US)
) -> dict:
    return {
        # Retorno
        "cagr": ...,                    # Compound Annual Growth Rate
        "total_return": ...,            # Retorno total acumulado
        "annualized_return": ...,       # Retorno anualizado

        # Risco
        "annualized_volatility": ...,   # Vol anualizada (√252 * daily_std)
        "max_drawdown": ...,            # Drawdown maximo
        "max_drawdown_duration": ...,   # Duracao do pior drawdown (dias)
        "calmar_ratio": ...,            # CAGR / Max Drawdown

        # Risco-retorno
        "sharpe_ratio": ...,            # (ret - rf) / vol
        "sortino_ratio": ...,           # (ret - rf) / downside_vol
        "information_ratio": ...,       # (ret - bench_ret) / tracking_error

        # vs Benchmark
        "alpha": ...,                   # Jensen's alpha (CAPM)
        "beta": ...,                    # Beta vs benchmark
        "tracking_error": ...,          # Std(portfolio_ret - bench_ret)
        "hit_rate": ...,                # % de meses com ret > bench

        # Turnover
        "avg_annual_turnover": ...,     # Rotacao media anual
        "avg_positions": ...,           # Numero medio de posicoes
    }
```

### 5.2 Taxa livre de risco

| Mercado | Taxa | Fonte |
|---------|------|-------|
| US | Treasury 3-month (~5% em 2024-2026) | FRED API |
| BR | SELIC (~10-13% historico) | BCB API |

**Impacto:** Sharpe ratio BR sera menor que US se usar rf=SELIC vs rf=T-bill.
Comparar Sharpe entre mercados nao e direto — usar Information Ratio vs
benchmark local.

### 5.3 Calendario de trading

```python
# Ajuste de anualizacao
TRADING_DAYS = {
    "US": 252,
    "BR": 245,  # B3 tem mais feriados
}
annualized_vol = daily_vol * np.sqrt(TRADING_DAYS[market])
```

---

## 6. Estrategia de Benchmark Recomendada

### Fase 1: Validacao do Pipeline (1-2 dias)

1. Rodar com 10 ativos US, 5 anos, modelo linear simples como `model_factory`
2. Verificar que CPCV, HRP, e metricas funcionam corretamente
3. Comparar contra SPY buy-and-hold
4. Meta: pipeline roda sem erros, metricas sao plausíveis

### Fase 2: Benchmark US Completo (3-5 dias)

1. 50 ativos US, 10 anos OOS (2016-2026)
2. PatchTST como modelo, CPCV com 15 paths
3. Walk-forward: retreinar a cada 6 meses (rolling)
4. Custos realistas (slippage 5bps, comissao 10bps)
5. Benchmark: SPY buy-and-hold
6. Meta: Sharpe > 0.5, Information Ratio > 0, Max DD < 30%

### Fase 3: Benchmark BR (opcional, 3-5 dias)

1. 48 ativos BR, 10 anos OOS (2016-2026)
2. Adaptar RAG para portugues (multilingual embeddings)
3. Custos BR (slippage 10-20bps, comissao 15bps)
4. Benchmark: Ibovespa (^BVSP)
5. rf = SELIC media do periodo
6. Meta: superar Ibovespa ajustado por risco

### Fase 4: Relatorio Final

Gerar PDF com:
- Equity curves (portfolio vs benchmark)
- Drawdown chart
- Tabela de metricas (Sharpe, Sortino, alpha, beta, etc.)
- Distribuicao de Sharpe por path CPCV (violinplot)
- Heatmap de pesos HRP ao longo do tempo
- Analise de atribuicao (quais ativos contribuiram mais)

---

## 7. Armadilhas a Evitar

### 7.1 Survivorship Bias

**Problema:** Selecionar ativos que existem hoje ignora empresas que faliram,
foram deslistadas, ou adquiridas. O backtest fica otimista demais.

**Mitigacao:**
- Usar lista de constituintes do indice no INICIO do periodo OOS
- Para S&P 500: Wikipedia tem historico de inclusoes/exclusoes
- Para Ibovespa: B3 publica carteiras teoricas trimestrais
- Alternativa pratica: filtrar apenas ativos com dados completos no periodo

### 7.2 Look-Ahead Bias na Selecao

**Problema:** Escolher "os 50 melhores ativos" usando informacao futura.

**Mitigacao:**
- Definir o universo ANTES de rodar qualquer backtest
- Usar criterios objetivos (ex: top 50 por market cap na data inicial)
- Documentar criterio de selecao

### 7.3 Overfitting de Hiperparametros

**Problema:** Ajustar PatchTST (input_size, h, learning_rate) nos dados OOS.

**Mitigacao:**
- CPCV ja protege contra isso (15 paths independentes)
- Nao ajustar hiperparametros apos ver resultados OOS
- Reportar resultados do PRIMEIRO run, nao do melhor

### 7.4 Transaction Costs Irrealistas

**Problema:** Ignorar custos infla retornos. Custos excessivos mata qualquer alpha.

**Mitigacao:**
- Usar custos conservadores mas realistas (tabela na secao 3.2)
- Reportar resultados COM e SEM custos
- Calcular break-even cost (ate quanto de custo o alpha sobrevive)

### 7.5 Rebalanceamento Irrealista

**Problema:** Rebalancear diariamente gera turnover altissimo.

**Mitigacao:**
- Rebalancear semanal ou mensal (mais realista)
- Adicionar threshold minimo para rebalancear (ex: so mover se delta > 2%)
- Reportar turnover anual

---

## 8. Ordem de Implementacao Sugerida

| # | Tarefa | Esforco | Dependencia | Prioridade |
|---|--------|---------|-------------|------------|
| 1 | Criar `config/tickers.json` e `src/config.py` | 30 min | - | P1 |
| 2 | Ajustar HRP `max_weight` dinamico | 15 min | - | P1 |
| 3 | **Walk-forward backtester** (`src/backtest/walk_forward.py`) | **6-8h** | 1, 2 | **P1 — critico** |
| 4 | **Metricas de portfolio** (`src/backtest/benchmark_metrics.py`) | **2-3h** | 3 | **P1 — critico** |
| 5 | Paralelizar download yfinance | 1h | - | P1 |
| 6 | Paralelizar `run_agent_debate()` | 2-3h | - | P1 |
| 7 | Integrar CPCV no pipeline com `model_factory` | 4-6h | 1 | P2 |
| 8 | Custos de transacao por ativo | 1h | - | P2 |
| 9 | Rodar benchmark US (50 ativos, 10 anos) | 9-18h (compute) | 1-6 | - |
| 10 | Gerar relatorio PDF final | 2-3h | 9 | - |
| 11 | (Opcional) Adaptar para BR + RAG multilingual | 4-6h | 9 | P3 |

**Sem os itens 3 e 4, nao existe benchmark.** O resto sao otimizacoes.

**Total de desenvolvimento:** ~25-35 horas de trabalho.
**Total de compute:** ~9-18 horas por benchmark completo (paralelo, sem CPCV).

---

## 9. Decisoes Tomadas e em Aberto

### Decidido

| Questao | Decisao | Justificativa |
|---------|---------|---------------|
| Primeiro benchmark | **US (50 ativos, S&P 500)** | Dados mais limpos, RAG funciona sem mudancas |
| Horizonte temporal | **10 anos OOS** | Equilibrio entre robustez e tempo de compute |
| Rebalanceamento | **Semanal (~521 iteracoes)** | Realista para fundo ativo |
| Retreino PatchTST | **A cada 6 meses (~20 treinos)** | Separa ciclo lento do rapido |
| Debate LLM no backtest | **Nao — rodar sem LLM ($0)** | LLM nao-deterministico; testar valor em 1 run separado |
| Custo alvo | **~$0 para walk-forward, ~$2 para 1 run com debate** | CPCV com debate ($1500) nao vale |

### Em aberto

1. **Modelo no CPCV:** PatchTST (lento, ~15h) ou proxy linear (rapido,
   ~15 min) para validacao inicial?

2. **Threshold de rebalanceamento:** rebalancear sempre ou so se delta > 2%?
   (impacta turnover e custos)

3. **Survivorship bias:** usar lista historica de constituintes do S&P 500
   ou aceitar o vies com os 50 ativos de hoje?

4. **BR como segundo benchmark:** implementar apos US ou descartar?
