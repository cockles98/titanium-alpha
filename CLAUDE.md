# Titanium Alpha — Contexto do Projeto

## O que é este projeto
Sistema de fundo de hedge agêntico multi-estratégia que combina:
- PatchTST (Deep Learning para séries temporais financeiras)
- Multi-agente LangGraph (debate entre Analista Técnico, Fundamentalista, Bear, Portfolio Manager)
- RAG financeiro com ChromaDB
- CPCV-OOS (Combinatorial Purged Cross-Validation com Deflated Sharpe Ratio)
- HRP (Hierarchical Risk Parity com Ledoit-Wolf shrinkage) para alocação
- Walk-forward backtest com vol targeting e drawdown killswitch

## Stack tecnológico
Python 3.10+, Polars, NeuralForecast (Nixtla), LangGraph, ChromaDB,
PostgreSQL, Streamlit, Plotly, Docker, Poetry, scikit-learn, scipy

## Padrões obrigatórios de código
- Type hints em TODOS os métodos
- Docstrings Google Style em toda função/classe pública
- Módulos independentes — sem imports circulares
- Logging com loguru (nunca print() em produção)
- Erros sempre re-raised com contexto (nunca bare except)

## Estrutura de pastas (sagrada — não mudar sem perguntar)
src/data/          → ingestão e pipelines de dados
src/models/        → PatchTST, features, previsões
src/agents/        → LangGraph agents, RAG, personas
src/backtest/      → CPCV, CPCV-OOS, walk-forward, benchmark metrics, reports
src/portfolio/     → HRP, decision engine
src/dashboard/     → Streamlit UI (4 abas)
src/utils/         → helpers compartilhados
tests/             → pytest, fixtures em conftest.py
notebooks/         → exploração (nunca importado por src/)
docs/              → documentação e referências
docker/            → Dockerfiles e compose
config/            → tickers.json (52 US large caps + SPY)

## Convenções de git
feat: nova funcionalidade
fix: correção de bug
refactor: sem mudança de comportamento
test: adição/modificação de testes
docs: apenas documentação

## Contexto financeiro importante
- Universo: 52 large caps US + SPY benchmark (config/tickers.json)
- Período de dados: últimos 15 anos (OHLCV diário via yfinance, ~3775 rows/ticker, 199.009 rows totais)
- Benchmark: SPY buy-and-hold
- Métricas-chave: Sharpe Ratio anualizado (rf=0.05), Max Drawdown, CAGR

## Configuração validada (Walk-Forward Benchmark — CPCV-OOS 3-Tier, 982 configs total)
- NaiveModelFactory(lookback=5) — 5-day momentum (proxy para CPCV-OOS)
- rebalance_every=15 (~3 semanas), retrain_every=126 (semestral)
- lookback_days=756 (~3 anos de covariância)
- target_vol=0.10 (10% ann., vol_lookback=63, min_leverage=0.5, max_leverage=1.0)
- HRPConfig: linkage=ward, shrinkage=True (Ledoit-Wolf), max_weight=min(0.06, 2/n)
- TransactionCosts: slippage=5bps, commission=10bps
- min_rebalance_delta=0.02
- top_n=None, killswitch=None (ambos prejudiciais)
- Baseline pré-tuning: Sharpe=0.611, CAGR=14.62%, MaxDD=-31.69%, Beta=0.842
- Recorde walk-forward 2y (pós fine-tuning sessão 39): Sharpe=0.712, CAGR=13.35%, MaxDD=-18.43%, Beta=0.532
- **Recorde walk-forward 10y OOS (sessão 42, DB 15y)**: Sharpe=0.766, CAGR=13.68%, MaxDD=-21.94%, Beta=0.566, Alpha=+2.57%
- TENTATIVA validation_6 (rb=21/mw=0.10): Sharpe=0.462 — REGRESSÃO, revertido
- Análise completa: data/outputs/validation_3tier_analysis.md | data/outputs/validation_6/
- LIÇÃO: max_weight solto (0.10) deixa HRP concentrar ~8.5% em DUK (utilities) — restrição 2/n é funcional

## Limitação conhecida: gap backtest-produção
O pipeline de produção (make decide) usa debate LangGraph, que nunca foi backtestado.
Quando os agentes falham, o sistema cai para fallback PatchTST (predictions.parquet)
ou BUY com confidence=0.5. Ver docs/design_gap_backtest_vs_production.md.

## Histórico de implementação

### Fase 1 — Infraestrutura e Dados (Sessões 1-7)
- Estrutura do projeto, Docker (PostgreSQL + ChromaDB), ingestão OHLCV e notícias
- Feature engineering (RSI, Bollinger, Vol, VWAP, OBV) — zero look-ahead bias
- PatchTST com MQLoss (5 quantis), CDF interpolation para prob_up contínuo
- Pipeline de predições (PostgreSQL → features → PatchTST → Parquet)

### Fase 2 — Camada Agêntica (Sessões 8-12)
- Design e implementação dos 4 agentes LangGraph (pipeline linear)
- State management com TypedDicts + reducers, Pydantic structured output
- RAG financeiro: sentence-transformers → ChromaDB → semantic retrieval
- Integração RAG + Agentes (fundamentalist cita fontes, graceful degradation)

### Fase 3 — Validação e Portfolio (Sessões 13-16)
- CPCV: 15 paths combinatoriais, purge+embargo, custos de transação
- HRP: Lopez de Prado 2016, confidence tilt, clipping, report PDF
- Decision engine: OHLCV → returns → debate → HRP → merge → save

### Fase 4 — Dashboard e Documentação (Sessões 17-20)
- Dashboard Streamlit 4 abas: Benchmark, Performance, War Room, Microstructure
- Streaming per-node com callbacks, replay animado, live debate
- CI/CD (GitHub Actions), Dockerfile, README, ARCHITECTURE.md

### Fase 4b — Benchmark Real (Sessões 21-28)
- Config de 52 tickers, HRP max_weight dinâmico, download paralelo
- Walk-forward backtester: two-cycle (retrain lento + rebalance rápido)
- 16 métricas portfolio vs benchmark, relatório PDF 6 páginas
- Pipeline orquestrador (run_benchmark.py), aba Benchmark no dashboard

### Fase 5 — Melhorias do Benchmark (Sessões 29-35)
- CPCV-OOS: validação de parâmetros com Deflated Sharpe Ratio
- Fixes: log returns fill_null, NaiveModelFactory scaling
- HRP: Ledoit-Wolf shrinkage, Ward linkage
- Volatility targeting (implementado, otimizado para 10% na Fase 8), drawdown killswitch (implementado, desabilitado — prejudicial)
- Grid search: 17+ configs × 8 momentum factories
- Bug fix: custos de transação desaparecendo do port_ret
- Bug fix: prob_up discreto → CDF interpolation contínua
- Bug fix: fan chart sort alfabético → sort por nível de quantil

### Fase 6 — Fixes Quantitativos + Redesign DecisionEngine (Sessão 36)
- Conversão geométrica de rf em todo o pipeline (benchmark_metrics, cpcv, cpcv_oos, walk_forward)
- benchmark_metrics: drawdown peak inicia em 1.0, monthly returns ordenado, IR corrigido
- cpcv: posição flat ganha rf, custo de saída forçada no fim do bloco
- cpcv_oos: DSR converte Sharpe anualizado→diário antes de calcular, purge_days no validator
- walk_forward: início 100% caixa (institucional), vol targeting ex-ante (pré-alocação),
  bankruptcy safeguard, killswitch preserva cash, remoção de tickers duplicados
- HRP: tilt sum-preserving (média ponderada como ponto neutro), waterfilling optimizer
  substituiu clip_and_normalise, turnover_threshold=0.02, previous_weights no optimize()
- DecisionEngine: modelo 3-tier (BUY=HRP, HOLD=HRP*conf, SELL=0), cash implícito,
  fallback PatchTST (predictions.parquet), classificação BUY/HOLD/SELL,
  HRP roda apenas no subset investable, metadata v1.1

### Fase 7 — Fine-Tuning + Data Integrity (Sessões 37-38)
- PatchTST: CDF rearrangement para monotonicity de quantis, NaN guards em predict
- Ingestão: corrigido bug thread-safety do yf.download() → yf.Ticker().history()
  (22 de 52 tickers tinham dados idênticos aos vizinhos adjacentes em tickers.json)
- DB limpo e re-populado com 12 anos de dados (2014-2026) via API thread-safe (atualizado para 15y na sessão 42)
- SPY benchmark adicionado explicitamente à ingestão (não está na lista de tickers)
- Testes: 761 passando, 5 pré-existentes corrigidos (config desatualizada, rf geométrico)
- Benchmark com dados limpos: Sharpe=0.611, CAGR=14.62%, MaxDD=-31.69%, Alpha=0.024

### Fase 8 — Fine-Tuning Pós-PatchTST (Sessão 39)
- Grid search CPCV-OOS 3-tier com 547 configs (T1:249 + T2:149 + T3:149)
- Restrito a parâmetros "pós-predição" (sem forçar re-treinamento do PatchTST)
- Findings: target_vol=0.10 (+0.035 Sharpe), rb=15, max_weight=0.06, ward+shrink+pearson
- top_n e killswitch prejudiciais (removidos); delta, tilt, turnover irrelevantes
- Config champion aplicada ao run_benchmark.py
- Recorde walk-forward 2y (sessão 39): Sharpe=0.712, CAGR=13.35%, MaxDD=-18.43%, Beta=0.532
- Trade-off: CAGR menor (13.35% vs 14.62%) mas risco cortado pela metade (MaxDD -18% vs -32%)

### Fase 9 — Publicação (Sessões 41-42)
- Sessão 41: .claude/agents/ refinados (architect, docs-writer, quant-reviewer, security-data, test-writer),
  dashboard bug fix (decision.get suggested_weight fallback), 17 deprecações use_container_width fixadas,
  RAG populada (172 artigos, P95=101ms), cobertura src.agents=90%
- Sessão 42: DB expandido para 15 anos (199.009 rows, 2011-2026) — antes tinha apenas 5y após reset Docker
- Benchmark 10y OOS (rebalance para mostrar ≥10y de equity após warmup 756d): Sharpe=0.766,
  CAGR=13.68%, MaxDD=-21.94%, Alpha=+2.57%, Beta=0.566, Total Return=259.3%, Tracking Error=9.11%
- 2514 dias de equity ativa (2016-04-19 → 2026-04-17), n_years=13 em run_benchmark.py

**Status atual:** 1002 testes passando | Fases 1-8 completas + Fase 9 (publicação) em andamento

## O que NUNCA fazer
- Nunca hardcode API keys (usar .env + python-dotenv)
- Nunca usar Pandas (usar Polars — é a escolha do projeto)
- Nunca fazer train/test split simples (sempre CPCV)
- Nunca rodar código que acessa APIs reais nos testes (usar mocks)
- Nunca commitar notebooks com output (limpar antes)
