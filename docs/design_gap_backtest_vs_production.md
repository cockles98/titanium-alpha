# Gap de Design: Backtest vs Produção

## Problema

Existe uma inconsistência fundamental entre o pipeline de backtest e o pipeline de produção no que diz respeito ao uso do sinal do PatchTST.

---

## Como funciona no Backtest (`make benchmark`)

```
OHLCV → PatchTST.predict_proba() → prob_up por ticker
                                         ↓
                              prob_up = confidence direto
                                         ↓
                              HRP com confidence tilt
                                         ↓
                              Pesos finais do portfólio
```

O `WalkForwardBacktester` consome o `prob_up` do PatchTST **diretamente** como sinal de confiança para o tilt do HRP. Não há debate de agentes. O código relevante está em `src/backtest/run_benchmark.py`:

```python
class _PatchTSTModelFactory:
    def predict(self, df):
        proba = self._forecaster.predict_proba(df)
        return {row["ticker"]: row["prob_up"] for row in proba.to_dicts()}
```

---

## Como funciona em Produção (`make decide`)

```
OHLCV → PatchTST.predict_proba() → prob_up por ticker
                                         ↓
                              LangGraph (4 agentes)
                              Técnico + Fundamentalista
                              + Bear + Portfolio Manager
                                         ↓
                              action (BUY/HOLD/SELL)
                              + confidence (0.1–1.0)
                                         ↓
                              HRP com confidence tilt
                                         ↓
                              Pesos finais do portfólio
```

O `DecisionEngine` passa o `prob_up` como **contexto nos prompts dos agentes**, mas a confidence que vai para o HRP tilt é gerada pelo debate — não pelo PatchTST diretamente. Se o debate falha (API key ausente, deps não instaladas), o sistema cai em fallback: todos os tickers recebem `action=BUY` e `confidence=0.5`, e o `prob_up` do PatchTST é descartado.

---

## Consequência

O **Sharpe de ~2.7 validado pelo CPCV-OOS** (`make validate`) foi calculado usando apenas o sinal do PatchTST, sem agentes. Esse número **não representa** a performance do sistema completo com debate LangGraph.

Em outras palavras:

| Pipeline | Sinal usado | Backtestado? | Sharpe validado? |
|----------|-------------|--------------|------------------|
| `make benchmark` | PatchTST `prob_up` direto | Sim | Sim (~2.7) |
| `make decide` | Debate LangGraph (4 agentes) | Não | Não |

O sistema de produção nunca foi backtestado. Não há evidência quantitativa de que o debate multi-agente melhora ou piora a performance em relação ao PatchTST sozinho.

---

## Possíveis soluções a implementar

### Opção A — Fallback inteligente no `DecisionEngine`
Quando o debate falha (ou agentes não estão configurados), usar o `prob_up` do PatchTST diretamente como confidence em vez de defaultar para 0.5:

```python
# Em _extract_confidences(), fallback para prob_up:
if not decisions:
    return load_patchtst_probabilities()  # lê predictions.parquet
```

Vantagem: produção degradada continua usando o sinal validado.
Desvantagem: não resolve a inconsistência de backtest.

### Opção B — Backtest do pipeline completo com agentes
Implementar um `AgentModelFactory` que chama o debate LangGraph durante o walk-forward, para que o CPCV-OOS valide o sistema completo.

Vantagem: consistência total entre backtest e produção.
Desvantagem: custo alto de API (dezenas de milhares de chamadas ao Claude), lentidão.

### Opção C — Agentes como filtro, PatchTST como sinal base
Redesenhar o pipeline para que os agentes apenas **filtrem** sinais (convertendo alguns BUY em HOLD/SELL), mas o sinal quantitativo base seja sempre o `prob_up`:

```
prob_up → HRP base weights
              ↓ (se agentes disponíveis)
         agente vote → zerar posições de alta convicção bearish
              ↓
         pesos finais
```

Vantagem: backtest do sinal base permanece válido; agentes adicionam valor incremental.
Desvantagem: requer redesign da integração agente-portfólio.

---

## Status atual

- Pendente — documentado em 2026-03-13
- Nenhuma das opções foi implementada
- O sistema funciona em produção mas com sinal degradado (0.5 uniform) quando agentes não estão disponíveis
