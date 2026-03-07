# Backtest: Metricas, Custos de Transacao e Relatorio PDF

Documentacao tecnica dos modulos `src/backtest/cpcv.py` e `src/backtest/report.py`.

---

## Indice

1. [Visao geral](#visao-geral)
2. [Metricas computadas](#metricas-computadas)
3. [Estrategia de avaliacao](#estrategia-de-avaliacao)
4. [Custos de transacao](#custos-de-transacao)
5. [Relatorio PDF](#relatorio-pdf)
6. [Exemplos de uso](#exemplos-de-uso)
7. [Referencia de classes](#referencia-de-classes)

---

## Visao geral

O backtester CPCV (Combinatorial Purged Cross-Validation) avalia um modelo
de previsao de retornos financeiros usando validacao cruzada combinatoria
com purga temporal e embargo. A partir da Sessao 14, o backtester tambem
simula **custos de transacao realistas** (slippage, comissao e impacto de
mercado) e gera **relatorios PDF** automatizados com tabelas de metricas,
curvas de equity e graficos de distribuicao.

### Fluxo de dados

```
DataFrame (OHLCV + features, ticker unico)
    |
    v
CPCVBacktester.run(df, model_factory)
    |-- generate_paths()          -> C(n_splits, n_test_groups) paths
    |-- Para cada path:
    |     |-- model_factory(train_df, test_df)  -> predictions
    |     |-- _evaluate_predictions()           -> FoldResult
    |-- _aggregate_results()      -> BacktestResult
    |
    v
BacktestReport(result, ticker).generate()
    |-- Pagina 1: tabela de metricas + equity curves
    |-- Pagina 2: violinplot Sharpe + barras de drawdown
    |
    v
data/outputs/backtest_report_{TICKER}.pdf
```

---

## Metricas computadas

### Sharpe Ratio (anualizado)

Formula:

```
Sharpe = (mean(excess_returns) / std(excess_returns)) * sqrt(periods_per_year)
```

Onde:
- `excess_returns[i] = return[i] - rf_per_period`
- `rf_per_period = rf / periods_per_year`
- `periods_per_year = 252 / h` (para retornos de h dias)
- Desvio padrao usa `ddof=1` (amostra)
- Retorna `0.0` se volatilidade for zero ou se houver menos de 2 retornos

Parametro padrao: `rf = 0.05` (5% ao ano, taxa livre de risco).

### Maximum Drawdown (MaxDD)

Maximo pico-a-vale da curva de equity cumulativa:

```
MaxDD = min_t((equity[t] - peak[t]) / peak[t])
```

- Valor negativo (ex: `-0.15` = queda de 15%)
- Retorna `0.0` se a serie for vazia ou monotonicamente crescente

### CAGR (Compound Annual Growth Rate)

Taxa de crescimento anual composta:

```
CAGR = equity_final^(1/anos) - 1
anos = n_dias / 252
```

- Retorna `-1.0` se o equity final for negativo ou zero
- Retorna `0.0` se o periodo for zero dias

### Numero de Trades (`n_trades`)

Conta cada mudanca de posicao:
- `flat -> long` = 1 trade
- `long -> flat` = 1 trade
- Posicao resetada no inicio de cada bloco contiguo de teste

### Custo Total (`total_costs`)

Soma de todos os custos de transacao incorridos ao longo de um path:

```
total_costs = sum(trade_cost_i)   para cada mudanca de posicao i
```

Cada `trade_cost` inclui custo fixo (slippage + comissao) e, opcionalmente,
impacto de mercado. Detalhes na secao seguinte.

### Metricas agregadas (`BacktestResult`)

| Campo                  | Descricao                                           |
|------------------------|-----------------------------------------------------|
| `n_paths`              | Numero total de paths combinatoriais avaliados       |
| `mean_sharpe`          | Media do Sharpe Ratio entre paths                   |
| `std_sharpe`           | Desvio padrao do Sharpe entre paths                 |
| `pct_positive_sharpe`  | Fracao de paths com Sharpe > 0                      |
| `mean_max_drawdown`    | Media do MaxDD entre paths                          |
| `mean_cagr`            | Media do CAGR entre paths                           |
| `mean_n_trades`        | Media do numero de trades por path                  |
| `mean_total_costs`     | Media do custo total por path                       |

---

## Estrategia de avaliacao

### Sinal: Long/Flat

A estrategia implementada e simples e deterministica:

- **Long** se `predicted_return > 0` (modelo preve alta)
- **Flat** (sem posicao) caso contrario

Nao ha posicoes short. O retorno em periodos flat e zero.

### Retornos nao-sobrepostos

Para evitar inflacao artifical do Sharpe Ratio por autocorrelacao:

- Retornos sao computados a cada `h` dias (horizonte de previsao)
- Ex: com `h=5`, o primeiro retorno cobre dias 0-5, o segundo dias 5-10, etc.
- Isso produz `~floor(block_len / h)` retornos por bloco contiguo

### Blocos contiguos

Quando `n_test_groups > 1`, o conjunto de teste pode ter lacunas temporais
(ex: grupos 1 e 4 de 6). O avaliador:

1. Identifica blocos contiguos de indices de teste (`_find_contiguous_blocks`)
2. Computa retornos **dentro** de cada bloco (nao cruza lacunas)
3. **Reseta a posicao** (para flat) no inicio de cada novo bloco
4. Concatena todos os retornos para calcular metricas agregadas do path

Isso evita retornos espurios entre periodos temporais desconexos.

### Curva de equity

A curva de equity e cumulativa e composta:

```
equity[0] = 1.0
equity[t+1] = equity[t] * (1 + strategy_return[t])
```

Onde `strategy_return` ja inclui a deducao dos custos de transacao.

---

## Custos de transacao

### Componentes

A dataclass `TransactionCosts` (frozen/imutavel) define tres componentes de custo,
todos em **basis points** (1 bp = 0.0001 = 0.01%):

| Componente            | Default | Descricao                                      |
|-----------------------|---------|-------------------------------------------------|
| `slippage_bps`        | 5.0     | Custo de bid-ask spread por trade               |
| `commission_bps`      | 10.0    | Taxa de corretagem por trade                    |
| `market_impact_bps`   | 0.0     | Impacto de mercado (depende do volume)          |

### Custo fixo

O custo fixo por trade (entrada ou saida de posicao) e:

```
fixed_cost = (slippage_bps + commission_bps) / 10_000
```

Com os defaults: `(5 + 10) / 10_000 = 0.0015` (0.15% por trade).

### Impacto de mercado

O impacto de mercado e um custo variavel, inversamente proporcional a
liquidez do ativo no momento da transacao:

```
relative_vol = volume[i] / avg_volume
impact = (market_impact_bps / 10_000) / sqrt(max(relative_vol, 1e-9))
```

Onde:
- `volume[i]` e o volume do dia da transacao
- `avg_volume` e o volume medio do bloco de teste
- Quando o volume esta abaixo da media (`relative_vol < 1`), o impacto
  **aumenta** (menos liquidez = mais impacto)
- Quando o volume esta acima da media, o impacto **diminui**

Requisitos:
- A coluna `volume` precisa existir no DataFrame de teste
- Se `market_impact_bps > 0` mas nao ha coluna `volume`, o impacto e
  ignorado (com log de aviso)

### Custo total por trade

```
trade_cost = fixed_cost + impact   (se houver mudanca de posicao)
trade_cost = 0                     (se a posicao nao mudar)
```

O custo e deduzido diretamente do retorno da estrategia:

```
strategy_return = (actual_return se long, 0 se flat) - trade_cost
```

### Quando nao usar custos

Se `costs=None` (padrao), nenhum custo e aplicado. Isso e util para:
- Validacao pura do sinal (sem atrito)
- Comparacao antes/depois de custos

---

## Relatorio PDF

### Classe `BacktestReport`

Gera um PDF de 2 paginas usando matplotlib e seaborn.

**Pagina 1 — Metricas e Equity:**
- Tabela com todas as metricas por path (Sharpe, MaxDD, CAGR, Trades, Costs)
- Linha de resumo (AVG) com media e desvio padrao
- Overlay de curvas de equity de todos os paths (colormap tab20)
- Linha horizontal em `y=1.0` como referencia (capital inicial)

**Pagina 2 — Distribuicoes:**
- Violinplot + stripplot do Sharpe Ratio (violin so se >= 4 paths)
- Linhas de referencia: zero (vermelho) e media (verde)
- Grafico de barras do MaxDD por path
- Barras vermelhas para drawdowns > 10%, azuis para menores
- Linha de referencia: media do drawdown (laranja)

### Arquivo de saida

```
data/outputs/backtest_report_{TICKER}.pdf
```

O diretorio e criado automaticamente se nao existir.

---

## Exemplos de uso

### Backtest basico (sem custos)

```python
from src.backtest.cpcv import CPCVBacktester

backtester = CPCVBacktester(n_splits=6, n_test_groups=2, h=5)
result = backtester.run(df, model_factory=my_factory)

print(f"Sharpe medio: {result.mean_sharpe:.3f}")
print(f"MaxDD medio: {result.mean_max_drawdown:.3f}")
print(f"% paths positivos: {result.pct_positive_sharpe:.1%}")
```

### Backtest com custos de transacao

```python
from src.backtest.cpcv import CPCVBacktester, TransactionCosts

costs = TransactionCosts(
    slippage_bps=5.0,       # 0.05% spread
    commission_bps=10.0,    # 0.10% comissao
    market_impact_bps=3.0,  # 0.03% base (ajustado pelo volume)
)

backtester = CPCVBacktester(n_splits=6, costs=costs)
result = backtester.run(df, model_factory=my_factory)

print(f"Trades medios por path: {result.mean_n_trades:.1f}")
print(f"Custo total medio: {result.mean_total_costs:.4f}")
```

### Comparacao antes e depois de custos

```python
bt_sem_custo = CPCVBacktester(n_splits=6)
bt_com_custo = CPCVBacktester(n_splits=6, costs=TransactionCosts())

r1 = bt_sem_custo.run(df, my_factory)
r2 = bt_com_custo.run(df, my_factory)

degradacao = r1.mean_sharpe - r2.mean_sharpe
print(f"Degradacao do Sharpe por custos: {degradacao:.3f}")
```

### Geracao do relatorio PDF

```python
from src.backtest.report import BacktestReport

report = BacktestReport(result, ticker="SPY")
pdf_path = report.generate()
# -> data/outputs/backtest_report_SPY.pdf
```

### Diretorio de saida customizado

```python
report = BacktestReport(result, ticker="NVDA", output_dir="reports/")
pdf_path = report.generate()
# -> reports/backtest_report_NVDA.pdf
```

### Pipeline completo (model_factory exemplo)

```python
import polars as pl
from src.backtest.cpcv import CPCVBacktester, TransactionCosts
from src.backtest.report import BacktestReport

# model_factory: recebe (train, test) e retorna DataFrame com
# colunas "date" e "predicted_return"
def simple_momentum_factory(
    train_df: pl.DataFrame, test_df: pl.DataFrame
) -> pl.DataFrame:
    """Estrategia naive: retorno dos ultimos 5 dias como previsao."""
    returns = test_df.with_columns(
        (pl.col("close").pct_change(5)).alias("predicted_return")
    ).select("date", "predicted_return").drop_nulls()
    return returns

# Configurar e rodar
costs = TransactionCosts(slippage_bps=5.0, commission_bps=10.0)
backtester = CPCVBacktester(n_splits=6, n_test_groups=2, costs=costs)
result = backtester.run(df_spy, model_factory=simple_momentum_factory)

# Gerar relatorio
report = BacktestReport(result, ticker="SPY")
pdf_path = report.generate()
print(f"Relatorio salvo em: {pdf_path}")
```

---

## Referencia de classes

### `TransactionCosts` (frozen dataclass)

| Atributo              | Tipo    | Default | Descricao                          |
|-----------------------|---------|---------|------------------------------------|
| `slippage_bps`        | `float` | `5.0`   | Spread bid-ask em basis points     |
| `commission_bps`      | `float` | `10.0`  | Comissao em basis points           |
| `market_impact_bps`   | `float` | `0.0`   | Impacto de mercado em basis points |

Propriedade:
- `fixed_cost_frac` -> `float`: custo fixo total como fracao decimal

### `FoldResult` (dataclass)

| Atributo       | Tipo              | Descricao                               |
|----------------|-------------------|-----------------------------------------|
| `path_id`      | `int`             | Identificador sequencial do path        |
| `test_groups`  | `tuple[int, ...]` | Indices dos grupos de teste             |
| `sharpe`       | `float`           | Sharpe Ratio anualizado                 |
| `max_drawdown` | `float`           | Maximo drawdown (valor negativo)        |
| `cagr`         | `float`           | CAGR (decimal)                          |
| `n_train`      | `int`             | Amostras de treino (apos purge+embargo) |
| `n_test`       | `int`             | Amostras de teste                       |
| `n_returns`    | `int`             | Retornos nao-sobrepostos                |
| `n_trades`     | `int`             | Numero de mudancas de posicao           |
| `total_costs`  | `float`           | Custo total acumulado                   |
| `equity_curve` | `list[float]`     | Curva de equity cumulativa (inicia em 1.0) |

### `BacktestResult` (dataclass)

| Atributo               | Tipo              | Descricao                            |
|------------------------|-------------------|--------------------------------------|
| `fold_results`         | `list[FoldResult]` | Resultados por path                 |
| `n_paths`              | `int`             | Total de paths avaliados             |
| `mean_sharpe`          | `float`           | Media do Sharpe                      |
| `std_sharpe`           | `float`           | Desvio padrao do Sharpe              |
| `pct_positive_sharpe`  | `float`           | Fracao com Sharpe > 0                |
| `mean_max_drawdown`    | `float`           | Media do MaxDD                       |
| `mean_cagr`            | `float`           | Media do CAGR                        |
| `mean_n_trades`        | `float`           | Media de trades por path             |
| `mean_total_costs`     | `float`           | Media de custo total por path        |

### `BacktestReport`

| Metodo       | Retorno | Descricao                                    |
|--------------|---------|----------------------------------------------|
| `__init__`   | `None`  | Recebe `result`, `ticker`, `output_dir`      |
| `generate()` | `Path`  | Gera PDF e retorna caminho do arquivo        |

Metodos internos (privados):
- `_page_metrics_and_equity()` — Pagina 1
- `_page_sharpe_and_drawdown()` — Pagina 2
- `_render_metrics_table()` — Tabela de metricas
- `_plot_equity_curves()` — Overlay de curvas de equity
- `_plot_sharpe_distribution()` — Violinplot + stripplot
- `_plot_drawdown()` — Barras de drawdown por path
