# Titanium Alpha — Plano de Melhorias do Benchmark

> Plano detalhado para melhorar a performance do walk-forward benchmark.
> Baseline (NaiveModelFactory): Sharpe=0.537, CAGR=12.74%, MaxDD=-33.86%, Alpha=1.99%
> Universo: 52 US large caps + SPY benchmark, rebalanceo semanal, custos 15bps.

---

## Filosofia Central: Nenhuma Melhoria Sem Validacao

**Regra de Ouro:** Qualquer mudanca de parametro que melhore metricas no OOS e suspeita de overfitting ate prova em contrario. A prova e o CPCV-OOS com Deflated Sharpe Ratio.

### Classificacao de Mudancas
```
1. Preferencia de risco  → nao precisa de CPCV (ex: volatility target = 10%)
2. Mudanca estrutural    → testar via CPCV-OOS (ex: ward linkage, lookback)
3. Tuning de parametro   → obrigatorio CPCV-OOS + Deflated Sharpe (ex: rebalance_every)
```

### Fluxo por Sessao
```
1. Implementar mudanca com testes unitarios
2. Rodar quant-reviewer para validar integridade
3. Se tipo 2 ou 3: rodar CPCV-OOS e reportar metricas
4. Aceitar SOMENTE se pct_positive > 66% (10/15 paths) E deflated_sharpe > baseline
5. Atualizar CLAUDE.md com decisao e justificativa
```

---

## Decisoes Tomadas

| Questao | Decisao | Justificativa |
|---------|---------|---------------|
| Infraestrutura de validacao | CPCV-OOS antes de tudo | Sem isso, qualquer melhoria e data mining |
| Volatility targeting | Prioridade maxima | Parametro de preferencia, nao de alpha; melhora Sharpe mecanicamente |
| Vol killswitch vs vol targeting | Fundir em vol targeting | Mesma mecanica (exposure = target/realized), com clamp [0.5, 1.0] |
| DD killswitch | Implementar mas esperar rejeicao | V-recoveries (Mar 2020) tipicamente destroem o resultado |
| NaiveModelFactory lookback | 5d -> testar 21/63/126 | Momentum 5d e ruido puro; Jegadeesh & Titman (1993) valida 63-252d |
| NaiveModelFactory scaling | Proporcional ao lookback | `ret * 10` satura com lookback > 5; corrigir para `ret * (50/lookback)` |
| HRP linkage | Testar ward vs single | Single causa chaining com 52 ativos (Raffinot, 2017) |
| Covariance shrinkage | Ledoit-Wolf | 3 linhas de codigo, estabiliza pesos, reduz turnover |
| Momentum pre-filter | Skip | Redundante com NaiveModelFactory lookback ajustavel |
| Sector constraints | Skip | Redundante com max_weight dinamico (2/n ~ 4%) |
| Liquidity filter | Skip | Universo 100% mega/large caps, nenhum excluido |
| Tracking error budget | Skip | Contradiz filosofia HRP; complexidade alta |
| Correlation killswitch | Skip | Complexo, fragil, janela 21d ruidosa com 52 ativos |
| Multiple testing correction | Deflated Sharpe Ratio | Bonferroni excessivamente conservador para parametros correlacionados |
| CPCV-OOS acceptance criteria | pct_positive > 66% (10/15) | 60% = 9/15 nao e estatisticamente significativo (p~0.30 binomial) |

---

## Sessoes de Implementacao

### Sessao 29 — CPCV-OOS Parameter Validator (PRE-REQUISITO)
**Tempo estimado:** ~4-6h
**Arquivos novos:** `src/backtest/cpcv_oos.py`
**Testes novos:** `tests/test_cpcv_oos.py`

**O que fazer:**

1. **Classe `CPCVParameterValidator`:**

```python
class CPCVParameterValidator:
    def __init__(
        self,
        ohlcv: pl.DataFrame,
        tickers: list[str],
        benchmark_ticker: str = "SPY",
        n_splits: int = 6,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,  # % do total de dados como embargo
    ) -> None: ...

    def validate(
        self,
        config: WalkForwardConfig,
        model_factory: ModelFactory,
        baseline_sharpe: float | None = None,
    ) -> ValidationResult: ...

    def grid_search(
        self,
        configs: list[WalkForwardConfig],
        model_factory: ModelFactory,
    ) -> list[ValidationResult]: ...
```

2. **Logica do `validate()`:**

```
OOS data split em 6 splits temporais contiguos
Para cada combinacao C(6,2) = 15 paths:
    - 4 splits = calibracao (rodar walk-forward inteiro)
    - 2 splits = teste (avaliar equity curve resultante)
    - Purge: remover embargo_days entre splits adjacentes
    - Computar Sharpe do teste (nao do calibracao!)
Retornar: mean_sharpe, std_sharpe, pct_positive, per-path results
```

3. **Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014):

```python
def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_benchmark: float = 0.0,
) -> float:
    """P-value ajustado pelo numero de trials."""
    ...
```

Formula: DSR = P(SR* < SR_observado) onde SR* ~ N(E[SR*], V[SR*])
com E[SR*] ajustado pelo max esperado de `n_trials` variaveis normais.

4. **Dataclass de resultado:**

```python
@dataclass
class ValidationResult:
    config: WalkForwardConfig
    mean_sharpe: float
    std_sharpe: float
    pct_positive: float          # % de paths com Sharpe > 0
    per_path_sharpe: list[float] # Sharpe de cada um dos 15 paths
    deflated_sharpe: float       # Sharpe ajustado por multiplos testes
    p_value: float               # p-value do DSR
    accepted: bool               # pct_positive > 66% AND deflated > baseline
    metadata: dict[str, Any]
```

5. **Testes (~25-30):**
   - Splits gerados corretamente (6 splits, 15 paths)
   - Embargo aplicado entre splits adjacentes
   - Walk-forward roda em cada fold (mock do backtester)
   - Sharpe computado no teste (nao no calibracao)
   - Deflated Sharpe < observed Sharpe para n_trials > 1
   - DSR = observed Sharpe quando n_trials = 1
   - grid_search retorna resultados ordenados por deflated_sharpe
   - accepted=True somente se pct_positive > 66%
   - Edge cases: todos os paths negativos, um unico config

**Criterio de aceite:** `CPCVParameterValidator.validate()` roda walk-forward em 15 paths CPCV e retorna metricas agregadas + DSR. Testes passam sem regressao.

**Revisao obrigatoria:** quant-reviewer (look-ahead bias nos splits, DSR formula)

---

### Sessao 30 — Fixes Estruturais no Walk-Forward
**Tempo estimado:** ~2h
**Arquivos alterados:** `src/backtest/walk_forward.py`
**Testes alterados:** `tests/test_walk_forward.py`

**O que fazer:**

1. **Fix `_compute_log_returns_for_hrp` — `drop_nulls` com 52 tickers:**

Problema atual: `drop_nulls()` global descarta toda a row se um unico ticker
tem dado faltante. Com 52 tickers, isso pode eliminar muitos dias.

```python
# Antes (problematico):
wide = log_ret.pivot(...).sort("date").drop_nulls()

# Depois (robusto):
wide = log_ret.pivot(...).sort("date").fill_null(0.0)
```

Justificativa: log return faltante = 0.0 e equivalente a "sem variacao",
que e conservador (nao infla covariancia). Melhor que perder dias inteiros.

2. **Fix `NaiveModelFactory.predict` — scaling proporcional ao lookback:**

Problema atual: `ret * 10` satura o clamp [0.05, 0.95] quando lookback > 5.
Com lookback=63, retorno acumulado de ~6% → `0.6 * 10 = 6.0` → clampado em 0.95.
Todos os tickers ficam em 0.95 ou 0.05 (binario), eliminando nuance do sinal.

```python
# Antes (problematico):
conf = max(0.05, min(0.95, 0.5 + ret * 10))

# Depois (escalado):
scaling = 50.0 / max(self.lookback, 1)
conf = max(0.05, min(0.95, 0.5 + ret * scaling))
```

Com lookback=5: scaling=10.0 (backward compat)
Com lookback=63: scaling=0.79 (retorno de 6% → conf=0.55, nuancado)
Com lookback=126: scaling=0.40 (retorno de 12% → conf=0.55, nuancado)

3. **Testes novos (~8-10):**
   - `fill_null` nao descarta rows com dados parciais
   - Scaling produz valores no range [0.05, 0.95] para lookback=63
   - Backward compat: lookback=5 produz mesmos resultados que antes
   - Edge case: lookback=1

**Criterio de aceite:** testes existentes continuam passando; novos testes validam os fixes.

---

### Sessao 31 — HRP Ward Linkage + Ledoit-Wolf Shrinkage
**Tempo estimado:** ~3h
**Arquivos alterados:** `src/portfolio/hrp.py`, `src/backtest/walk_forward.py`
**Testes alterados:** `tests/test_hrp.py`, `tests/test_walk_forward.py`

**O que fazer:**

1. **HRP: Ledoit-Wolf Shrinkage na covariancia:**

Adicionar opcao `shrinkage: bool = False` ao `HRPConfig`.
Quando ativo, substituir sample covariance por Ledoit-Wolf:

```python
from sklearn.covariance import LedoitWolf

def _compute_covariance(self, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if self.config.shrinkage:
        lw = LedoitWolf().fit(returns)
        cov = lw.covariance_
        # Corr from shrunk cov
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)
    else:
        # Existing logic (backward compat)
        ...
    return cov, corr
```

Impacto esperado: pesos mais estaveis → menor turnover → menores custos.
Risco de overfitting: zero (shrinkage e estimador, nao parametro).

2. **HRP: testar ward linkage como default candidato:**

Nao mudar o default (`single`), mas preparar para A/B test via CPCV-OOS.
Ward resolve o chaining problem (1 ativo isolado em cluster degenerado com
`single` linkage e 52 ativos).

O `linkage_method` ja e configuravel no `HRPConfig` — nao precisa mudar codigo,
apenas testar via `HRPConfig(linkage_method="ward")`.

3. **Dependencia:** `sklearn` (scikit-learn) — verificar se ja esta no
   pyproject.toml. Se nao, adicionar `scikit-learn>=1.3`.

4. **Testes novos (~10-12):**
   - `shrinkage=True` produz cov matrix valida (positiva semi-definida)
   - Pesos com shrinkage somam 1.0
   - Shrinkage nao altera resultado quando n_obs >> n_assets (convergencia)
   - Ward linkage produz clusters mais balanceados que single (medir dispersao)
   - Backward compat: `shrinkage=False` produz resultados identicos

**Criterio de aceite:** HRP aceita `shrinkage=True`; ward linkage funciona sem erro; testes passam.

**Revisao obrigatoria:** quant-reviewer (covariancia shrunk valida, ward vs single)

---

### Sessao 32 — Volatility Targeting
**Tempo estimado:** ~3-4h
**Arquivos alterados:** `src/backtest/walk_forward.py`
**Testes novos/alterados:** `tests/test_walk_forward.py`

**O que fazer:**

1. **Novo campo em `WalkForwardConfig`:**

```python
@dataclass(frozen=True)
class WalkForwardConfig:
    # ... campos existentes ...
    target_vol: float | None = None   # ex: 0.10 para 10% annualized
    vol_lookback: int = 63            # janela de vol realizada (trimestre)
    max_leverage: float = 1.0         # cap de alavancagem (1.0 = sem leverage)
    min_leverage: float = 0.5         # floor de exposicao (nunca < 50%)
```

Quando `target_vol=None` (default): nenhum vol targeting aplicado (backward compat).
Quando `target_vol=0.10`: exposure_factor = target_vol / realized_vol,
clamped a [min_leverage, max_leverage].

2. **Logica no loop principal (apos rebalance, antes de aplicar retornos):**

```python
if cfg.target_vol is not None and len(returns_port) >= cfg.vol_lookback:
    # Realized vol dos ultimos vol_lookback dias
    recent_rets = returns_port[-cfg.vol_lookback:]
    realized_vol = _std(recent_rets) * math.sqrt(cfg.trading_days_per_year)

    if realized_vol > 0:
        raw_leverage = cfg.target_vol / realized_vol
        leverage = max(cfg.min_leverage, min(cfg.max_leverage, raw_leverage))

        # Escalar holdings pelo leverage
        total = sum(holdings.values())
        cash_frac = 1.0 - leverage
        for t in available_tickers:
            holdings[t] *= leverage
        # Cash implícito (portfolio_value - sum(holdings))
```

3. **Subsume o Volatility Regime Killswitch (seção 3.2B do improvement.md):**

O killswitch de vol e um caso particular do vol targeting com `max_leverage=1.0`
(nunca alavanca) e `min_leverage=0.0` (pode ir 100% cash).

Para o killswitch conservador: `target_vol=0.10, max_leverage=1.0, min_leverage=0.0`
Para vol targeting padrao: `target_vol=0.10, max_leverage=1.0, min_leverage=0.5`

4. **Impacto na equity curve:**

Dias em que holdings < portfolio_value geram cash implícito:
- `portfolio_return = sum(holding_return) / portfolio_value`
- Cash nao rende (simplificacao; rf/252 por dia e negligivel)

5. **Testes novos (~12-15):**
   - `target_vol=None`: backward compat exato
   - `target_vol=0.10` com vol constante: leverage ~1.0 (nenhum efeito)
   - `target_vol=0.10` com vol alta (0.30): leverage cai para 0.33 → clamped em 0.5
   - `target_vol=0.10` com vol baixa (0.05): leverage sobe para 2.0 → clamped em 1.0
   - `max_leverage=1.0` nunca excedido
   - `min_leverage=0.5` nunca excedido
   - Vol lookback insuficiente (< vol_lookback dias): nenhum targeting aplicado
   - Equity curve com vol targeting tem vol mais constante que sem
   - Holdings negativas impossíveis

**Criterio de aceite:** vol targeting funciona com dados sinteticos; backward compat preservado; testes passam.

**Revisao obrigatoria:** quant-reviewer (calculo de vol, look-ahead no leverage, cash handling)

---

### Sessao 33 — Drawdown Killswitch
**Tempo estimado:** ~2-3h
**Arquivos alterados:** `src/backtest/walk_forward.py`
**Testes novos/alterados:** `tests/test_walk_forward.py`

**O que fazer:**

1. **Dataclass de configuracao:**

```python
@dataclass(frozen=True)
class KillswitchConfig:
    max_drawdown_pct: float = -0.15       # trigger: DD piora de -15%
    recovery_threshold_pct: float = -0.05  # re-entry: DD melhora para -5%
    ramp_up_days: int = 21                 # re-entry gradual (21 dias)
```

2. **Novo campo em `WalkForwardConfig`:**

```python
killswitch: KillswitchConfig | None = None  # None = desligado
```

3. **Logica no loop principal:**

```python
# APOS atualizar holdings com retornos diarios:
if cfg.killswitch is not None:
    peak_value = max(peak_value, portfolio_value)
    dd = (portfolio_value / peak_value) - 1.0

    if dd <= cfg.killswitch.max_drawdown_pct and not in_cash:
        # Mover para cash (vender tudo)
        turnover_cost = _apply_costs(effective_weights, {t: 0 for t}, ...)
        portfolio_value -= turnover_cost
        holdings = {t: 0.0 for t in available_tickers}
        in_cash = True
        days_recovering = 0
        logger.warning("KILLSWITCH ON: DD={:.2%}", dd)

    elif in_cash:
        # DD e calculado sobre o contrafactual (mercado), nao sobre cash
        # Usar benchmark como proxy de recuperacao do mercado
        bench_dd = (benchmark_value / bench_peak) - 1.0

        if bench_dd >= cfg.killswitch.recovery_threshold_pct:
            days_recovering += 1
            ramp = min(1.0, days_recovering / cfg.killswitch.ramp_up_days)

            if ramp >= 1.0:
                in_cash = False
                # Proxima iteracao fara rebalance normal
                days_since_rebalance = cfg.rebalance_every
```

NOTA CRITICA: O drawdown recovery deve usar o **benchmark** como proxy,
nao o portfolio (que esta em cash e nao melhora). Isso evita o bug logico
onde o portfolio nunca sai de cash porque seu DD em cash e constante.

4. **Custos de exit/re-entry:**
   - Exit (ir para cash): custo normal de turnover sobre |pesos atuais|
   - Re-entry: custo normal no proximo rebalance (ramp gradual)

5. **Expectativa:** CPCV-OOS vai rejeitar a maioria das configuracoes.
   Mar 2020: SPY caiu -34% em 23 dias e recuperou em ~5 meses.
   O killswitch teria vendido no fundo e perdido a recuperacao.
   Implementar para completude e para demonstrar rigor na validacao.

6. **Testes novos (~10-12):**
   - `killswitch=None`: backward compat exato
   - DD atinge threshold: muda para cash no dia seguinte
   - Em cash: portfolio_value constante (sem retornos)
   - Recovery via benchmark_dd: ramp gradual de 0 a 1
   - Custos de exit aplicados
   - Ramp completo: sai do modo cash, rebalance ocorre
   - Interacao com vol targeting: killswitch tem prioridade
   - Edge case: DD trigger no primeiro dia ativo

**Criterio de aceite:** killswitch funciona com dados sinteticos; backward compat; testes passam.

**Revisao obrigatoria:** quant-reviewer (bug do recovery, custos, interacao com vol targeting)

---

### Sessao 34 — Validacao CPCV-OOS de Todas as Melhorias
**Tempo estimado:** ~3-4h (principalmente compute)
**Arquivos novos:** `src/backtest/run_validation.py`
**Testes novos:** `tests/test_run_validation.py`

**O que fazer:**

1. **Script de validacao que roda grid search via CPCV-OOS:**

```python
def run_improvement_validation(
    ohlcv: pl.DataFrame,
    tickers: list[str],
    benchmark_ticker: str = "SPY",
    output_dir: str = "data/outputs",
) -> dict[str, ValidationResult]: ...
```

2. **Grid de configuracoes a testar:**

```python
configs = {
    # Baseline
    "baseline": WalkForwardConfig(rebalance_every=5, ...),  # config atual

    # Tier 1 — Preferencia de risco (nao precisa de CPCV, mas rodar para baseline)
    "vol_target_10": WalkForwardConfig(..., target_vol=0.10),
    "vol_target_08": WalkForwardConfig(..., target_vol=0.08),
    "vol_target_12": WalkForwardConfig(..., target_vol=0.12),

    # Tier 1 — Mudancas estruturais
    "ward_linkage": WalkForwardConfig(...)  + HRPConfig(linkage_method="ward"),
    "shrinkage": WalkForwardConfig(...)     + HRPConfig(shrinkage=True),
    "ward_shrinkage": WalkForwardConfig(...) + HRPConfig(linkage_method="ward", shrinkage=True),

    # Tier 1 — NaiveModelFactory lookback
    "momentum_21d":  config + NaiveModelFactory(lookback=21),
    "momentum_63d":  config + NaiveModelFactory(lookback=63),
    "momentum_126d": config + NaiveModelFactory(lookback=126),

    # Tier 2 — Rebalance frequency
    "rebalance_10d": WalkForwardConfig(rebalance_every=10, ...),
    "rebalance_21d": WalkForwardConfig(rebalance_every=21, ...),

    # Tier 2 — HRP confidence tilt
    "no_tilt":     config + HRPConfig(confidence_tilt_cap=0.0),
    "tilt_010":    config + HRPConfig(confidence_tilt_cap=0.10),
    "tilt_030":    config + HRPConfig(confidence_tilt_cap=0.30),

    # Tier 2 — Lookback HRP covariance
    "lookback_252":  WalkForwardConfig(lookback_days=252, ...),
    "lookback_756":  WalkForwardConfig(lookback_days=756, ...),

    # Tier 2 — Killswitch (esperar rejeicao)
    "killswitch_15":  config + KillswitchConfig(max_drawdown_pct=-0.15),
    "killswitch_20":  config + KillswitchConfig(max_drawdown_pct=-0.20),
    "killswitch_25":  config + KillswitchConfig(max_drawdown_pct=-0.25),

    # Combinacoes das melhores Tier 1
    "best_combo": "determinado apos Tier 1 resultados",
}
```

3. **Output da validacao:**

```
data/outputs/
  validation_results.json     — metricas por config
  validation_summary.md       — tabela legivel com ranking
  validation_per_path.parquet — Sharpe por path por config (para analise)
```

4. **Tabela de resultados (validation_summary.md):**

```markdown
| Config | Mean Sharpe | Std | Pct+ | DSR | p-value | Accepted |
|--------|------------|-----|------|-----|---------|----------|
| baseline | 0.537 | ... | ... | ... | ... | (ref) |
| vol_target_10 | ... | ... | ... | ... | ... | ... |
| ward_shrinkage | ... | ... | ... | ... | ... | ... |
```

5. **Makefile targets:**

```makefile
validate:
    poetry run python -m src.backtest.run_validation

validate-fast:
    poetry run python -m src.backtest.run_validation --subset tier1
```

6. **Testes (~10):**
   - Grid search roda com dados sinteticos e NaiveModelFactory
   - Resultados ordenados por deflated_sharpe
   - Output files gerados corretamente
   - validation_summary.md legivel

**Criterio de aceite:** grid search completo roda sem erro; resultados tabulados; DSR reportado.

---

### Sessao 35 — Aplicar Melhorias Aceitas + Benchmark Final
**Tempo estimado:** ~2-3h
**Arquivos alterados:** `src/backtest/run_benchmark.py`, `src/dashboard/app.py`
**Testes alterados:** conforme necessario

**O que fazer:**

1. **Aplicar SOMENTE as configs aceitas pelo CPCV-OOS:**
   - Atualizar defaults em `run_benchmark.py` com os parametros validados
   - Documentar cada mudanca com referencia ao ValidationResult

2. **Re-rodar benchmark completo com configs finais:**
   ```bash
   make benchmark-fast  # validar pipeline
   make benchmark       # run final com PatchTST (se disponivel)
   ```

3. **Atualizar dashboard aba Benchmark:**
   - Adicionar comparacao antes/depois se relevante
   - Mostrar configs aceitas/rejeitadas em um expander

4. **Atualizar documentacao:**
   - `docs/benchmark_improvement.md` → marcar itens aceitos/rejeitados com resultados
   - CLAUDE.md → adicionar sessoes 29-35 ao historico
   - README.md → atualizar metricas se melhoraram significativamente

5. **Gerar relatorio final:**
   - Tabela de melhorias tentadas vs aceitas
   - Deflated Sharpe antes e depois
   - Metricas finais: Sharpe, CAGR, MaxDD, Alpha, Beta, Sortino

**Criterio de aceite:** benchmark final roda com configs validadas; documentacao atualizada; zero mudancas sem validacao CPCV-OOS.

---

## Ordem de Execucao e Dependencias

```
Sessao 29 (CPCV-OOS)           ──────────────────────────────┐
                                                              │
Sessao 30 (Fixes walk-forward) ─┐                            │
                                 ├─► Sessao 32 (Vol Target)   │
Sessao 31 (Ward + Shrinkage)   ─┘                            │
                                     │                        │
                                     ├─► Sessao 33 (Killswitch)
                                     │         │
                                     │         v
                                     └──► Sessao 34 (Validacao CPCV-OOS) ──► Sessao 35 (Aplicar)
```

Sessoes 29, 30 e 31 sao independentes entre si — podem ser feitas em paralelo.
Sessao 32 depende de 30 (fix do walk-forward).
Sessao 33 depende de 32 (interacao killswitch + vol targeting).
Sessao 34 depende de 29 (CPCV-OOS) + 31 + 32 + 33 (tudo pronto para validar).
Sessao 35 depende de 34 (resultados da validacao).

---

## Bugs e Limitacoes Conhecidas no Codigo Atual

Estes devem ser corrigidos ANTES de rodar validacao (Sessao 30):

| # | Arquivo | Problema | Fix |
|---|---------|----------|-----|
| 1 | `walk_forward.py:303` | `drop_nulls()` global descarta rows se 1 ticker falta | `fill_null(0.0)` |
| 2 | `walk_forward.py:127` | `ret * 10` satura com lookback > 5 | `ret * (50/lookback)` |
| 3 | HRP single linkage | Chaining problem com 52 ativos: clusters degenerados | Testar ward |

---

## Metricas de Avaliacao

Para cada configuracao testada, reportar:

| Metrica | Descricao |
|---------|-----------|
| `mean_sharpe_cpcv` | Sharpe medio nos 15 paths CPCV |
| `std_sharpe_cpcv` | Desvio padrao do Sharpe nos paths |
| `pct_positive_cpcv` | % de paths com Sharpe > 0 |
| `deflated_sharpe` | Sharpe ajustado por multiplos testes (DSR) |
| `p_value` | Significancia estatistica vs baseline |
| `mean_max_dd` | MaxDD medio nos paths |
| `mean_cagr` | CAGR medio nos paths |
| `mean_turnover` | Turnover medio anualizado nos paths |
| `accepted` | pct_positive > 66% AND deflated > baseline |

---

## Estimativas de Tempo

| Sessao | Descricao | Tempo dev | Tempo compute |
|--------|-----------|-----------|---------------|
| 29 | CPCV-OOS + Deflated Sharpe | 4-6h | — |
| 30 | Fixes walk-forward | 2h | — |
| 31 | Ward + Ledoit-Wolf | 3h | — |
| 32 | Volatility Targeting | 3-4h | — |
| 33 | Drawdown Killswitch | 2-3h | — |
| 34 | Validacao grid search | 3-4h | 2-4h (15 paths x ~20 configs) |
| 35 | Aplicar + benchmark final | 2-3h | 1-2h (run final) |
| **Total** | | **~19-25h dev** | **~3-6h compute** |

---

## Checklist de Revisao por Sessao

Cada sessao deve passar por:

- [ ] Testes unitarios passando (pytest)
- [ ] Suite completa sem regressoes
- [ ] quant-reviewer em sessoes com logica financeira (29, 30, 31, 32, 33)
- [ ] Zero look-ahead bias (teste dedicado)
- [ ] Backward compatibility preservada (defaults inalterados)
- [ ] Logging com loguru (sem print)
- [ ] Type hints em todos os metodos
- [ ] Docstrings Google Style em funcoes publicas

---

## Referencias

- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 10-12 (CPCV, backtest overfitting).
- Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out-of-Sample." *Journal of Portfolio Management*.
- Raffinot, T. (2017). "Hierarchical Clustering-Based Asset Allocation." *Journal of Portfolio Management*.
- Jegadeesh, N., & Titman, S. (1993). "Returns to Buying Winners and Selling Losers." *Journal of Finance*.
- Ledoit, O., & Wolf, M. (2004). "A well-conditioned estimator for large-dimensional covariance matrices." *Journal of Multivariate Analysis*.
- Sortino, F. A., & van der Meer, R. (1991). "Downside Risk." *Journal of Portfolio Management*.
