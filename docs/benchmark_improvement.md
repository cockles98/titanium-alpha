# Benchmark Improvement Plan

> Análise de possibilidades de melhoria para o walk-forward benchmark.
> Baseline: Sharpe=0.537, CAGR=12.74%, MaxDD=-33.86%, Alpha=1.99%

---

## 1. Metodologia Anti-Overfitting

**Problema**: ajustar parâmetros no OOS e selecionar os que performaram melhor é overfitting clássico (data snooping). Qualquer melhoria precisa ser validada com rigor estatístico.

**Solução proposta**: CPCV no período OOS.

### 1.1 Design do CPCV-OOS

```
OOS (~15 anos, 2011-2026)
┌─────────────────────────────────────────────────────┐
│ Split 1 │ Split 2 │ Split 3 │ Split 4 │ Split 5 │ S6│  ← 6 splits temporais
│  ~2.5yr │  ~2.5yr │  ~2.5yr │  ~2.5yr │  ~2.5yr │   │
└─────────────────────────────────────────────────────┘
Para cada combinação C(6,2)=15 paths:
  - 4 splits = treino do modelo/calibração dos parâmetros
  - 2 splits = teste out-of-fold
  - Purge + embargo entre splits adjacentes
```

**Implementação**: criar `CPCVParameterValidator` que:
1. Recebe um dict de parâmetros candidatos
2. Roda o walk-forward em cada um dos 15 paths CPCV
3. Computa Sharpe por path
4. Reporta: mean_sharpe, std_sharpe, pct_positive, p-value vs baseline
5. Um parâmetro só é aceito se `pct_positive > 60%` AND `mean_sharpe > baseline`

### 1.2 Multiple Testing Correction

Com N configurações testadas, a chance de encontrar uma que "funciona" por acaso cresce. Aplicar:
- **Bonferroni**: dividir α por N (conservador)
- **Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014): ajusta o Sharpe pelo número de trials
- Reportar o Sharpe ajustado junto com o bruto

---

## 2. Parâmetros Tunáveis

### 2.1 Walk-Forward Config

| Parâmetro | Atual | Candidatos | Hipótese |
|---|---|---|---|
| `rebalance_every` | 5 (semanal) | 10, 21 | Menos rebalanceos = menos custos, mas mais drift |
| `retrain_every` | 126 (semestral) | 63, 252 | Trimestral pode adaptar mais rápido; anual pode ser mais estável |
| `lookback_days` | 504 (~2 anos) | 252, 756 | Janela mais curta = mais responsivo; mais longa = mais estável |
| `min_rebalance_delta` | 0.02 | 0.01, 0.05, 0.10 | Threshold maior = menos trades ruidosos |
| `initial_capital` | 1M | — | Não afeta métricas relativas |

### 2.2 Transaction Costs

| Parâmetro | Atual | Candidatos | Nota |
|---|---|---|---|
| `slippage_bps` | 5.0 | 3.0, 8.0 | Depende do universo (large caps = menor spread) |
| `commission_bps` | 10.0 | 5.0, 15.0 | Interativo broker ~$0.005/share ≈ 3-5 bps |
| `market_impact_bps` | 0.0 | 1.0, 3.0 | Importante para posições grandes |

**Nota**: custos não devem ser "otimizados" — devem refletir a realidade. A análise aqui é de **sensibilidade**, não de tuning.

### 2.3 HRP Config

| Parâmetro | Atual | Candidatos | Hipótese |
|---|---|---|---|
| `linkage_method` | single | ward, complete | Ward cria clusters mais balanceados; single segue LdP original |
| `correlation_method` | pearson | spearman | Spearman é mais robusto a outliers (crashes) |
| `confidence_tilt_cap` | 0.20 | 0.10, 0.30, 0.40 | Cap maior = mais convicção do modelo; menor = mais puro HRP |
| `max_weight` | min(0.25, 2/n) | min(0.15, 1.5/n) | Cap mais agressivo = diversificação forçada |

### 2.4 NaiveModelFactory

| Parâmetro | Atual | Candidatos | Hipótese |
|---|---|---|---|
| `lookback` | 5 | 10, 21, 63, 126 | Momentum de curto vs médio prazo |
| `scaling_factor` | 10 (hardcoded) | 5, 20 | Sensibilidade do sinal: menor = mais conservador |
| `clamp_range` | [0.05, 0.95] | [0.10, 0.90], [0.20, 0.80] | Range mais estreito = tilt mais suave |

---

## 3. Killswitch (Circuit Breaker)

### 3.1 Motivação

O MaxDD atual é -33.86%. Em produção, um drawdown de -20% ou mais pode ser inaceitável. Um killswitch é um mecanismo de proteção que **reduz ou zera exposição** quando condições adversas são detectadas.

### 3.2 Triggers Propostos

#### A) Max Drawdown Killswitch
```python
@dataclass(frozen=True)
class KillswitchConfig:
    max_drawdown_pct: float = -0.15       # trigger: DD piora de -15%
    recovery_threshold_pct: float = -0.05  # re-entry: DD melhora para -5%
    ramp_up_days: int = 21                 # re-entry gradual (21 dias)
```

**Lógica**:
- Se drawdown do portfolio atinge `max_drawdown_pct` → mover para 100% cash
- Permanecer em cash até drawdown recuperar para `recovery_threshold_pct`
- Re-entry gradual: weight_target * (days_since_recovery / ramp_up_days)

**Trade-off**: protege contra tail risk, mas pode whipsaw em V-recoveries (ex: Mar 2020).

#### B) Volatility Regime Killswitch
```python
vol_window: int = 21                  # janela de vol realizada
vol_threshold_multiplier: float = 2.0 # trigger: vol > 2x média histórica
```

**Lógica**:
- Computar rolling realized vol (21 dias) vs média de longo prazo (252 dias)
- Se vol_curta > multiplier * vol_longa → reduzir exposição proporcionalmente
- Exposure_factor = min(1.0, vol_target / vol_realizada)

**Vantagem**: mais suave que o DD killswitch, reduz gradualmente.

#### C) Correlation Spike Killswitch
```python
corr_window: int = 21
corr_threshold: float = 0.7  # avg pairwise corr > 0.7 → systemic risk
```

**Lógica**: quando correlações médias sobem muito (crises sistêmicas), o benefício de diversificação do HRP desaparece. Reduzir exposição quando avg_pairwise_corr > threshold.

### 3.3 Implementação no Walk-Forward

O killswitch deve ser implementado **dentro do loop principal** do `WalkForwardBacktester.run()`, entre o cálculo de retornos e a atualização de holdings:

```python
# Pseudo-código no loop diário
dd = (portfolio_value / peak_value) - 1.0
if dd <= killswitch.max_drawdown_pct and not in_cash:
    # Move to cash
    holdings = {t: 0.0 for t in available_tickers}
    cash = portfolio_value
    in_cash = True
    logger.warning("KILLSWITCH: DD={:.2%}, moving to cash", dd)

if in_cash and dd >= killswitch.recovery_threshold_pct:
    # Begin ramp-up
    days_recovering += 1
    ramp = min(1.0, days_recovering / killswitch.ramp_up_days)
    # Apply ramp factor to target weights
```

### 3.4 Validação

O killswitch **também deve ser validado via CPCV** — não podemos simplesmente escolher thresholds que funcionam no OOS. Testar pelo menos:
- `max_drawdown_pct`: [-0.10, -0.15, -0.20, -0.25]
- `recovery_threshold_pct`: [-0.03, -0.05, -0.08]
- `ramp_up_days`: [5, 10, 21, 42]
- `killswitch=None` (baseline sem killswitch)

---

## 4. Outras Melhorias Estruturais

### 4.1 Volatility Targeting

**O que é**: escalar o tamanho total da posição para que a volatilidade do portfolio se mantenha constante (ex: 10% ao ano).

```python
target_vol: float = 0.10  # 10% annualized
realized_vol = rolling_std(port_returns, 63) * sqrt(252)
leverage = target_vol / realized_vol  # clamped to [0.5, 1.5]
effective_weights = {t: w * leverage for t, w in weights.items()}
```

**Impacto esperado**: melhora o Sharpe (denominador mais constante), reduz MaxDD em crises (alavancagem reduz quando vol sobe).

**Risco de overfitting**: baixo — target_vol é um parâmetro de preferência de risco, não de alpha.

### 4.2 Covariance Shrinkage (Ledoit-Wolf)

**Problema**: sample covariance com 52 ativos e ~504 obs é noisy (ratio assets/obs ≈ 0.10 — aceitável mas não ótimo).

**Melhoria**: usar Ledoit-Wolf shrinkage, que combina a sample cov com um target estruturado (identidade escalada):
```
Σ_shrunk = (1-δ) * Σ_sample + δ * μ * I
```

**Impacto**: pesos HRP mais estáveis entre rebalanceos → menor turnover → menores custos.

**Implementação**: `sklearn.covariance.LedoitWolf` ou manual em numpy (fórmula analítica).

### 4.3 Momentum Pre-Filter

**O que é**: antes do HRP, filtrar tickers com momentum negativo nos últimos N dias, excluindo-os da alocação.

```python
# Pseudo-código
for ticker in tickers:
    ret_n = (close[-1] / close[-momentum_lookback]) - 1
    if ret_n < momentum_threshold:
        exclude ticker from HRP
```

**Parâmetros candidatos**:
- `momentum_lookback`: 63, 126, 252
- `momentum_threshold`: 0.0, -0.05, -0.10

**Impacto esperado**: evita alocar em ativos em queda livre (catching falling knives).

**Risco**: alta sensibilidade ao lookback → deve ser validado via CPCV.

### 4.4 Sector Constraints

**O que é**: limitar a exposição máxima por setor (ex: max 30% em Technology).

Mapeamento de tickers por setor:
```
Technology: AAPL, MSFT, GOOG, AMZN, META, NVDA, TSLA, AVGO, CRM, AMD (10)
Financials: JPM, BAC, GS, MS, WFC, BLK, AXP, C (8)
Healthcare: UNH, JNJ, LLY, PFE, ABBV, MRK, TMO (7)
Consumer:   PG, KO, PEP, COST, WMT, HD, MCD (7)
Energy:     XOM, CVX, COP, SLB (4)
Industrials: CAT, HON, UPS, BA, GE, RTX (6)
Utilities/REIT: NEE, DUK, AMT, PLD (4)
Comms/Media: DIS, NFLX, CMCSA (3)
Materials:   LIN, APD, NEM (3)
```

**Impacto**: diversificação setorial forçada. O HRP já clusteriza por correlação (que correlaciona com setor), então o impacto marginal pode ser pequeno.

**Risco de overfitting**: baixo — sector caps são constraints de risco, não de alpha.

### 4.5 Universo Dinâmico (Liquidity Filter)

**O que é**: excluir tickers com volume médio abaixo de um threshold (ex: bottom 10% do universo).

**Impacto**: melhora a qualidade da estimativa de custos e reduz slippage real.

**Risco**: baixo — todos os 52 tickers atuais são large caps muito líquidos.

### 4.6 Benchmark-Aware (Tracking Error Budget)

**O que é**: adicionar um constraint de tracking error ao HRP, limitando o desvio em relação ao benchmark SPY.

**Impacto**: reduz o risk de underperformance extrema vs benchmark.

**Trade-off**: reduz alpha potencial (portfolio fica mais parecido com SPY).

---

## 5. Priorização

### Tier 1 — Alto impacto, baixo risco de overfitting
| Melhoria | Impacto esperado | Complexidade |
|---|---|---|
| **Killswitch (DD-based)** | MaxDD -15% a -20% (vs -33.86%) | Média |
| **Volatility Targeting** | Sharpe +0.1 a +0.2, MaxDD melhora | Baixa |
| **Covariance Shrinkage** | Turnover -20%, estabilidade de pesos | Baixa |

### Tier 2 — Impacto moderado, requer validação CPCV
| Melhoria | Impacto esperado | Complexidade |
|---|---|---|
| **Rebalance frequency** (5→10 ou 21) | Custos -30 a -50% | Baixa |
| **HRP linkage** (single→ward) | Pesos mais balanceados | Baixa |
| **Momentum pre-filter** | Evita falling knives | Média |
| **NaiveModelFactory lookback** (5→21 ou 63) | Sinal mais estável | Baixa |

### Tier 3 — Nice-to-have, menor impacto
| Melhoria | Impacto esperado | Complexidade |
|---|---|---|
| **Sector constraints** | Diversificação marginal | Média |
| **Spearman correlation** | Robustez a outliers | Baixa |
| **Liquidity filter** | Marginal (universo já é liquid) | Baixa |
| **Tracking error budget** | Risco relativo controlado | Alta |

---

## 6. Plano de Implementação

### Fase 1: Infraestrutura CPCV-OOS
1. Criar `src/backtest/cpcv_oos.py`: CPCVParameterValidator
2. Integrar com walk-forward backtester (roda walk-forward em cada fold)
3. Implementar Deflated Sharpe Ratio
4. Testes

### Fase 2: Killswitch
1. Criar `KillswitchConfig` em walk_forward.py
2. Implementar lógica no loop principal
3. Validar via CPCV-OOS (grid de thresholds)
4. Testes

### Fase 3: Melhorias Tier 1
1. Volatility targeting (novo parâmetro em WalkForwardConfig)
2. Covariance shrinkage (opção em HRPOptimizer)
3. Validar via CPCV-OOS

### Fase 4: Tuning Tier 2
1. Grid search de parâmetros (rebalance_every, lookback, linkage, etc.)
2. CPCV-OOS para cada candidato
3. Aceitar apenas melhorias com pct_positive > 60%

---

## 7. Métricas de Avaliação

Para cada melhoria, reportar:

| Métrica | Descrição |
|---|---|
| `mean_sharpe_cpcv` | Sharpe médio nos 15 paths CPCV |
| `std_sharpe_cpcv` | Desvio padrão do Sharpe nos paths |
| `pct_positive_cpcv` | % de paths com Sharpe > 0 |
| `mean_sharpe_delta` | Diferença vs baseline |
| `deflated_sharpe` | Sharpe ajustado por múltiplos testes |
| `max_dd_cpcv` | MaxDD médio nos paths |
| `avg_turnover_cpcv` | Turnover médio nos paths |
| `p_value` | Significância estatística vs baseline |

---

## 8. Referências

- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 10-12 (CPCV, backtest overfitting).
- Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out-of-Sample." *Journal of Portfolio Management*.
- Ledoit, O., & Wolf, M. (2004). "A well-conditioned estimator for large-dimensional covariance matrices." *Journal of Multivariate Analysis*.
