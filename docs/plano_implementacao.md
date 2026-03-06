# 🏦 Titanium Alpha — Plano de Implementação com Claude Code + Agentes

> Mapeamento completo de ferramentas, subagentes e fluxos para construir o projeto em 8 semanas.

---

## 🧠 Filosofia Central: Como Usar Claude Code no Projeto

**Regra de Ouro:** Claude Code não é um autocomplete. É um **desenvolvedor sênior acompanhando cada decisão**.

### Estrutura de Trabalho por Sessão
```
1. Planejar → Use o Plan Mode antes de qualquer tarefa complexa
2. Codificar → Deixe o Claude Code operar de forma autônoma
3. Revisar → Subagente especializado revisa o output
4. Compactar → /compact antes do contexto lotar
```

### CLAUDE.md (criar na raiz do projeto no Dia 1)
```markdown
# Titanium Alpha — Regras do Projeto

## Stack
- Python 3.10+, Polars, NeuralForecast, LangGraph, Streamlit
- PostgreSQL + ChromaDB via Docker
- Poetry para dependências

## Padrões de Código
- Type hints obrigatórios em todos os métodos
- Docstrings no formato Google Style
- Módulos independentes em /src/{módulo}/
- Testes em /tests/ com pytest
- Nenhum notebook em produção — só em /notebooks/

## Estrutura de Pastas
src/
├── data/          # ingestão e pipelines
├── models/        # PatchTST e features
├── agents/        # LangGraph agents
├── backtest/      # CPCV e simulações
├── portfolio/     # HRP e alocação
├── dashboard/     # Streamlit UI
└── utils/         # helpers compartilhados

## Convenções
- Logs estruturados com loguru
- Erros nunca silenciosos — sempre raise com contexto
- Commits atômicos: feat/fix/refactor/test
```

---

## 🤖 Arquitetura de Subagentes (.claude/agents/)

Crie estes arquivos **antes de começar qualquer fase**. Eles serão reutilizados ao longo das 8 semanas.

---

### `architect.md`
```yaml
---
name: architect
description: Revisa decisões de design e arquitetura antes de implementar. Usa quando for criar nova classe, módulo ou integração.
tools: Read, Glob, Grep
model: claude-opus-4-6
---
Você é um engenheiro sênior de sistemas quant. Antes de qualquer implementação:
1. Leia os módulos existentes para evitar duplicação
2. Proponha a interface pública (inputs/outputs) antes do código
3. Aponte acoplamentos desnecessários
4. Valide se a estrutura de pastas faz sentido
Responda sempre com: [DESIGN], [RISCOS], [ALTERNATIVAS]
```

---

### `quant-reviewer.md`
```yaml
---
name: quant-reviewer
description: Revisa código quantitativo (modelos, backtests, features). Chame após implementar qualquer lógica matemática ou financeira.
tools: Read, Bash, Grep
model: claude-opus-4-6
---
Você é um quantitative researcher com foco em integridade estatística. Revise:
1. Look-ahead bias (dados do futuro vazando para treino)
2. Overfitting (hiperparâmetros demais, dados de menos)
3. Implementação correta de métricas (Sharpe anualizado? qual rf?)
4. Edge cases: NaN, gaps de pregão, splits de ações
Seja cético. Prefira falsos negativos a aprovar código bugado.
```

---

### `security-data.md`
```yaml
---
name: security-data
description: Valida pipelines de dados — schema, tipos, missing values, inconsistências. Use após qualquer script de ingestão.
tools: Read, Bash, Glob
model: claude-sonnet-4-6
---
Você valida a integridade de dados financeiros. Verifique:
1. Schema correto (tipos, nomes de colunas)
2. Missing values em datas de pregão
3. Preços negativos ou zero
4. Duplicatas de timestamp
5. Consistência entre fontes diferentes
Gere um relatório com: [OK], [WARNING], [ERROR] por check.
```

---

### `test-writer.md`
```yaml
---
name: test-writer
description: Escreve testes unitários e de integração para código novo. Use após implementar qualquer função ou classe.
tools: Read, Write, Bash
model: claude-sonnet-4-6
---
Você escreve testes pytest para código financeiro. Para cada função:
1. Happy path com dados reais de mercado
2. Edge cases: série vazia, NaN, um único ponto
3. Testes de contrato (inputs/outputs corretos)
4. Mock de APIs externas (Yahoo, OpenAI)
Coverage mínimo: 80%. Use fixtures compartilhadas em conftest.py.
```

---

### `docs-writer.md`
```yaml
---
name: docs-writer
description: Gera e atualiza documentação técnica (README, docstrings, comentários). Use ao finalizar cada módulo.
tools: Read, Write, Glob
model: claude-sonnet-4-6
---
Você documenta sistemas quant para portfólio profissional. Escreva:
1. Docstrings Google Style em cada método público
2. README de módulo com exemplo de uso
3. Diagrama de fluxo em Mermaid quando relevante
4. Explicação da matemática por trás (para recrutadores entenderem)
Tom: técnico mas acessível. O leitor pode não ser quant.
```

---

## 📅 Semana a Semana — Comandos e Fluxos

---

### FASE 1 — Semanas 1-2: Infraestrutura e Dados

#### Semana 1: Setup e Pipeline de Dados

**Sessão 1 — Estrutura inicial (2-3h)**
```bash
# No terminal, dentro do projeto:
claude "Inicialize a estrutura do projeto Titanium Alpha usando Poetry.
Crie as pastas: src/{data,models,agents,backtest,portfolio,dashboard,utils},
tests/, notebooks/, docker/.
Gere o pyproject.toml com as dependências:
polars, sqlalchemy, psycopg2-binary, chromadb, sentence-transformers,
langchain, langgraph, neuralforecast, vectorbt, streamlit, loguru, pytest.
Crie também um Makefile com comandos: make setup, make ingest, make test, make run."
```

**Sessão 2 — Docker e Bancos (1-2h)**
```bash
claude "Use o architect subagent para revisar antes de implementar.
Depois crie o docker-compose.yml com:
- PostgreSQL 15 (porta 5432, volume persistente)
- ChromaDB (porta 8000)
- Variáveis via .env (não hardcode credenciais)
Crie também src/utils/db.py com connection pooling usando SQLAlchemy."
```

**Sessão 3 — Ingestão de Dados (3-4h)**
```bash
claude "Crie src/data/ingestion.py com a classe MarketDataIngester usando Polars.
Deve baixar OHLCV dos últimos 5 anos para [SPY, NVDA, AAPL, QQQ] via yfinance,
validar com o security-data subagent, e salvar em PostgreSQL.
Depois chame o test-writer subagent para escrever os testes."
```

**Sessão 4 — Ingestão de Notícias (2h)**
```bash
claude "Crie src/data/news_ingestion.py para buscar notícias via NewsAPI ou RSS (FT, Reuters).
Parse com BeautifulSoup, extraia: título, data, fonte, texto resumido.
Salve raw no PostgreSQL. Use o security-data subagent para validar."
```

**Checkpoint Semana 1:**
```bash
# Rode no terminal para validar:
make ingest
# Esperado: banco populado, logs sem ERROR
```

---

#### Semana 2: Motor Quantitativo (PatchTST)

**Sessão 5 — Feature Engineering (3h)**
```bash
claude "Crie src/models/features.py.
Implemente as funções com Polars (não Pandas):
- rsi(series, period=14)
- bollinger_bands(series, period=20, std=2)
- realized_volatility(series, window=21)
- volume_profile(ohlcv_df)
Chame o quant-reviewer subagent para revisar look-ahead bias em cada feature."
```

**Sessão 6 — PatchTST com NeuralForecast (4h)**
```bash
claude "Crie src/models/patchtst_model.py usando NeuralForecast.
A classe TitaniumForecaster deve:
1. Receber DataFrame Polars com OHLCV + features
2. Treinar PatchTST para prever retorno dos próximos 5 dias
3. Retornar probabilidade de alta/baixa por ativo
4. Salvar modelo treinado em /models/checkpoints/
Parâmetros iniciais: input_size=60, h=5, batch_size=32.
Chame quant-reviewer para validar a ausência de data leakage."
```

**Sessão 7 — Pipeline de Previsões (2h)**
```bash
claude "Crie src/models/predict.py que:
1. Carrega dados do PostgreSQL via Polars
2. Aplica features
3. Roda o TitaniumForecaster
4. Salva predictions.parquet em /data/outputs/
5. Loga métricas: MAE, RMSE por ativo
Adicione ao Makefile: make predict"
```

---

### FASE 2 — Semanas 3-4: Camada Agêntica

#### Semana 3: Multi-Agente com LangGraph

**Sessão 8 — Design dos Agentes (1h — só planejamento)**
```bash
# Use PLAN MODE antes de escrever código:
claude --plan "Vou criar um sistema multi-agente com LangGraph para análise financeira.
Preciso de 4 agentes: TechnicalAnalyst, FundamentalistAnalyst, BearAgent, PortfolioManager.
Como estruturar o grafo? Qual o estado compartilhado? Como evitar loops?"
# Revise o plano antes de aprovar
```

**Sessão 9 — State e Personas (3h)**
```bash
claude "Crie src/agents/state.py com o InvestmentState (TypedDict para LangGraph):
- ticker: str
- predictions: dict (output do PatchTST)
- technical_report: str
- fundamental_report: str
- bear_critique: str
- final_decision: dict (action, confidence, reasoning, weights)
- sources_cited: list[str]

Depois crie src/agents/personas.py com os system prompts detalhados para cada agente.
O BearAgent deve ser explicitamente instruído a encontrar falhas, questionar premissas
e citar riscos de cauda. Use o docs-writer subagent para documentar cada persona."
```

**Sessão 10 — Grafo LangGraph (4h)**
```bash
claude "Crie src/agents/graph.py implementando o fluxo LangGraph:
TechnicalAnalyst → FundamentalistAnalyst → BearAgent → PortfolioManager
Use a Anthropic API (claude-sonnet-4-6) para cada nó.
O PortfolioManager recebe todos os relatórios e decide: BUY/SELL/HOLD com peso sugerido.
Adicione logging estruturado do 'debate' para exibir no dashboard.
Chame o architect subagent para revisar o design do grafo antes de implementar."
```

---

#### Semana 4: RAG Financeiro (Memória dos Agentes)

**Sessão 11 — Embeddings e ChromaDB (3h)**
```bash
claude "Crie src/agents/rag.py com a classe FinancialRAG:
1. Processe notícias do PostgreSQL com sentence-transformers (all-MiniLM-L6-v2)
2. Armazene vetores no ChromaDB com metadata: ticker, data, fonte, categoria
3. Implemente retrieve(ticker, query, top_k=5) que retorna chunks relevantes
4. Inclua um mecanismo de reranking simples por data (notícias recentes primeiro)
Chame security-data subagent para validar que não há dados futuros nos embeddings."
```

**Sessão 12 — Integração RAG + Agentes (2h)**
```bash
claude "Modifique src/agents/personas.py para que o FundamentalistAnalyst:
1. Chame FinancialRAG.retrieve() antes de gerar seu relatório
2. Inclua as fontes no relatório: 'Baseado na notícia X de DD/MM/YYYY...'
3. Nunca faça afirmações sem citar fonte do RAG ou do predictions.parquet
Teste o pipeline completo com NVDA. Chame quant-reviewer para validar ausência de alucinação."
```

---

### FASE 3 — Semanas 5-6: Validação e Portfolio

#### Semana 5: Backtesting com CPCV

**Sessão 13 — CPCV Implementation (4h — crítico)**
```bash
claude "Crie src/backtest/cpcv.py implementando Combinatorial Purged Cross-Validation.
Parâmetros: n_splits=6, embargo_days=10 (para evitar overlap de features).
A classe CPCVBacktester deve:
1. Gerar todas as combinações de folds de treino/teste sem sobreposição temporal
2. Aplicar purging (remover amostras com label overlap)
3. Aplicar embargo (gap entre treino e teste)
4. Retornar distribuição de Sharpe Ratios por fold
ANTES de implementar, chame architect subagent.
DEPOIS de implementar, chame quant-reviewer — este é o ponto mais crítico do projeto."
```

**Sessão 14 — Simulação de Custos e Relatório (3h)**
```bash
claude "Adicione custos reais ao CPCVBacktester:
- Slippage: 0.05% por operação (bid-ask spread)
- Corretagem: 0.10% por operação
- Market impact: proporcional ao volume
Crie src/backtest/report.py que gera PDF automático com:
- Curva de equity por fold
- Distribuição de Sharpe (violinplot)
- Max Drawdown e duração
- Tabela de métricas por ativo
Use matplotlib/seaborn para os gráficos. Chame docs-writer para documentar as métricas."
```

---

#### Semana 6: HRP e Alocação de Portfolio

**Sessão 15 — Hierarchical Risk Parity (3h)**
```bash
claude "Crie src/portfolio/hrp.py implementando HRP (Lopez de Prado, 2016).
A classe HRPOptimizer deve:
1. Receber matriz de covariância + 'confiança' dos agentes (0-1) por ativo
2. Calcular correlações via scipy
3. Aplicar hierarchical clustering (ward linkage)
4. Quasi-diagonalização da matriz de covariância
5. Recursive bisection para pesos finais
6. Ajustar pesos pela confiança dos agentes (boost/penalidade ±20%)
O output: dict {ticker: weight} normalizado para somar 1.
Chame quant-reviewer para validar a implementação matemática."
```

**Sessão 16 — Pipeline Final de Decisão (2h)**
```bash
claude "Crie src/portfolio/decision_engine.py que orquestra tudo:
1. Carrega predictions.parquet
2. Roda o grafo LangGraph para cada ativo
3. Alimenta confiança dos agentes no HRPOptimizer
4. Retorna DecisionOutput: {ticker: {action, weight, reasoning, sources}}
5. Salva em decisions.json com timestamp
Adicione ao Makefile: make decide"
```

---

### FASE 4 — Semanas 7-8: Dashboard e Deploy

#### Semana 7: Dashboard Streamlit

**Sessão 17 — Estrutura do Dashboard (2h)**
```bash
# Leia o skill de frontend antes:
claude "Crie src/dashboard/app.py em Streamlit com 3 abas:
1. Performance: curva P&L vs SPY (benchmark), métricas CPCV, Sharpe por fold
2. Sala de Guerra: exibe o debate dos agentes em tempo real (streaming)
   - Cada agente com cor/ícone diferente
   - Decisão final em destaque
3. Microestrutura: gráfico PatchTST com intervalo de confiança por ativo
Use Plotly para todos os gráficos (interativo). Design dark theme profissional."
```

**Sessão 18 — Streaming dos Agentes (2h)**
```bash
claude "Modifique o grafo LangGraph para suportar streaming via LangChain callbacks.
No dashboard, a aba 'Sala de Guerra' deve mostrar os agentes 'digitando' em tempo real
usando st.write_stream(). Cada mensagem aparece com o nome do agente e timestamp.
Teste com st.rerun() para simular o fluxo ao vivo."
```

---

#### Semana 8: Documentação e Deploy

**Sessão 19 — README e Documentação (2h)**
```bash
claude "Use o docs-writer subagent para gerar:
1. README.md principal com: visão geral, arquitetura (diagrama Mermaid), 
   quick start, resultados do backtest, stack tecnológico
2. ARCHITECTURE.md com diagrama completo do fluxo de dados
3. Docstrings em todos os arquivos src/ que ainda não têm
O README deve ser escrito em inglês (para GitHub internacional) e impressionar recrutadores
que não são quants — explique o valor de negócio, não só a matemática."
```

**Sessão 20 — CI/CD e Organização Final (2h)**
```bash
claude "Configure GitHub Actions em .github/workflows/:
1. ci.yml: roda pytest em cada PR (usa dados mock, não APIs reais)
2. lint.yml: ruff + mypy em cada push
Crie também um Dockerfile de produção para o dashboard Streamlit.
Organize o repositório: delete arquivos temporários, verifique que nenhuma API key
está no código, adicione .gitignore completo."
```

---

## 🔄 Fluxo de Sessão Recomendado (Diário)

```
Início de cada sessão:
1. claude "Resuma o estado atual do projeto lendo os últimos commits"
2. Defina 1 objetivo claro para a sessão
3. Use plan mode para tarefas > 3 arquivos

Durante a sessão:
- Chame subagentes após cada módulo novo
- /compact quando contexto estiver pesado (≈ 100k tokens)
- Commits frequentes: "feat: implement PatchTST training loop"

Final da sessão:
- claude "Gere um resumo do que foi feito hoje e 3 próximos passos"
- Atualize CLAUDE.md se algo mudou na arquitetura
```

---

## ⚠️ Armadilhas Comuns — O Que Evitar

| Armadilha | Solução |
|-----------|---------|
| Sessão única gigante sem /compact | Compactar a cada 2-3 tarefas |
| Implementar sem plan mode | Sempre planejar tarefas com >3 arquivos |
| Confiar no backtest sem quant-reviewer | Nunca pular a revisão de look-ahead bias |
| MCP servers demais | Máximo 3-4 MCPs ativos simultâneos |
| CLAUDE.md desatualizado | Atualizar sempre que mudar a arquitetura |
| Subagentes genéricos | Cada subagente tem **uma** responsabilidade clara |

---

## 🚀 Comandos de Referência Rápida

```bash
# Delegar tarefa específica a um subagente:
claude "Use the quant-reviewer subagent to review src/models/patchtst_model.py"

# Paralelismo (múltiplas análises simultâneas):
claude -p "Analyze SPY predictions" &
claude -p "Analyze NVDA predictions" &
wait

# Headless para automação diária:
claude -p "Run the daily decision pipeline and save to decisions/$(date +%Y%m%d).json" \
  --allowedTools "Read,Write,Bash"

# Plan mode explícito:
claude --plan "Refactor the agents module to support async execution"
```

---

## 📊 Critério de Sucesso por Fase

| Fase | Entregável | Validação |
|------|-----------|-----------|
| 1 | `make ingest` popula banco sem erros | security-data subagent ✅ |
| 2 | `predictions.parquet` com prob. por ativo | quant-reviewer valida look-ahead ✅ |
| 3 | Log de debate entre agentes com fontes citadas | Nenhuma alucinação no FundamentalistAnalyst |
| 4 | `decisions.json` com pesos HRP | quant-reviewer valida CPCV ✅ |
| 5 | Dashboard rodando `make run` | 3 abas funcionais com dados reais |
| 6 | Repositório público com README em inglês | Diagrama Mermaid + vídeo demo |
