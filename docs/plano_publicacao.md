# Plano de Publicação — Titanium Alpha
**Criado:** 2026-04-19  
**Objetivo:** Publicar o projeto como portfólio quant de alto impacto no GitHub  
**Meta de resultado:** Projeto que demonstra autoridade no setor quant através de metodologia rigorosa + sistema multi-agente LLM funcionando ao vivo

---

## Estado Atual (Baseline)

| Componente | Status |
|---|---|
| Walk-forward backtest | ✅ Sharpe=0.710, CAGR=13.33%, MaxDD=-18.46% |
| CPCV-OOS fine-tuning | ✅ 982 configs testadas (sessões 39-40) |
| PatchTST predictions | ✅ Funcionando com cache |
| Dashboard (4 abas) | ✅ Benchmark, Performance, War Room, Microstructure |
| Testes | ✅ 1002 passando |
| Pipeline de agentes LangGraph | ⚠️ Implementado mas nunca validado ponta-a-ponta |
| War Room ao vivo | ⚠️ Implementado mas depende de agentes funcionando |
| ChromaDB / RAG | ⚠️ Pode estar sem dados de notícias |
| Documentação pública | ❌ README incompleto, sem demo visual |
| GitHub público | ❌ Repositório privado |

**Lição da sessão 40:** Parâmetros de concentração HRP (max_weight) não devem ser otimizados via CPCV-OOS — a restrição `2/n` é princípio de risco, não parâmetro livre. Fine-tuning via CPCV-OOS é válido apenas para parâmetros de timing (rebalance_every, target_vol).

---

## Cronograma de Alto Nível

```
Fase 1  │ Agentes: debug + teste end-to-end          │ ~2 sessões
Fase 2  │ Refinamento dos agentes de desenvolvimento  │ ~1 sessão
Fase 3  │ Melhorias de produto                        │ ~2 sessões
Fase 4  │ Preparação para portfólio                   │ ~2 sessões
Fase 5  │ Publicação                                  │ ~0.5 sessão
────────┴───────────────────────────────────────────────────────────
Total estimado: 7-8 sessões de desenvolvimento
```

---

## Fase 1 — Teste e Validação do Pipeline de Agentes
**Objetivo:** War Room mostrando debate completo de todos os tickers com dados reais ao vivo.  
**Por que primeiro:** É o componente mais diferenciado do projeto e atualmente não foi validado ponta-a-ponta. Sem isso, o projeto não tem o seu elemento mais impressionante funcionando.

### 1.1 — Pré-requisitos e verificação de ambiente

**Tarefas:**
- [ ] Verificar se o modelo Gemini configurado existe e está acessível  
  - Código usa `gemini-3.1-flash-lite-preview` (`src/agents/graph.py`)  
  - Verificar nome correto na API do Google AI Studio (pode ser `gemini-2.0-flash-lite` ou similar)  
  - Atualizar `.env` e `graph.py` se necessário
- [ ] Verificar se `predictions.parquet` e `features.parquet` existem em `data/outputs/`  
  - Se não: rodar `poetry run python -m src.models.patchtst_model` ou `make predict`
- [ ] Verificar se a tabela `financial_news` em PostgreSQL tem dados  
  - Se vazia: rodar `make ingest-news` e verificar NEWSAPI_KEY no `.env`
- [ ] Verificar se ChromaDB está populado  
  - Deve ter a coleção `financial_news` com documentos

**Critério:** Todos os pré-requisitos verificados e corrigidos.

### 1.2 — Debug do pipeline de agentes (subconjunto)

**Tarefas:**
- [ ] Rodar debate para 3 tickers: `poetry run python -m src.agents.graph AAPL MSFT NVDA`
- [ ] Verificar se os 4 agentes completam (Technical → Fundamental → Bear → PM)
- [ ] Verificar se `decisions.json` e `debate_history.json` são criados em `data/outputs/`
- [ ] Verificar se a estrutura do `debate_history.json` é compatível com o que o dashboard espera
- [ ] Corrigir quaisquer erros de parsing de output estruturado do LLM (Pydantic validation errors são comuns)
- [ ] Verificar se o fallback funciona quando RAG não tem dados para um ticker

**Critério:** 3 tickers completam o debate sem erros, arquivos JSON gerados com estrutura válida.

### 1.3 — Execução completa (52 tickers)

**Tarefas:**
- [ ] Rodar `make decide` (pipeline completo: predict → debate → HRP → save)
- [ ] Monitorar uso de prompts (52 tickers × 4 agentes = ~208 prompts por run, dentro dos 500/dia)
- [ ] Verificar se o `DecisionEngine` processa corretamente o output dos agentes
- [ ] Verificar se pesos HRP finais somam ≤ 1.0 e nenhum ticker ultrapassa max_weight
- [ ] Registrar tempo total do run (referência para documentação)

**Critério:** 52 tickers com decisões válidas (BUY/HOLD/SELL + pesos), `decisions.json` completo.

### 1.4 — Validação do War Room no dashboard

**Tarefas:**
- [ ] Abrir `http://localhost:8501` e navegar para a aba War Room
- [ ] Testar **Modo Replay**: selecionar um ticker, verificar que as 4 bolhas de debate aparecem sequencialmente com efeito typing
- [ ] Testar **Modo Live**: clicar em "Run Debate" e verificar streaming por nó do LangGraph
- [ ] Verificar cartão final do Portfolio Manager (cor por ação, peso, confiança, dissidência do Bear)
- [ ] Verificar se a aba Performance reflete as decisões atuais (tabela de decisões, donut de pesos)
- [ ] Corrigir quaisquer bugs de rendering encontrados

**Critério:** War Room exibe debate completo com visual polido. Modo Live funciona. Nenhum erro Python no terminal do Streamlit durante a demonstração.

---

## Fase 2 — Refinamento dos Agentes de Desenvolvimento (`.claude/agents`)
**Objetivo:** Agentes de desenvolvimento mais inteligentes sobre este projeto específico, acelerando as próximas fases.  
**Por que agora:** Com os agentes do projeto funcionando, a equipe tem o contexto completo para refinar os agentes de dev. Eles serão usados intensamente nas Fases 3 e 4.

### 2.1 — `quant-reviewer.md`

**Melhorias:**
- [ ] Adicionar regra explícita sobre separação de parâmetros de timing vs concentração (lição da sessão 40)
- [ ] Adicionar verificação: "max_weight está sendo otimizado via grid search?" → REPROVADO com explicação
- [ ] Adicionar verificação de DSR: diferenças < 0.02 entre candidatos CPCV-OOS são ruído
- [ ] Adicionar contexto sobre o backtest-produção gap: qualquer código que misture sinal dos agentes com backtest deve ser sinalizado
- [ ] Adicionar checklist específico para validação de CPCV: purge days, embargo correto, rf geométrico

### 2.2 — `architect.md`

**Melhorias:**
- [ ] Adicionar conhecimento sobre o modelo three-tier (BUY=HRP, HOLD=HRP×conf, SELL=0)
- [ ] Adicionar princípio: "agentes LangGraph não entram no loop de backtest — token cost proíbe"
- [ ] Adicionar constraint: "novos módulos em src/ seguem a estrutura de pastas sagrada do CLAUDE.md"
- [ ] Adicionar aviso sobre imports circulares (src/agents ↔ src/portfolio é um risco real)

### 2.3 — `docs-writer.md`

**Melhorias:**
- [ ] Atualizar com status real do projeto (Fase 8 completa, 982 configs testadas, Sharpe=0.710)
- [ ] Adicionar template específico para documentar resultados de benchmark (formato padronizado)
- [ ] Adicionar instrução: documentação pública no README deve ter screenshots do dashboard
- [ ] Adicionar regra: nunca documentar Sharpe ~2.7 (era bug) — usar apenas os números pós-sessão 37
- [ ] Adicionar template para seção "Metodologia" do README (explica CPCV, DSR, HRP para audiência quant)

### 2.4 — `test-writer.md`

**Melhorias:**
- [ ] Adicionar padrões específicos de mock para LangGraph (como mockar nós do grafo)
- [ ] Adicionar padrões para testar agentes: mockar LLM calls com respostas Pydantic válidas
- [ ] Adicionar regra: testes de agentes NUNCA chamam API real (custo de tokens)
- [ ] Adicionar exemplos de fixtures Polars (o projeto usa Polars, não Pandas)
- [ ] Adicionar teste-padrão para outputs do LLM: verificar que Pydantic validation não quebra com outputs parciais

### 2.5 — `security-data.md`

**Melhorias:**
- [ ] Adicionar verificação específica: yfinance deve usar `yf.Ticker().history()` (não `yf.download()` — bug de thread-safety já corrigido na sessão 37)
- [ ] Adicionar verificação de ChromaDB: coleção `financial_news` deve existir e ter documentos
- [ ] Adicionar verificação de `decisions.json`: campos obrigatórios, pesos somando ≤ 1.0
- [ ] Adicionar alerta: se `financial_news` está vazia, o agente Fundamentalista usa fallback sem RAG

**Critério de conclusão da Fase 2:** Todos os 5 agentes de desenvolvimento atualizados com contexto da sessão 40 e padrões específicos do projeto.

---

## Fase 3 — Melhorias de Produto
**Objetivo:** Polir os componentes existentes para qualidade de portfólio público.

### 3.1 — Dashboard: polish e UX

**Tarefas:**
- [ ] Corrigir warnings de `use_container_width` → substituir por `width='stretch'` / `width='content'` (Streamlit deprecation, deadline 2025-12-31 já passou)
- [ ] Adicionar timestamp "Última atualização" em todas as abas (mostra que dados são ao vivo)
- [ ] Adicionar tratamento de erro explícito quando `decisions.json` não existe (primeira execução)
- [ ] Adicionar botão "Exportar PDF" na aba Benchmark que abre o relatório salvo
- [ ] Melhorar loading states no War Room Live Mode (progress bar por ticker)
- [ ] Verificar responsividade em telas menores (importante para screenshots do portfólio)
- [ ] Adicionar aba ou seção "Sobre o Projeto" no dashboard com link para GitHub e metodologia

### 3.2 — PatchTST: análise de importância de features

**Tarefas:**
- [ ] Criar notebook `notebooks/feature_analysis.ipynb` que analisa quais features mais impactam `prob_up`
- [ ] Verificar se features de volume (OBV, VWAP) contribuem positivamente
- [ ] Avaliar se RSI e Bollinger Bands têm sinal real vs ruído no contexto CPCV-OOS
- [ ] Documentar findings — mesmo que "features X são irrelevantes" é um resultado honesto que impressiona quant reviewers
- [ ] **Não retreinar o modelo** sem CPCV-OOS validação — só analisar o modelo atual

### 3.3 — Ingestão de notícias e RAG

**Tarefas:**
- [ ] Verificar se NEWSAPI_KEY está configurado e ativo no `.env`
- [ ] Rodar `make ingest-news` e verificar quantos artigos foram carregados no ChromaDB
- [ ] Testar recuperação RAG para pelo menos 5 tickers (`src/agents/rag.py`)
- [ ] Verificar se o agente Fundamentalista cita fontes corretamente no debate (campo `sources` do AgentReport)
- [ ] Se NEWSAPI não disponível: documentar degradação graceful e testar que o agente funciona sem notícias

### 3.4 — Testes: cobertura e qualidade

**Tarefas:**
- [ ] Rodar `poetry run pytest --cov=src --cov-report=term-missing` e identificar módulos com < 70% cobertura
- [ ] Adicionar testes para o pipeline de agentes com mocks de LLM (prioritário — maior gap atual)
- [ ] Adicionar teste de integração: `run_agent_debate` com mock LLM para 3 tickers → verifica `decisions.json`
- [ ] Garantir que todos os 1002 testes existentes continuam passando após mudanças das Fases 1-3

**Critério de conclusão da Fase 3:** Dashboard sem warnings Python, War Room com UX polida, cobertura de testes dos agentes > 70%, RAG com dados reais.

---

## Fase 4 — Preparação para Portfólio
**Objetivo:** Transformar o projeto técnico em vitrine profissional para a comunidade quant.

### 4.1 — README profissional (prioridade máxima)

O README é a primeira coisa que qualquer pessoa vê. Deve conter:

**Estrutura proposta:**
```
1. Header: logo/banner + badges (tests, coverage, Python version, license)
2. Tagline: "Multi-agent hedge fund system: PatchTST + LangGraph + HRP"
3. Live Demo GIF: War Room mostrando debate ao vivo (gravar com screen recorder)
4. Performance Results: tabela com Sharpe=0.710, CAGR=13.33%, MaxDD=-18.46% + equity curve chart
5. Architecture: diagrama Mermaid do pipeline completo
6. How It Works: explicação para audiência quant (CPCV-OOS, DSR, HRP, LangGraph debate)
7. Quick Start: docker-compose up → make benchmark → make decide → make run
8. Project Structure: árvore de pastas comentada
9. Methodology: links para docs/ (backtest_metrics.md, design_gap.md)
10. Results & Limitations: honestidade sobre o gap backtest-produção
11. License + Acknowledgments
```

**Tarefas:**
- [ ] Criar diagrama de arquitetura (Mermaid ou draw.io exportado como PNG)
- [ ] Gravar GIF do War Room (debate de 3-5 tickers em Replay Mode, ~30 segundos)
- [ ] Tirar screenshot do dashboard Benchmark com equity curve
- [ ] Escrever seção "Methodology" explicando CPCV/DSR em inglês para audiência quant
- [ ] Escrever seção "Limitations" com honestidade sobre o que não foi validado (ganha credibilidade)
- [ ] Adicionar badges: `tests passing`, `coverage`, `Python 3.10+`, `License MIT`

### 4.2 — Notebook de análise: metodologia

Criar `notebooks/01_methodology.ipynb`:
- [ ] Explicar o problema: por que equity selection é difícil (efficient market hypothesis context)
- [ ] Mostrar CPCV-OOS com visualização dos 15 paths combinatórios
- [ ] Mostrar curva do Deflated Sharpe Ratio (DSR) vs número de trials
- [ ] Mostrar o trade-off MaxDD vs Sharpe (gráfico pré/pós fine-tuning)
- [ ] Incluir a lição da sessão 40: o perigo de otimizar max_weight via CPCV (gráfico comparativo dos dois runs)
- [ ] Resultados finais: equity curve vs SPY com regime overlay

### 4.3 — Notebook de resultados: análise de performance

Criar `notebooks/02_results_analysis.ipynb`:
- [ ] Equity curve do portfólio vs SPY com bandas de confiança
- [ ] Rolling Sharpe (252 dias) mostrando estabilidade
- [ ] Análise por regime de mercado: bull market (2019-2021), COVID crash (2020-03), bear (2022), recovery (2023-2024)
- [ ] Drawdown profile e tempo de recuperação
- [ ] Heatmap de correlação entre ações do portfólio (pós-HRP)
- [ ] Análise de turnover: quantas posições mudam por rebalanceamento
- [ ] Comparação vs benchmarks alternativos: equal-weight, risk parity ingênua, momentum puro

### 4.4 — ARCHITECTURE.md: atualização

**Tarefas:**
- [ ] Atualizar com estado atual das Fases 1-8 + fixes
- [ ] Adicionar diagrama do pipeline de dados (ingestão → features → PatchTST → agentes → HRP → execução)
- [ ] Documentar claramente o backtest-produção gap e a escolha de Option A (fallback PatchTST)
- [ ] Adicionar seção "Design Decisions" com os trade-offs importantes (ward vs single linkage, 2/n cap, etc.)
- [ ] Remover planos futuros que nunca serão implementados (evitar promessas não cumpridas)

**Critério de conclusão da Fase 4:** README com GIF do War Room, 2 notebooks executáveis sem erros, ARCHITECTURE.md atualizado, projeto apresentável para recrutadores e quant practitioners.

---

## Fase 5 — Publicação
**Objetivo:** Repositório público com lançamento estratégico.

### 5.1 — Auditoria de segurança

**Tarefas:**
- [ ] Rodar `git log --all --full-history -- .env` e verificar se `.env` foi commitado em algum momento
- [ ] Se sim: usar `git filter-repo` para remover do histórico antes de tornar público
- [ ] Verificar `.gitignore`: `*.env`, `data/outputs/*.parquet`, `data/outputs/*.json` (outputs com dados reais)
- [ ] Criar `.env.example` com todas as variáveis e placeholders (sem valores reais)
- [ ] Verificar se há API keys hardcoded em qualquer arquivo de código ou notebook
- [ ] Limpar notebooks de output: `jupyter nbconvert --clear-output notebooks/*.ipynb`

### 5.2 — CI/CD: verificação final

**Tarefas:**
- [ ] Verificar se GitHub Actions (.github/workflows/) está configurado e passando
- [ ] Adicionar badge de status dos testes no README
- [ ] Verificar se `docker-compose up` funciona em ambiente limpo (sem dados pré-existentes)
- [ ] Testar quick start completo: `docker-compose up → make benchmark --naive → make run`
- [ ] Adicionar `CONTRIBUTING.md` básico (como rodar testes, como contribuir)

### 5.3 — Licença e metadados

- [ ] Adicionar `LICENSE` (MIT recomendado para portfólio)
- [ ] Atualizar `pyproject.toml` com versão `1.0.0`, descrição, author, homepage
- [ ] Criar release tag `v1.0.0` com changelog

### 5.4 — Lançamento

**Tarefas:**
- [ ] Tornar o repositório público no GitHub
- [ ] Criar GitHub Release `v1.0.0` com summary dos resultados
- [ ] Adicionar topics no GitHub: `quantitative-finance`, `algorithmic-trading`, `langchain`, `langgraph`, `portfolio-optimization`, `machine-learning`, `python`
- [ ] Postar no LinkedIn: foco na inovação (multi-agent LLM + quant rigoroso), incluir GIF do War Room e gráfico de performance
- [ ] Opcional: postar no r/algotrading ou QuantConnect community com link para o repo

---

## O que NÃO fazer (riscos de escopo)

| Tentação | Por que não fazer |
|---|---|
| Mais fine-tuning CPCV-OOS | Espaço de parâmetros esgotado; risco de otimização enganosa (sessão 40) |
| Integrar agentes no walk-forward | Custo de tokens inviável; backtest-produção gap não resolve o problema fundamental |
| Adicionar mais tickers | 52 large caps US é suficiente e bem documentado; mais tickers = mais ruído |
| Retreinar PatchTST sem CPCV-OOS | Look-ahead bias potencial; não fazer sem validação completa |
| Refatorar arquitetura existente | O que funciona, funciona; refatorar sem necessidade é risco de quebrar os 1002 testes |

---

## Checklist de Publicação

```
[ ] Pipeline de agentes validado ponta-a-ponta (Fase 1)
[ ] War Room ao vivo funcional (Fase 1)
[ ] Agentes de desenvolvimento refinados (Fase 2)
[ ] Dashboard sem warnings de deprecação (Fase 3)
[ ] Cobertura de testes dos agentes > 70% (Fase 3)
[ ] README com GIF do War Room (Fase 4)
[ ] Notebooks executáveis (Fase 4)
[ ] Zero API keys no histórico do git (Fase 5)
[ ] GitHub Actions passando (Fase 5)
[ ] Repositório público (Fase 5)
```

---

## Métricas de Sucesso do Portfólio

O projeto será considerado de alto impacto se:
- **Metodologia:** CPCV-OOS + DSR + HRP são explicados corretamente e com profundidade
- **Honestidade:** O backtest-produção gap é documentado explicitamente (isso ganha credibilidade na comunidade quant)
- **Diferenciação:** A combinação PatchTST + LangGraph debate + HRP é genuinamente novel — não existe projeto público equivalente
- **Funcionamento:** War Room ao vivo é o "wow factor" — um debate de 4 agentes de IA sobre portfólio, visível em tempo real
- **Rigor:** Sharpe=0.710 com metodologia anti-overfitting documentada é resultado real e defensável
