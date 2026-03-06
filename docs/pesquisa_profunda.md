# **Fronteiras da Engenharia Financeira Quantitativa: Um Roteiro Técnico e Estratégico para o Desenvolvedor Independente na Era da IA (2025-2026)**

## **Sumário Executivo: A Nova Hegemonia do "Quant Aumentado"**

O ecossistema da negociação quantitativa atravessou um ponto de inflexão decisivo na metade da década de 2020\. Historicamente, a geração de *alpha* (retorno acima do mercado ajustado ao risco) era um privilégio exclusivo de instituições com capital massivo, acesso a dados proprietários e exércitos de PhDs em física e matemática. No entanto, o cenário de 2025 e 2026 revela uma democratização radical impulsionada pela convergência de três vetores tecnológicos: a maturidade da Inteligência Artificial Generativa e Agêntica, a institucionalização das estruturas de mercado de Finanças Descentralizadas (DeFi) e o avanço de arquiteturas de *Deep Learning* especializadas em séries temporais financeiras.

Para o desenvolvedor independente que busca construir um portfólio de "estado da arte" (SOTA \- *State of the Art*), o desafio não reside mais na escassez de ferramentas, mas na complexidade da integração sistêmica. Um projeto que se pretenda avançado hoje não pode se limitar a regressões lineares ou estratégias de cruzamento de médias móveis. A fronteira do conhecimento exige a implementação de **Transformers de Séries Temporais (como PatchTST e LiT)**, o uso de **Aprendizado por Reforço Profundo (Deep Reinforcement Learning \- DRL)** enriquecido por análise de sentimento via LLMs, e a exploração de **Intenções e Solvers** em ambientes de liquidez programável como o Uniswap v4.

Este relatório técnico oferece uma análise exaustiva e um guia de implementação para essas tecnologias. Ele foi desenhado para o desenvolvedor solitário que, armado com assistentes de codificação baseados em IA (como Cursor e Claude 3.5), deseja replicar a sofisticação de um *hedge fund* multiestratégia. A análise a seguir transcende a superficialidade, mergulhando nas nuances matemáticas das arquiteturas de modelos, na engenharia de dados necessária para alimentar agentes autônomos e nos protocolos rigorosos de validação estatística que separam a descoberta de *alpha* genuíno do mero ruído estatístico.

## ---

**Parte I: A Fronteira da Geração de Alpha (Deep Learning & Microestrutura)**

O coração de qualquer sistema quantitativo é o modelo de previsão, ou "Modelo Alpha". A evolução recente neste campo foi marcada pelo abandono gradual das Redes Neurais Recorrentes (RNNs) e Long Short-Term Memory (LSTMs), que dominaram a literatura até 2022, em favor de arquiteturas baseadas em *Transformers*. A premissa central é que a capacidade de capturar dependências de longo prazo e interações não lineares em dados de alta frequência exige mecanismos de atenção mais robustos do que os oferecidos pelas recorrências sequenciais.

### **1.1 A Revolução dos Transformers em Séries Temporais: Além da NLP**

A aplicação direta de Transformers (como o BERT ou GPT) a dados financeiros enfrentou, inicialmente, barreiras significativas. Dados financeiros são contínuos, ruidosos e não possuem a semântica discreta de um vocabulário linguístico. A solução para este impasse emergiu com o desenvolvimento de técnicas de *tokenização* adaptadas ao domínio temporal, onde o modelo **PatchTST** (Patch Time Series Transformer) se estabeleceu como um divisor de águas entre 2024 e 2025\.1

#### **1.1.1 PatchTST e a Independência de Canal**

O PatchTST introduziu uma mudança paradigmática ao questionar a necessidade de modelar explicitamente as correlações cruzadas entre ativos dentro do mecanismo de atenção. Em vez de processar cada ponto de tempo individualmente, o modelo segmenta a série temporal em "patches" (subsequências ou janelas de tempo), que funcionam como os *tokens* de entrada para o Transformer.2

A arquitetura opera sob dois princípios fundamentais que um desenvolvedor independente deve compreender para implementação:

1. **Segmentação (Patching):** Uma série temporal univariate ![][image1] de comprimento ![][image2] é dividida em patches de comprimento ![][image3] com um passo (*stride*) ![][image4]. Isso gera uma sequência de aproximadamente ![][image5] tokens. A implicação prática é uma redução drástica na complexidade computacional da memória de atenção, que cai de quadrática em relação ao tempo (![][image6]) para quadrática em relação ao número de patches. Isso permite que o modelo "olhe para trás" (look-back window) por períodos muito mais longos — 512 ou 1024 pontos de tempo — capturando ciclos de mercado e tendências sazonais que modelos anteriores (limitados a janelas curtas) perdiam.3  
2. **Independência de Canal (Channel Independence):** Ao lidar com dados multivariados (por exemplo, preços de fechamento de 500 ações), o PatchTST trata cada série como um canal independente que compartilha os mesmos pesos do modelo (embedding e atenção). Contrariando a intuição de que a correlação entre ativos (ex: Apple e Microsoft movendo-se juntas) deve ser o foco primário, a evidência empírica sugere que forçar o modelo a aprender uma dinâmica temporal universal, aplicável a qualquer ativo, resulta em generalização superior e menor *overfitting*. As correlações cruzadas acabam sendo capturadas indiretamente, pois o modelo aprende a reagir a padrões globais de mercado que afetam todas as séries.1

Para um projeto de portfólio, a implementação do PatchTST demonstra domínio sobre as técnicas mais eficientes de pré-processamento e modelagem. O desenvolvedor deve utilizar bibliotecas como NeuralForecast ou implementações diretas em PyTorch, focando na otimização dos hiperparâmetros de *stride* e *patch length* para se adequar à volatilidade do ativo em questão.

#### **1.1.2 Limit Order Book Transformers (LiT): Microestrutura de Alta Frequência**

Enquanto o PatchTST domina a previsão de tendências de médio prazo, o estado da arte para negociação de alta frequência (HFT) e análise de microestrutura reside nos **Limit Order Book Transformers (LiT)**.4 O Livro de Ofertas (LOB) apresenta uma estrutura de dados tridimensional complexa: tempo, níveis de preço e volume (bid/ask), exigindo uma arquitetura que compreenda tanto a dinâmica temporal quanto a espacial.

Diferente de abordagens anteriores que utilizavam Redes Neurais Convolucionais (CNNs) para extrair características "espaciais" do LOB (tratando-o como uma imagem), o LiT emprega mecanismos de autoatenção para modelar as dependências entre diferentes níveis de preço. A inovação crucial aqui é a compreensão de que a liquidez não é estática; a presença de uma grande ordem de venda a cinco níveis de distância do preço atual (Best Ask \+ 5 ticks) exerce uma "pressão gravitacional" sobre o preço que modelos lineares ignoram.

A arquitetura LiT utiliza "patches estruturados" que agrupam níveis de preço e janelas de tempo, permitindo que o modelo aprenda padrões complexos como *spoofing* (ordens falsas colocadas para manipular o preço) ou exaustão de liquidez.4 Para o desenvolvedor independente, treinar um modelo LiT requer acesso a dados de LOB de alta fidelidade (como os disponibilizados publicamente pela Binance ou datasets acadêmicos como LOBSTER), e o projeto serve como uma demonstração poderosa de engenharia de dados e arquitetura de *Deep Learning* avançada.

### **1.2 O Renascimento da Análise de Frequência: TimesNet e iTransformer**

Além da atenção temporal, 2025 viu o ressurgimento da análise no domínio da frequência. Modelos como o **TimesNet** transformam a série temporal 1D em representações 2D baseadas em periodicidade (usando Transformada Rápida de Fourier \- FFT), permitindo o uso de *backbones* de visão computacional modernos (como Inception blocks) para capturar variações intra e inter-períodos.1

Similarmente, o **iTransformer** inverte a lógica convencional dos Transformers. Em vez de calcular a atenção sobre o eixo temporal (tokens de tempo), ele calcula a atenção sobre o eixo das variáveis (tokens de ativos).5 Isso é particularmente eficaz para capturar correlações complexas em portfólios diversificados, onde a relação entre, por exemplo, o preço do petróleo e uma ação de companhia aérea, é dinâmica e não-estacionária. Implementar e comparar o PatchTST contra o iTransformer em um conjunto de dados proprietário constitui um projeto de pesquisa robusto e atual.

## ---

**Parte II: Execução Adaptativa e Aprendizado por Reforço Profundo (DRL)**

Se os modelos de previsão são o "cérebro" analítico, o Aprendizado por Reforço Profundo (DRL) é o sistema nervoso responsável pela ação. A transição de modelos supervisionados (que apenas prevêem o preço) para agentes de DRL (que aprendem uma política de negociação) é um marco de sofisticação em projetos quantitativos. O problema fundamental que o DRL resolve é a natureza estocástica e interativa do mercado: uma previsão de preço correta não garante lucro se a execução for pobre ou se a gestão de risco for falha.

### **2.1 A Fusão NLP-DRL: O Framework "Primo"**

Uma das tendências mais avançadas documentadas em 2025 é a integração de características semânticas extraídas de Grandes Modelos de Linguagem (LLMs) diretamente no espaço de observação do agente de DRL. O framework **Primo** exemplifica essa abordagem híbrida, combinando um módulo de NLP (**PrimoGPT**) com um módulo de controle (**PrimoRL**).6

#### **Arquitetura e Fluxo de Informação**

O desenvolvedor deve conceber o "Estado" (![][image7]) do agente não apenas como um vetor de preços e indicadores técnicos, mas como uma representação multimodal do mercado. No Primo, o estado é expandido para incluir:

1. **Indicadores Técnicos:** RSI, MACD, Bandas de Bollinger, Retornos Logarítmicos.  
2. **Estado do Portfólio:** Saldo em caixa, posições abertas, lucro não realizado.  
3. **Embeddings Semânticos:** Vetores densos gerados por um LLM (como FinBERT ou um modelo Llama-3 fine-tuned) que processou as notícias e relatórios do dia. Estes vetores codificam sentimentos sutis — incerteza, otimismo cauteloso, medo regulatório — que não aparecem nos preços imediatamente.6

A inovação reside na fusão desses dados. O agente, tipicamente operando sob o algoritmo **PPO (Proximal Policy Optimization)**, aprende a ponderar a importância do sinal técnico versus o sinal textual. Por exemplo, em dias de alta incerteza macroeconômica (detectada pelo componente de NLP), o agente pode aprender a reduzir o tamanho das posições ou exigir confirmações técnicas mais fortes antes de entrar em um trade, comportando-se de maneira mais conservadora do que um algoritmo puramente técnico.

### **2.2 Ensembles e Meta-Agentes**

A instabilidade é o "calcanhar de Aquiles" do DRL. Para mitigar o risco de um agente aprender uma política degenerada, o estado da arte envolve o uso de **Ensembles de Agentes**. Um projeto avançado não deve depender de um único algoritmo. Em vez disso, treina-se simultaneamente agentes baseados em **PPO** (bom para estabilidade), **A2C** (Advantage Actor-Critic, rápido) e **DDPG** (Deep Deterministic Policy Gradient, eficiente em espaços de ação contínuos).7

Um **Meta-Agente** ou mecanismo de votação supervisiona esses sub-agentes. O Meta-Agente avalia o desempenho recente (janela deslizante) de cada sub-agente e aloca capital dinamicamente para aquele que melhor se adapta ao regime de mercado atual. Se o mercado está em tendência forte, o DDPG pode performar melhor; em mercados laterais e ruidosos, o PPO pode ser superior. A implementação dessa hierarquia demonstra uma compreensão profunda de arquitetura de sistemas e gestão de risco algorítmica.

### **2.3 Generalização Zero-Shot com Cluster Embedding (CE-PPO)**

Um desafio comum para desenvolvedores independentes é a escassez de dados históricos para todos os ativos que desejam negociar. O framework **CE-PPO** (Cluster Embedding PPO) ataca este problema introduzindo a capacidade de generalização *Zero-Shot*.9

A técnica envolve o aprendizado não supervisionado de "Cluster Embeddings" — representações vetoriais que agrupam ativos com comportamentos dinâmicos similares (ex: ações de tecnologia de alta volatilidade vs. *utilities* de baixo beta). Quando o sistema encontra um ativo novo, que não estava no conjunto de treinamento, ele primeiro identifica a qual *cluster* este ativo pertence e, em seguida, aplica a política de negociação aprendida para aquele cluster. Isso permite que o sistema negocie centenas de ativos sem a necessidade de re-treinar modelos individuais para cada um, uma característica essencial para escalabilidade em produção.9

## ---

**Parte III: A Camada Cognitiva \- Agentes Autônomos e Raciocínio (Agentic AI)**

Enquanto o Deep Learning lida com padrões numéricos, a "Nova Fronteira" de 2026 é o uso de IA Agêntica para replicar o processo de raciocínio de analistas humanos. Projetos nesta área não buscam apenas prever o "o quê" (preço), mas entender o "porquê" (causalidade), utilizando a capacidade de raciocínio lógico dos LLMs mais recentes.

### **3.1 Frameworks Multi-Agentes: A Firma de Trading Virtual**

O projeto mais ambicioso para um portfólio moderno é a construção de um sistema multi-agente, como o descrito na arquitetura **TradingAgents**.10 A ideia central é decompor o problema complexo da decisão de investimento em sub-tarefas especializadas, cada uma executada por uma instância de LLM com um *persona* e ferramentas específicas.

#### **Arquitetura e Papéis**

Utilizando bibliotecas como LangGraph ou AutoGen, o desenvolvedor orquestra a interação entre:

1. **Analista de Fundamentos:** Acessa APIs (como SEC EDGAR ou Yahoo Finance) para ler balanços e *filings* (10-K, 10-Q). Sua função é calcular métricas de *valuation* e identificar riscos de liquidez ou solvência.  
2. **Analista Técnico:** Recebe dados de preço e volume, calculando indicadores e identificando padrões gráficos.  
3. **Analista de Sentimento:** Varre redes sociais (Twitter/X, Reddit) e feeds de notícias, utilizando ferramentas de NLP para quantificar o "humor" do mercado.  
4. **Pesquisadores (Touro e Urso):** Aqui reside a inovação do "Debate Estruturado". Um agente é instruído a ser otimista (*Bull*) e encontrar todas as razões para comprar. Outro é instruído a ser pessimista (*Bear*) e encontrar falhas na tese. Eles debatem entre si, gerando argumentos e contra-argumentos.10  
5. **Gerente de Risco:** Monitora a volatilidade e a exposição do portfólio, tendo poder de veto sobre as decisões.  
6. **Trader (Decisor):** Sintetiza os relatórios dos analistas e o resultado do debate para tomar a decisão final de alocação.

Este fluxo simula o "Sistema 2" de pensamento humano (lento, deliberativo e lógico), contrastando com a natureza "Sistema 1" (rápido e intuitivo) das redes neurais. Para o portfólio, documentar os *logs* desses debates e mostrar como o sistema evitou um trade ruim devido à intervenção do "Agente Urso" é uma prova poderosa de robustez.

### **3.2 Interpretando o "Fedspeak" com Quantificação de Incerteza**

A comunicação de bancos centrais ("Fedspeak") é intencionalmente ambígua. Modelos simples de sentimento falham em capturar as nuances de uma ata do FOMC. O estado da arte envolve o uso de LLMs para quantificar a **Incerteza Perceptual**.12

O método envolve pedir ao LLM que preveja as próximas palavras em declarações chave de política monetária e medir a *entropia* dessa distribuição de probabilidade. Alta entropia sugere que o próprio banco central está incerto ou enviando sinais mistos. Além disso, utiliza-se o LLM para extrair grafos de causalidade (Causal Graphs) do texto, mapeando afirmações como "Se a inflação persistir acima de 3%, então as taxas serão mantidas". Integrar esse sinal de incerteza em estratégias de volatilidade (como a compra de *straddles* antes de anúncios) é uma aplicação direta e sofisticada.

## ---

**Parte IV: Microestrutura DeFi, Intenções e Liquidez Programável**

O mercado de criptoativos e Finanças Descentralizadas (DeFi) oferece um *playground* único para engenharia financeira, onde as regras do mercado são definidas por código (Smart Contracts) e não por reguladores humanos. Em 2026, a inovação migrou da simples arbitragem de preços para a otimização de execução via **Intenções** e **Hooks**.

### **4.1 Uniswap v4 Hooks: O Poder da Liquidez Dinâmica**

O Uniswap v4 introduziu o conceito de "Hooks" — contratos inteligentes que executam lógica arbitrária em pontos específicos do ciclo de vida de uma *pool* de liquidez (ex: antes de um swap, depois de adicionar liquidez).13 Isso permite a criação de estratégias de *Market Making* on-chain que reagem em tempo real às condições do mercado.

**Projeto de Portfólio: Hook de Taxas Baseadas em Volatilidade**

Uma implementação avançada envolveria criar um Hook que atua como um oráculo de volatilidade interno.

* **Mecanismo:** O contrato armazena um histórico curto dos últimos preços de transação. A cada novo swap, ele calcula a variância desses preços.  
* **Lógica:** Se a variância (volatilidade) exceder um certo limiar, o Hook aumenta automaticamente a taxa de swap da pool (ex: de 0.3% para 1.0%).  
* **Justificativa:** Em momentos de alta volatilidade, o risco de Perda Impermanente (Impermanent Loss \- IL) para os provedores de liquidez aumenta drasticamente. Aumentar as taxas compensa os LPs por esse risco adicional e desencoraja a arbitragem tóxica (LVR \- Loss Versus Rebalancing).  
* **Diferencial:** Demonstrar, através de simulações em Python (usando o *framework* Foundry para testes), que essa pool dinâmica supera uma pool estática em rentabilidade.15

### **4.2 Solvers e o Paradigma de Intenções (CoW Swap, UniswapX)**

O modelo de transação tradicional ("Eu envio esta transação exata") está sendo substituído pelo modelo de Intenções ("Eu quero trocar X por Y ao melhor preço, não importa como"). Quem executa essa vontade são agentes off-chain chamados **Solvers**.16

Construir um Solver é um projeto de engenharia de software e otimização matemática de elite.

* **O Desafio:** O Solver compete em leilões de lote (batch auctions) para encontrar a melhor execução para um conjunto de ordens. Isso é uma variação do problema da mochila (*Knapsack Problem*) ou do problema do caixeiro viajante, onde o objetivo é maximizar o excedente (surplus) dos usuários.  
* **Tecnologia:** Solvers competitivos são escritos em **Rust** ou **C++** para minimizar a latência e maximizar a eficiência computacional. Eles interagem com a *mempool* pública e privada, buscando liquidez em múltiplas fontes (DEXs, CEXs, Market Makers privados) e combinando ordens que se anulam (Coincidence of Wants \- CoW) para economizar taxas de gás e impacto de preço.17

## ---

**Parte V: Engenharia Rigorosa e Validação (Onde a Maioria Falha)**

A marca de um quant amador é um backtest com Sharpe Ratio de 5.0 que perde dinheiro na primeira semana de produção. O profissionalismo em 2026 é definido pelo rigor estatístico da validação.

### **5.1 Combinatorial Purged Cross-Validation (CPCV)**

A validação cruzada padrão (K-Fold) é matematicamente inválida para séries temporais financeiras devido à correlação serial dos dados. O método **CPCV** (Combinatorial Purged Cross-Validation), formalizado por Marcos Lopez de Prado, é o padrão ouro.19

O desenvolvedor deve implementar três salvaguardas críticas:

1. **Purging (Expurgo):** Remover do conjunto de treinamento quaisquer amostras cujos rótulos (labels) se sobreponham temporalmente às amostras do conjunto de teste. Se você está prevendo o retorno de 5 dias à frente, deve remover os 5 dias de dados imediatamente anteriores ao início do teste para evitar vazamento de informação (*leakage*).  
2. **Embargo:** Descartar um período adicional *após* o conjunto de teste. Como as correlações financeiras decaem lentamente, amostras imediatamente posteriores ao teste ainda podem carregar "memória" do que aconteceu durante o teste.22  
3. **Combinatorial Split:** Em vez de um único caminho de teste (Walk-Forward), o CPCV gera múltiplos caminhos combinatórios de treino/teste. Isso permite gerar uma **distribuição** de Sharpe Ratios, não apenas um número. Se a estratégia falha em uma porcentagem significativa desses caminhos, ela deve ser descartada, independentemente de quão bom seja o resultado médio.

### **5.2 VectorBT Pro: O Motor de Simulação**

Para implementar CPCV e testar milhões de combinações de parâmetros, bibliotecas baseadas em loops (como Backtrader ou Zipline) são inviáveis. A solução SOTA é o **VectorBT Pro**.23

* **Vetorização:** O VectorBT opera inteiramente sobre arrays NumPy e utiliza a biblioteca Numba para compilar código Python em linguagem de máquina. Ele evita iterações linha-a-linha, calculando o PnL de milhares de estratégias simultaneamente através de operações de álgebra linear e *broadcasting*.  
* **Simulação de Portfólio:** Ele permite simular não apenas sinais de compra/venda, mas a dinâmica complexa de um portfólio rebalanceado, considerando taxas, *slippage* variável e impacto de mercado, tudo em segundos. Dominar essa ferramenta é essencial para a iteração rápida exigida em projetos avançados.

## ---

**Parte VI: O Stack Tecnológico Assistido por IA**

Para um desenvolvedor solo realizar tudo o que foi descrito acima, o uso eficiente de ferramentas de IA não é opcional, é estrutural.

### **6.1 O Fluxo de Trabalho: Cursor \+ Claude 3.5 Sonnet**

O IDE **Cursor** (um fork do VS Code) integrado ao modelo **Claude 3.5 Sonnet** (ou GPT-4o) atua como um par programador sênior.25

* **Regras de Contexto (.cursorrules):** O segredo para código de alta qualidade é definir regras estritas no arquivo .cursorrules do projeto.  
  * *Exemplo de Regra:* "Ao escrever código Python, utilize sempre *type hints* estritos. Prefira a biblioteca Polars em vez de Pandas para manipulação de dados. Utilize VectorBT para lógica de backtesting. Siga o padrão de design Factory para instanciar estratégias."  
* **Modo Composer:** Permite editar múltiplos arquivos simultaneamente. Você pode solicitar: "Refatore a classe AlphaModel para incluir uma camada de atenção cruzada e atualize os testes unitários em test\_alpha.py para refletir essa mudança", e a IA executará as alterações em todo o sistema de arquivos coerentemente.27

### **6.2 Polars vs. Pandas**

Em 2026, o uso de **Polars** tornou-se o padrão para processamento de dados financeiros em Python. Escrito em Rust, o Polars utiliza execução preguiçosa (*lazy evaluation*) e paralelismo multithread nativo, sendo 10 a 50 vezes mais rápido que o Pandas em grandes datasets.28 Para um projeto de LOB (Limit Order Book) com terabytes de dados, o Pandas é inviável, enquanto o Polars processa eficientemente em uma máquina local robusta.

## ---

**Parte VII: Projetos de Portfólio (Blueprints Concretos)**

Abaixo, detalham-se três especificações de projetos que cobrem diferentes espectros do conhecimento quantitativo. Escolher e executar *um* destes com profundidade posicionará o portfólio na elite do desenvolvimento independente.

### **Projeto A: "O Oráculo de Microestrutura" (Foco: Deep Learning & HFT)**

* **Objetivo:** Prever movimentos de curto prazo (mid-price) usando dados de Livro de Ofertas.  
* **Dados:** Dataset público de LOB (ex: FI-2010 ou dados de cripto da Binance via API Tardis).  
* **Arquitetura:** Implementar o **LiT (Limit Order Book Transformer)**.  
  * Entrada: Snapshots do LOB ![][image8].  
  * Modelo: Encoder Transformer com atenção espacial sobre os níveis de preço e atenção temporal sobre a sequência.  
  * Saída: Classificação ternária (Preço sobe, desce ou mantém em ![][image9] ticks).  
* **Diferencial SOTA:** Implementar uma camada de "Interpretabilidade". Usar mapas de atenção (*Attention Maps*) para visualizar quais níveis do livro de ofertas o modelo está "olhando" antes de grandes movimentos de preço. Isso demonstra não apenas habilidade de ML, mas entendimento de mercado.

### **Projeto B: "O Fundo de Hedge Agêntico" (Foco: LLMs & Engenharia de Sistemas)**

* **Objetivo:** Sistema autônomo de análise e recomendação de investimento *end-to-end*.  
* **Tech Stack:** Python, LangGraph, OpenAI API (ou Claude), Streamlit.  
* **Fluxo:**  
  1. O usuário insere um ativo.  
  2. O sistema ativa 5 agentes especializados (Macro, Técnico, Fundamentalista, Notícias, Risco).  
  3. Os agentes coletam dados reais.  
  4. Ocorre uma rodada de "Debate" onde o Agente de Risco desafia as teses dos outros.  
  5. O "Trader Agent" emite o veredito final com um "Score de Confiança".  
* **Diferencial SOTA:** Implementar **Memória de Longo Prazo** (via Vector DB). O sistema deve lembrar de análises passadas: "Há duas semanas, eu recomendei COMPRA em NVDA baseado em RSI, mas o trade falhou. Hoje, o RSI está similar, mas dado o erro anterior, vou reduzir a confiança."

### **Projeto C: "O Provedor de Liquidez Inteligente" (Foco: DeFi & Solidity)**

* **Objetivo:** Backtesting e Deploy de um Uniswap v4 Hook.  
* **Tech Stack:** Solidity, Foundry, Python (para análise de dados).  
* **Lógica:** Hook de taxa dinâmica baseada em volatilidade *on-chain* e desbalanceamento da pool.  
* **Validação:** Criar um script de simulação em Python que gera trajetórias de preço (Movimento Browniano Geométrico) e compara o retorno do Lider de Liquidez (LP) usando o Hook versus uma pool padrão do Uniswap v3.  
* **Diferencial SOTA:** Publicar o contrato na Testnet Sepolia e criar um *dashboard* simples (Dune Analytics ou customizado) mostrando as taxas capturadas em tempo real.

## ---

**Conclusão**

A barreira para a entrada na finança quantitativa de alto nível nunca foi tão baixa em termos de acesso a ferramentas, mas nunca foi tão alta em termos de complexidade intelectual exigida. O desenvolvedor que domina a sintaxe do **Transformer**, a semântica dos **Agentes** e a estrutura dos **Hooks DeFi** possui uma vantagem competitiva massiva.

A chave para o sucesso neste projeto de portfólio não é a amplitude — tentar fazer tudo — mas a profundidade e o rigor. Um único modelo LiT bem validado com CPCV e documentado com clareza vale mais do que dez bots de trading simples baseados em tutoriais genéricos. Utilize a IA como seu acelerador, mas mantenha o rigor matemático e a curiosidade investigativa como seus guias.

**Citações:** 1 (Séries Temporais/Transformers) 6 (Deep Reinforcement Learning) 10 (Agentes/LLMs/Fedspeak) 13 (DeFi/Hooks/Solvers) 19 (Backtesting/Validação) 25 (Ferramentas/Stack)

#### **Works cited**

1. (PDF) Transformers for time-series forecasting: A comprehensive survey through 2024, accessed February 10, 2026, [https://www.researchgate.net/publication/399324903\_Transformers\_for\_time-series\_forecasting\_A\_comprehensive\_survey\_through\_2024](https://www.researchgate.net/publication/399324903_Transformers_for_time-series_forecasting_A_comprehensive_survey_through_2024)  
2. PatchTST: Transformer-based Time-Series Modeling \- Emergent Mind, accessed February 10, 2026, [https://www.emergentmind.com/topics/patchtst](https://www.emergentmind.com/topics/patchtst)  
3. A Closer Look at Transformers for Time Series Forecasting: Understanding Why They Work and Where They Struggle \- ICML 2026, accessed February 10, 2026, [https://icml.cc/virtual/2025/poster/44262](https://icml.cc/virtual/2025/poster/44262)  
4. LiT: limit order book transformer \- Frontiers, accessed February 10, 2026, [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full)  
5. thuml/Time-Series-Library: A Library for Advanced Deep Time Series Models for General Time Series Analysis. \- GitHub, accessed February 10, 2026, [https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)  
6. Automated Trading Framework Using LLM-Driven Features and ..., accessed February 10, 2026, [https://www.mdpi.com/2504-2289/9/12/317](https://www.mdpi.com/2504-2289/9/12/317)  
7. \[2511.12120\] Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy \- arXiv, accessed February 10, 2026, [https://arxiv.org/abs/2511.12120](https://arxiv.org/abs/2511.12120)  
8. \[2511.00190\] Deep reinforcement learning for optimal trading with partial information \- arXiv, accessed February 10, 2026, [https://arxiv.org/abs/2511.00190](https://arxiv.org/abs/2511.00190)  
9. Deep Reinforcement Learning for Financial Trading: Enhanced by ..., accessed February 10, 2026, [https://www.mdpi.com/2073-8994/18/1/112](https://www.mdpi.com/2073-8994/18/1/112)  
10. TauricResearch/TradingAgents: TradingAgents: Multi ... \- GitHub, accessed February 10, 2026, [https://github.com/TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)  
11. TradingAgents: Multi-Agents LLM Financial Trading Framework, accessed February 10, 2026, [https://tradingagents-ai.github.io/](https://tradingagents-ai.github.io/)  
12. Interpreting Fedspeak with Confidence: A LLM-Based Uncertainty-Aware Framework Guided by Monetary Policy Transmission Paths \- arXiv, accessed February 10, 2026, [https://arxiv.org/html/2508.08001v1](https://arxiv.org/html/2508.08001v1)  
13. Unlocking Uniswap V4: Hooks as the Foundation for Next-Level DeFi \- DEV Community, accessed February 10, 2026, [https://dev.to/codebyankita/unlocking-uniswap-v4-hooks-as-the-foundation-for-next-level-defi-61o](https://dev.to/codebyankita/unlocking-uniswap-v4-hooks-as-the-foundation-for-next-level-defi-61o)  
14. johnsonstephan/awesome-uniswap-v4-hooks \- GitHub, accessed February 10, 2026, [https://github.com/johnsonstephan/awesome-uniswap-v4-hooks](https://github.com/johnsonstephan/awesome-uniswap-v4-hooks)  
15. Uniswap V4 Concentrated Liquidity Pricing: a Machine Learning ..., accessed February 10, 2026, [https://www.suaspress.org/ojs/index.php/JIET/article/view/v1n1a03](https://www.suaspress.org/ojs/index.php/JIET/article/view/v1n1a03)  
16. cowdao-grants/cow-py: CoW Protocol Python SDK \- GitHub, accessed February 10, 2026, [https://github.com/cowdao-grants/cow-py](https://github.com/cowdao-grants/cow-py)  
17. UniswapX, CoW Protocol, and 1inch Fusion Now Handle Billions in Monthly Volume Through Solver Auctions \- The Intent Architecture Explained for Builders \- General \- Web3 Developer Forum \- BlockEden.xyz, accessed February 10, 2026, [https://blockeden.xyz/forum/t/uniswapx-cow-protocol-and-1inch-fusion-now-handle-billions-in-monthly-volume-through-solver-auctions-the-intent-architecture-explained-for-builders/462](https://blockeden.xyz/forum/t/uniswapx-cow-protocol-and-1inch-fusion-now-handle-billions-in-monthly-volume-through-solver-auctions-the-intent-architecture-explained-for-builders/462)  
18. StateOfTheArt-quant/StateOfTheArt.quant: State-of-the-art performance in quantitative trading domain \- GitHub, accessed February 10, 2026, [https://github.com/StateOfTheArt-quant/StateOfTheArt.quant](https://github.com/StateOfTheArt-quant/StateOfTheArt.quant)  
19. Using Neural Networks and Combinatorial Cross-Validation for ..., accessed February 10, 2026, [https://fizzbuzzer.com/posts/using-neural-networks-and-ccv-for-smarter-stock-strategies/](https://fizzbuzzer.com/posts/using-neural-networks-and-ccv-for-smarter-stock-strategies/)  
20. The Combinatorial Purged Cross-Validation method \- Towards AI, accessed February 10, 2026, [https://towardsai.net/p/l/the-combinatorial-purged-cross-validation-method](https://towardsai.net/p/l/the-combinatorial-purged-cross-validation-method)  
21. Purged cross-validation \- Wikipedia, accessed February 10, 2026, [https://en.wikipedia.org/wiki/Purged\_cross-validation](https://en.wikipedia.org/wiki/Purged_cross-validation)  
22. Cross-validation tools for time series | by Samuel Monnier \- Medium, accessed February 10, 2026, [https://medium.com/@samuel.monnier/cross-validation-tools-for-time-series-ffa1a5a09bf9](https://medium.com/@samuel.monnier/cross-validation-tools-for-time-series-ffa1a5a09bf9)  
23. VectorBT® PRO: Getting started, accessed February 10, 2026, [https://vectorbt.pro/](https://vectorbt.pro/)  
24. Fundamentals \- VectorBT® PRO, accessed February 10, 2026, [https://vectorbt.pro/documentation/fundamentals/](https://vectorbt.pro/documentation/fundamentals/)  
25. AI Coding with Claude & Cursor \- Create a Complete App with Cursor \- YouTube, accessed February 10, 2026, [https://www.youtube.com/watch?v=kN66McRztJU\&vl=en](https://www.youtube.com/watch?v=kN66McRztJU&vl=en)  
26. Cursor AI Tutorial for Beginners \- YouTube, accessed February 10, 2026, [https://www.youtube.com/watch?v=3289vhOUdKA](https://www.youtube.com/watch?v=3289vhOUdKA)  
27. Mastering Long Codebases with Cursor, Gemini, and Claude: A Practical Guide, accessed February 10, 2026, [https://forum.cursor.com/t/mastering-long-codebases-with-cursor-gemini-and-claude-a-practical-guide/38240](https://forum.cursor.com/t/mastering-long-codebases-with-cursor-gemini-and-claude-a-practical-guide/38240)  
28. The Ultimate Python Quantitative Trading Ecosystem (2025 Guide) | by Mahmoud Ali | Jan, 2026 | Medium, accessed February 10, 2026, [https://medium.com/@mahmoud.abdou2002/the-ultimate-python-quantitative-trading-ecosystem-2025-guide-074c480bce2e](https://medium.com/@mahmoud.abdou2002/the-ultimate-python-quantitative-trading-ecosystem-2025-guide-074c480bce2e)  
29. Open-Finance-Lab/AgenticTrading \- GitHub, accessed February 10, 2026, [https://github.com/Open-Finance-Lab/AgenticTrading](https://github.com/Open-Finance-Lab/AgenticTrading)  
30. Uniswap v4 and the Future of Liquidity Provision: A Market Maker's Perspective, accessed February 10, 2026, [https://acherontrading.com/blog/uniswap-v4-and-the-future-of-liquidity-provision](https://acherontrading.com/blog/uniswap-v4-and-the-future-of-liquidity-provision)  
31. CoW Protocol \- GitHub, accessed February 10, 2026, [https://github.com/cowprotocol](https://github.com/cowprotocol)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAAAyklEQVR4Xu3RsQtBURTH8SMpokRSVslgYDAbKMrfIKtZFgMjo1JmyYBRFrNdKYPJ5A+wKJOB73n3vZIno+n96jPczjm3+84T8fLv+FFEDWH4kEUVobc+iWKJDno4YowhplgjqI16wwAla0wkhQtWyOOKHSJajKMv9iQp4IYGAmgiZ9dc0aa7mPf/jD5phj1iHzUr+nETtJDAScyADmp0G1qzUscTI5TxQNeu6UVzZOyzpHHAAhu0cRazsi0qTqMT3URSzI/5dvbiygvC9RzA6VnpHQAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAYCAYAAAAh8HdUAAAAtUlEQVR4XmNgGP7AEYhfA/F/JPwLiHcDsTCSOqxgDhD/A2IPdAlcQBCITwPxAyCWRpXCDTSB+C0QrwFiFjQ5nCCaAeKXcnQJfGASEP8GYht0CVwA5p+7QCyOJocTEPIPGxCzogvC/FOELgEEjEDcBMQ66BKg+MHlHxUgngvEnMiC+OIH5KRZDBCXoABjIP7KgOkfSQaIhkdArAgTdAHiZwyItPYXiJ9AMYgNE1/OgD1wRgHtAQAv+ie2Ic8IBwAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAAA2UlEQVR4Xu3SPQtBURzH8TMYDB4ySZRZmWRRDMrsTXgHRq9CRikZbFYLBmVRXoNioQiLRQrf+3Dq3r+Lu6r7q0/d7u+c0/13rlJBqtjj6XDGwX6+oo2Y3uCVHu4oi/cFZR00QUR0ZqJYYI2k6IwNczxQc1dWcjhhhJDoElgp768yU1fWfE1ZkBJuWCIuOjMd5X2ysXiGI4qiM6NnMk4fomsbYIc+MnqxjJ53iixSDmHHOs/oeVuy8BNj3o/X8C36fjdIu6vfyeOCsfIxn04FW/X+Pzeci4L8dV5HLDA3ZZscAQAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAYCAYAAAAh8HdUAAABE0lEQVR4Xu3SIUtDYRTG8TNUcCgTFJsWsQxsCrJmEpNFwYFrC4LBNBFlVUQwyAzLaxaTXcSoyWARBAd+AcPi1P+zs+G9h9tMAx/4wTjnfe/Ozp3Z0GYSa1jHVL82g9nBgWTGUMcbDrCPJ1zgAcXfo55RNHGNiURd3/CIe/MJUimhjaXYICdoxKJyig/MxQY5xGYsKi184xgjobeI6VDrpWx+Sbq4QxWF5KEYbe7M/MLgsjxb9sipaDSt9hyf5hf3UifMD2nmXGyQDXzhKDYWcGX+nmKW0cFubGiVtxiPDVLBK+ZjQ+9HT1sNdY2sJWyFeu9vcYMaXvqfteZLvGPHMn5r3vyJila+gm3zf3jWuP/5U34AsNUreE1r6AoAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACQAAAAYCAYAAACSuF9OAAACIklEQVR4Xu2VP0hXURTHv0FGUSD9gYga1IKQ2qIhaFEcaijC1dHBRQKDCsmpiKglKlrSbIxIx6Yg3BQEJ0ERRJFoMBxditLv1/Nev/vOu/e9QfK3+IEP6DmP97vv3HPPBfbZe1roIR9sJgP0hg8GaMGX6B3aRg9ksfPBM//ooj/pVuAv+pWeDJ5L0Uo/0NM+kdFNV+hr2kc/00n6CfYhScboX1R/aYxr9JkPZtymS7Dq5Kg6T+gmvRLECxyns3SVni2manlOr/sgrHIzdNAnyFU6TU/5RE4n3aAT9KDLVaFt0nbpxz1a5G/a6xOwyryCVSuK9la989AnatD2PvLBjFuwd36jJ1xO/19wsQJqOH1NrPQpVMm39LJPZLTT72gclEX6FDULEXn/LCN9UmLoB3UQjvhEwE26jvIJVvWS1PWPBp5mhkfbXHlsA/Sh/XQetqgv9HDhiYC8f+75BKzpHqO8LXrZKO1w8Zxz9KgPwha2QKfosWKqgcqe6h/t93uUt0ULVP/EKqrYG8QXq0VMIb0blfNH2/QOVkHPfdjQi6H3aP6oFTz6wFXER8EOmgeamH7FZ2CLWYM1b0jdVZHPn2EU54yqrN3Qe0s92UN/oNH5f2BHVOrvPP4R5dLqqniB9FBT9V7CPnKOPqB3Yb2j8RLrrV0xgni/5VyEVUALboPd8DrmqYruCt0944hfFU2h6qrYc1IzqWmoGYdQnkn7/He2Af4XYU9C0RjXAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAYCAYAAABXysXfAAADE0lEQVR4Xu2WTaiNQRjHH6F8Jh+Rr0RSopDIAgtJLFhgodgpbH0lVmdjYSNJkdSVwsJaSuIWpVhRopBIhLCxovD/3Znn3Dlz3o9zdS4W91f/3vPOzHlnnpnnY8yGGFRGS3uk8/HJ+z9jnDQqb+yQEdIxaYEFIy5KV6SRsX+4NFEaFt8HxGRpg7RdWmjhY1UslXqkCXlHh0yXXksH4vtq6Z20JL5jxEFpf/xdC4PWSg+kG9LOqJvSc2lF/9AWZkl3pEV5h5gi3ZV+ZWLha5JxzL3SwnhYb8EYNtLhlC5J25K2Qhh4Qnpl7YumDz/+Ki3L+ljEKamRtedssmDEGavfWVzugnQ2/k5ZLN2XZmftTVgsf/xiYXeKYIc+W/ti+Piz+KziuAVjtuQdBWyVTltxAsC4q1axefukn/FZBsH3UHpi/a4AR6TrVh349DHmvTQv68vhBIkNNniqNKm1uw9cn7WwphbmW/DNp9K0rC/FjcHXCVbwRZKFqsAADOm1kPHKwCv41gwLcxyyEI85JAXWkYdD33Fx/LhBFb6g1BjPQJt9UAkEMydfNQfZ85G1JgkSx/h0UMTnxR2bsEu9FiZiwip8QekEy6UPFtJoFR4vuFA38HXj4k3cQgI7TYFFEJAsqJG0Ycyb+CyjLl6oX2PyxhrcmJaTdmNS1ymCNEid+WSttaQTY9w9b0ljsz7YKO3NG2twY0jfTchKZKcqY0jDRy2cClkmpRNjvL4UxYuXhLq0nlPoZp6zf1i535NhKJYUTb8nOew6mbAqFqrqC5X8srUXxjo8s/rVpwkVncX2WPti10kfpZNWXMD8ZMvcxHcwjxe+tVv6Ju1I2juFdP3SSpIW+fqFhTuZ38e4mz22YFDZ9YN2NoHkkEKxI+t9t/40y6bgkhjgbW+lufE/AwEvYhNn5h0OWYWMxi2ZukHhKjMihZ1lE9qq8SDSsBAeA3XPWih29yxkpb8B89221ht3VyG4r1lxXHWbXdI5a4/vroE7Ho7qxDX/FO6RxPKcvKPbsFNcDFflHV2CKxRpvujiOcR/y29q5pgeHVdu+wAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAABZklEQVR4XuXUvyuFURzH8a/8iPwqPzZKsigbJWUw6U4GFEUZDMpgEImskqL8GMw2A5MMFmRkMliUUP4Bg9GP98e5D+c5PU/PZeRTr7p9z7nnPud8z3PN/mWq0Id+1OZr9WiMJmSlFMu4wwymcYV1XKD9e2p6SrCLfVR6dT3JJc7NPWlmevCIjnCALGE7LKZlBU9oCgfIPAbCYlr28I5FFAdjbagLaqkZNbeQvOIUk6jxJxUSdWzV3CLRgnJtydvNjLalNq/h2dxiU954kcW7+hV9UWegCWFyeMOCV1N3jy3hKrRix9w9CtOJF4x5tVlLuQpq6xHKwwEyjls0owVbuDd3ObVgWTRR0f3Rr3b7RXPb1UEPebUGnFnCq6J9HmION/nPavkmHjBi8bPrxQmqvdpnKsz9sqL2d2HY3JuftFV1L/F8fhI14wCD5s5sIj5ceLRFbXnD3F9N9D/1q2gxnU/SnfuL+QAULTZ3RW1dwgAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHEAAAAYCAYAAADNhRJCAAAFnUlEQVR4Xu2ZachuUxTH/zJknoe45BoiU5QhCtcHwgdD+HARH8iQkFkoHiQpZIxM16VryND9YhYvbjKUoYgMhUQUH4RIhvWzznqfffZzznnP85z33vd1Pf/6d9+z9z772Xuttf9rnX2lMcYYY/ZiTeOqeeMYyxwrG9fJG9tgN+MCjfjycgqCemPjinnHUgZOvMl4VN7RhM2NLxt3StpY/EvGr1ryRn+tNVY3PmT80/h3wu+NJyTjZgI7Gj+Tr+dD44bl7s5YzXic8S7jLcYDNRgoGxifNu6VtVdiBbnXe1n7IcaPjPPkkQEOlW/sNvl7YDO5sy8tnofFrsafjM/KNzdbsJ7xbeMi9fc6HUDpnjSeYtzGeIU8kJ8p+lLgg+flitCInY0fF/8GWDQn66CkDVwjd+LhWfvFGvLoJzhePidzzCbsYPzBeFre0RFnGh8wrl08Y+urVW0DnPq6cX7WPgBefErlggb5uEPlCFjD+KLxW+PWSTu4xLhn1tYWyMkfxn3zjhkGwbU01rVQ7rDzkjZs96vcvtg5BQfnceNKWfskcBwOzKWQnHBk1objcOCEBo830kAOHRYhWV8Y55S7aoG07yFf3xZZH89IPvkksKk8r1RJIvs/wHiYyu8AgqsuHzLXXPka9lc/3bQB7zBvqnK7G39RtW3Zz5fyuqUSbJABbGIqRD4kMqYLIVmNkVYAQ11o/NR4tlxiKMbCGBjnbnkaeF/ulFAPci65N0A7TnrPeLLxROMS9Qu7CK6qdbGO240PG48xXm58Qt3yeaSUG/IOuYO/VoPSMeA7tZOMunzYBbH4VFqqgCEppnDg3KSd/MK7qMB9csf15IFJgHJijjX+KN8rINKp+l5VuZBgfgIVNOVDig3yFI5mXTiz7sS2AWtgLZ8Yt8z6wJQHjY3xeRAbrAMbn1B1PuyCpnyIAyI/cNqo4HrFM6fhYONz8o0jrzgLg7whN2ycIIz9qHGT4vl041/qFwtIKieRijGc2pQPqSF+N54qD5pdCo4C9niR/HOGwKlCOJE1VaKtE5vy4agIyfpcfQOnoFq+Sr7Re9T/hmQ8TkEGo8IL7CMvENJqjlxyp9xZIa/M9Y28Kr9XLoupHDblwwgo5oAvGDcqjWiPo42vqawuOcKJtWrV1olI6HTnw/g+rMo7OI5cg3RF8RUS2QTWl6sFc5xU/I1TcM6E6oOxKR8GtpWfoA/ULh1UAQfybbh+8cx6OG35teeUcspmicjIBXUYJh9y80COyheTo+n7kGqSj38MikMXqdrw/BY3P4GFKo/DCdcbtyqe15JHPuNyINGrqJwPtzdeVvRzUilgyF1RiYfD09zJPPQ3Vazsjz2lOZnfvVmDVfSUPorIrErggWHzITKHc9K8lIOFLtBg3sEpR8jVoZe0I49sJP19jHqtyld0yC5OwlmAC4iz+t3/oic3PA4IcBpYD2vBWOQ8/j5DfWlG8imsGBfSy30zFS4nM0CFyf57SVsKnEUOJDWwzyDFV5XSUZWSQupy5qQxyQEpOEX3yw0X2g/JBZS750+OHATH/jf5D+fyR056xPiz+nOy+NhEtGFE8luAqL7O+K78vpGcSAEzLxkDMCqGRqYYg0HzE0H0E2CvyOdaLP9U2a7o5xS9U7Q9WIwH2OpceQDwHqf5TQ2u4Rx54UTuzZUD8F5q05RVxQttE6qeaxJE2lsqR2ZXEAS3qrpg6QI2QmA0bSjkfN28IwP9zFUl+02SyHjea5ofhWP/+e3LsIhPmF7WPgDK5CXyAmC6QA4iF9XJ6fIOVITCpyuQaRQnletaULA8pm63DgFOAg6kFP8/ApXg4iFuf0YF8n2lPBjyYqcSDOJKC7Z6oQFIDcVJ13n+q6AAyfPkKNhP/pmT//dUI9D/C4x75x1jLHPMkVeqQzlwjDHGmAn8A/kDKXknjKQFAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAXCAYAAADduLXGAAAA4UlEQVR4XuXSoYpCQRTG8SMquKigGA0iirDNBzBqsrnR4AtYtBjFKJpsxu2iGOyC0bppk0Fs+wIK6v/cOwMy94p1wQ9+cOfMwJk5XJF/mQhyyLgbbsY444a+sxeaFi6ouRthmeGAvFMPJI0dNkg4e4F84g8Ds9bHVtDAhz1k08YVdcQxwgRrCXmwvW8JQ1TFPxSYThZ7/GAu/pU0eo0ekmbt5XFkZfxiJU8e6o7sW/xO2rGJjqlLClssEDM1PaxrncIURVOXAk7o2gL5whFLU9cxetEPbRe1BRPt+PKHet/cAcfeIy832IBiAAAAAElFTkSuQmCC>