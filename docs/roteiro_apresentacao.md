# Roteiro de Apresentação

Sugestão para uma apresentação de 8 a 10 minutos.

## 1. Abertura

Apresentar o tema:

> Nosso trabalho usa Inteligência Artificial para prever se uma música estilo Spotify será popular ou não popular com base em atributos musicais.

Falar o problema:

> O objetivo é classificar músicas em duas classes: popular e não popular.

## 2. Dataset

Explicar:

- A base foi criada pela equipe em `src/generate_dataset.py`.
- Ela tem 5.000 músicas e 15 colunas.
- Os atributos incluem energia, dançabilidade, intensidade sonora, acústica, valência, tempo, duração e gênero.
- A variável alvo é `popularidade_alta`, criada a partir de `popularity >= 70`.

Mostrar:

- `data/spotify_tracks_sample.csv`
- `results/distribuicao_popularidade.png`
- `results/distribuicao_classes.png`

## 3. Preparação dos Dados

Explicar o pipeline:

- Remoção de valores ausentes.
- Criação da classe alvo.
- Padronização dos atributos numéricos com `StandardScaler`.
- Codificação do gênero musical com `OneHotEncoder`.
- Divisão estratificada em 75% treino e 25% teste.

Frase pronta:

> A divisão estratificada mantém proporção semelhante de músicas populares e não populares nos conjuntos de treino e teste.

## 4. Modelos

Modelo da Parte 1:

> Usamos Árvore de Decisão, que aprende regras interpretáveis a partir dos atributos.

Modelo da Parte 2:

> Usamos Rede Neural MLP, que aprende combinações mais complexas e não lineares entre os atributos.

Mostrar:

- `results/arvore_decisao_visualizacao.png`
- `results/importancia_atributos_arvore_decisao.png`
- `results/curva_treinamento_rede_neural_mlp.png`

## 5. Treinamento rodando

Rodar no terminal:

```bash
python src/main.py
```

Enquanto roda, explicar:

> O script carrega a base, prepara os dados, treina os dois modelos, salva os modelos em `models/` e gera métricas e gráficos em `results/`.

## 6. Resultados

Mostrar:

- `results/metricas_modelos.csv`
- `results/comparacao_modelos.png`
- `results/matriz_confusao_arvore_decisao.png`
- `results/matriz_confusao_rede_neural_mlp.png`

Frase pronta:

> A Rede Neural MLP teve melhor F1-score, então foi considerada o melhor modelo neste experimento. A Árvore de Decisão teve desempenho menor, mas ajuda a explicar as regras aprendidas.

## 7. Demonstração de Predição

Rodar:

```bash
python src/predict.py
```

Explicar:

> Aqui passamos exemplos de músicas fictícias para o modelo treinado. Ele retorna a classe prevista e a probabilidade estimada de a música ser popular.

## 8. Conclusão

Fechar com:

> O projeto cobre o processo completo de IA: definição do problema, criação do dataset, preparação dos dados, treinamento, avaliação, comparação e conclusão. O principal aprendizado foi que modelos diferentes têm vantagens diferentes: a árvore é mais interpretável, enquanto a MLP teve melhor desempenho.

## 9. Divisão entre integrantes

Sugestão:

- Integrante 1: contexto, problema e hipótese.
- Integrante 2: dataset e preparação dos dados.
- Integrante 3: Árvore de Decisão e interpretabilidade.
- Integrante 4: Rede Neural MLP, resultados e conclusão.

Se a equipe tiver menos integrantes, juntar os blocos de modelos e resultados na mesma fala.
