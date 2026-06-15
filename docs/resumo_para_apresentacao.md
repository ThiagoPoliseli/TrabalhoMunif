# Resumo Para Apresentacao

## Visao geral

Este projeto usa Inteligencia Artificial para prever se uma musica estilo Spotify sera popular ou nao popular com base em caracteristicas musicais.

O problema foi tratado como classificacao binaria:

- `popularidade_alta = 1`: musica popular, quando `popularity >= 70`.
- `popularidade_alta = 0`: musica nao popular, quando `popularity < 70`.

## O que o projeto aborda

O projeto cobre o fluxo completo de uma solucao de IA:

1. Definicao do problema.
2. Criacao do dataset.
3. Preparacao dos dados.
4. Treinamento dos modelos.
5. Avaliacao com metricas.
6. Geracao de graficos.
7. Comparacao dos resultados.
8. Salvamento dos modelos treinados.

## Dataset

A base foi criada pela equipe no arquivo `src/generate_dataset.py`.

Ela possui 5.000 musicas e 15 colunas, com atributos como:

- `danceability`: dancabilidade.
- `energy`: energia.
- `loudness`: intensidade sonora.
- `acousticness`: nivel acustico.
- `valence`: positividade da musica.
- `tempo`: batidas por minuto.
- `duration_ms`: duracao.
- `track_genre`: genero musical.

## Preparacao dos dados

No `src/main.py`, a preparacao acontece na funcao `prepare_data`.

Ela faz:

- Validacao das colunas obrigatorias.
- Remocao de valores ausentes.
- Criacao da variavel alvo `popularidade_alta`.
- Separacao entre `x`, que sao os atributos, e `y`, que e a classe que o modelo deve prever.

Depois, o `build_preprocessor` cria o tratamento automatico dos dados:

- `StandardScaler` padroniza os atributos numericos.
- `OneHotEncoder` transforma o genero musical em colunas numericas.

## Modelos utilizados

Foram usados dois modelos de aprendizado supervisionado:

- Arvore de Decisao: metodo da Parte 1 da disciplina. E mais facil de interpretar, pois gera regras de decisao.
- Rede Neural MLP: metodo da Parte 2 da disciplina. Aprende relacoes mais complexas e nao lineares entre os atributos.

Os modelos sao criados na funcao `build_models`.

## Treinamento e avaliacao

Na funcao `main`, o script:

1. Cria as pastas `models/` e `results/`.
2. Carrega o dataset.
3. Gera graficos de distribuicao.
4. Prepara os dados.
5. Divide a base em 75% treino e 25% teste.
6. Treina a Arvore de Decisao e a Rede Neural MLP.
7. Salva os modelos treinados em `models/`.
8. Avalia os modelos com dados de teste.
9. Gera matrizes de confusao, graficos e arquivos de metricas.

## Metricas

As metricas usadas foram:

- Acuracia: porcentagem geral de acertos.
- Precisao: entre as musicas previstas como populares, quantas realmente eram populares.
- Recall: entre as musicas realmente populares, quantas o modelo conseguiu encontrar.
- F1-score: equilibrio entre precisao e recall.
- Matriz de confusao: mostra acertos e erros por classe.

O F1-score foi usado como criterio principal para escolher o melhor modelo.

## Resultado principal

A Rede Neural MLP apresentou melhor desempenho geral no projeto.

Ela teve F1-score maior que a Arvore de Decisao, entao foi considerada o melhor modelo neste experimento.

Mesmo assim, a Arvore de Decisao continua importante porque e mais facil de explicar visualmente e ajuda a entender quais atributos influenciaram as decisoes.

## Fala curta para apresentar

Nosso trabalho desenvolve uma solucao de Inteligencia Artificial para prever se uma musica sera popular ou nao popular. Para isso, criamos um dataset inspirado em musicas do Spotify, com atributos como energia, dancabilidade, valencia, tempo, duracao e genero musical.

Transformamos a coluna `popularity` em uma classe binaria chamada `popularidade_alta`, considerando popular a musica com nota maior ou igual a 70. Depois, preparamos os dados com padronizacao dos atributos numericos e codificacao do genero musical.

Treinamos dois modelos: uma Arvore de Decisao, que e mais interpretavel, e uma Rede Neural MLP, que consegue aprender relacoes mais complexas. Avaliamos os dois com acuracia, precisao, recall, F1-score e matriz de confusao.

No resultado final, a Rede Neural MLP teve melhor F1-score, entao foi escolhida como melhor modelo para este dataset. A Arvore de Decisao teve desempenho menor, mas foi util para explicar as regras e a importancia dos atributos.

Com isso, o projeto mostra o processo completo de IA: criacao da base, preparacao dos dados, treinamento, avaliacao, comparacao e conclusao.
