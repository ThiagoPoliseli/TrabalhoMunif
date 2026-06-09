# Guia de Estudo dos Algoritmos

Este guia foi feito para ajudar a equipe a explicar o trabalho sem depender apenas da leitura do código.

## 1. Qual é o problema?

O projeto tenta responder:

> É possível prever se uma música será popular ou não popular usando suas características musicais?

A coluna original `popularity` vai de 0 a 100. Para transformar o problema em classificação binária, criamos:

```text
popularidade_alta = 1, quando popularity >= 70
popularidade_alta = 0, quando popularity < 70
```

Assim, o modelo não prevê o número exato de popularidade. Ele prevê a classe: popular ou não popular.

## 2. Quais dados entram no modelo?

Entram atributos numéricos e um atributo categórico:

- Numéricos: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`.
- Categórico: `track_genre`.

O pipeline faz dois tratamentos:

- `StandardScaler`: padroniza atributos numéricos para ficarem em escala comparável.
- `OneHotEncoder`: transforma o gênero musical em colunas binárias, como `track_genre_pop` e `track_genre_rock`.

## 3. Árvore de Decisão

A Árvore de Decisão é o método da Parte 1. Ela aprende regras do tipo:

```text
se danceability > certo valor
e acousticness <= certo valor
então a música tende a ser popular
```

Ela é boa para apresentação porque é interpretável. O arquivo `results/arvore_decisao_visualizacao.png` mostra parte dessas regras.

Pontos para explicar:

- Cada nó faz uma pergunta sobre um atributo.
- Cada divisão tenta separar melhor as classes popular e não popular.
- `max_depth=6` limita a profundidade para evitar uma árvore muito grande e difícil de generalizar.
- A importância dos atributos fica em `results/importancia_atributos_arvore_decisao.png`.

## 4. Rede Neural MLP

A Rede Neural MLP é o método da Parte 2. Ela recebe os atributos tratados e passa os valores por camadas de neurônios artificiais.

Configuração usada:

```python
MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=400,
    random_state=42,
    early_stopping=True
)
```

Como explicar:

- A primeira camada recebe os atributos da música.
- As camadas ocultas aprendem combinações entre os atributos.
- A função ReLU ajuda a rede a aprender relações não lineares.
- O otimizador Adam ajusta os pesos para reduzir o erro.
- `early_stopping=True` para o treinamento quando a validação deixa de melhorar.

O arquivo `results/curva_treinamento_rede_neural_mlp.png` mostra a perda durante o treinamento e a acurácia de validação.

## 5. Métricas

As métricas principais estão em `results/metricas_modelos.csv`.

- Acurácia: proporção total de acertos.
- Precisão: entre as músicas previstas como populares, quantas realmente eram populares.
- Recall: entre as músicas realmente populares, quantas o modelo conseguiu encontrar.
- F1-score: equilíbrio entre precisão e recall.
- Matriz de confusão: mostra acertos e erros separados por classe.

Neste projeto, o F1-score foi usado como critério principal porque ele equilibra precisão e recall.

## 6. Como apresentar o resultado

Uma boa explicação curta:

> A Árvore de Decisão foi mais fácil de interpretar, pois mostra regras visuais. A Rede Neural MLP teve melhor desempenho geral pelo F1-score, pois conseguiu aprender combinações mais complexas entre os atributos musicais. Por isso, para este dataset, a MLP foi escolhida como melhor modelo.

## 7. Perguntas prováveis do professor

**Por que vocês transformaram popularidade em 0 e 1?**  
Para resolver como classificação binária e comparar modelos por acurácia, precisão, recall, F1-score e matriz de confusão.

**Por que usar `StandardScaler`?**  
Porque a MLP é sensível à escala dos atributos. `tempo`, `duration_ms` e atributos entre 0 e 1 precisam ficar em escala comparável.

**Por que usar `OneHotEncoder`?**  
Porque `track_genre` é texto. O modelo precisa receber números, então cada gênero vira uma coluna binária.

**Qual modelo foi melhor?**  
A Rede Neural MLP, considerando o F1-score.

**Por que a árvore ainda é útil?**  
Porque ela é mais interpretável e ajuda a explicar quais atributos influenciaram a decisão.

**O dataset é real?**  
Ele é sintético, criado pela equipe, mas inspirado em atributos comuns de bases públicas do Spotify. O enunciado permite dataset próprio.
