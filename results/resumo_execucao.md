# Resumo da Execucao

## Resultado principal
O melhor modelo pelo F1-score foi `rede_neural_mlp` com F1-score de 0.6127.

## Comparacao rapida

| Modelo | Acuracia | Precisao | Recall | F1-score | VP | VN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arvore_decisao | 0.6992 | 0.6010 | 0.5165 | 0.5556 | 235 | 639 | 156 | 220 |
| rede_neural_mlp | 0.7472 | 0.6925 | 0.5495 | 0.6127 | 250 | 684 | 111 | 205 |

## Leitura para apresentacao
A Arvore de Decisao e mais facil de explicar porque transforma os dados em regras. Neste experimento, ela atingiu F1-score de 0.5556.
A Rede Neural MLP conseguiu desempenho melhor porque aprende combinacoes nao lineares entre os atributos. Neste experimento, ela atingiu F1-score de 0.6127.

## Atributos mais importantes na Arvore de Decisao

| Atributo | Importancia |
|---|---:|
| danceability | 0.2517 |
| energy | 0.2126 |
| tempo | 0.1210 |
| acousticness | 0.1002 |
| valence | 0.0971 |

## Arquivos gerados

- `results/metricas_modelos.csv`
- `results/comparacao_modelos.png`
- `results/matriz_confusao_arvore_decisao.png`
- `results/matriz_confusao_rede_neural_mlp.png`
- `results/importancia_atributos_arvore_decisao.png`
- `results/curva_treinamento_rede_neural_mlp.png`
