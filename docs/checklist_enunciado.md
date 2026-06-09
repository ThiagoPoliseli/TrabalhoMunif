# Checklist do Enunciado

Checklist baseado no arquivo `enunciado_trabalho_final_ia_2026.pdf`.

| Exigência | Onde está no projeto | Status |
|---|---|---|
| Código-fonte do projeto | `src/` | OK |
| Dataset ou instruções de obtenção | `data/spotify_tracks_sample.csv` e `src/generate_dataset.py` | OK |
| Modelo treinado ou instruções claras | `models/arvore_decisao.pkl` e `models/rede_neural_mlp.pkl` | OK |
| README.md | `README.md` | OK |
| PDF com conteúdo principal do README | `relatorio.pdf` | OK |
| Nome completo e RA dos integrantes | Seção `Integrantes` do README e PDF | PREENCHER |
| Contextualização do tema | README, seção 1 | OK |
| Problema investigado | README, seção 2 | OK |
| Hipótese da equipe | README, seção 3 | OK |
| Origem do dataset | README, seção 4.1 | OK |
| Quantidade de registros/amostras | README, seção 4.3 | OK |
| Principais atributos | README, seção 4.4 | OK |
| Variável alvo | README, seção 4.5 | OK |
| Tratamento/preparação dos dados | README, seção 4.6 | OK |
| Divisão treino/teste | README, seção 4.7 | OK |
| Método da Parte 1 | Árvore de Decisão em `src/main.py` | OK |
| Método da Parte 2 | Rede Neural MLP em `src/main.py` | OK |
| Métricas de avaliação | `results/metricas_modelos.csv` | OK |
| Gráficos de avaliação | `results/*.png` | OK |
| Comparação entre modelos | README, seção 8 e `results/comparacao_modelos.png` | OK |
| Conclusão | README, seção 12 | OK |
| Mostrar treinamento rodando | Comando `python src/main.py` | OK |
| Material para perguntas | `docs/guia_estudo_algoritmos.md` | OK |

## Pendência antes da entrega

Preencher nome completo e RA de todos os integrantes no `README.md` e regenerar o PDF com:

```bash
python src/generate_pdf.py
```

O enunciado avisa que trabalhos sem nome completo e RA no README e no PDF podem perder nota.
