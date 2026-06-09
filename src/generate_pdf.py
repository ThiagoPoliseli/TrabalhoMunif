"""Gera o relatorio PDF com o conteudo principal do README."""
from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT_DIR / "relatorio.pdf"
RESULTS_DIR = ROOT_DIR / "results"
METRICS_PATH = RESULTS_DIR / "metricas_modelos.csv"

MODEL_LABELS = {
    "arvore_decisao": "Árvore de Decisão",
    "rede_neural_mlp": "Rede Neural MLP",
}


def add_title(story, text, styles):
    story.append(Paragraph(text, styles["TitleCustom"]))
    story.append(Spacer(1, 0.35 * cm))


def add_heading(story, text, styles):
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph(text, styles["HeadingCustom"]))
    story.append(Spacer(1, 0.15 * cm))


def add_body(story, text, styles):
    story.append(Paragraph(text, styles["BodyCustom"]))
    story.append(Spacer(1, 0.12 * cm))


def add_image(story, filename, caption, styles, width=15.5 * cm, height_ratio=0.62):
    image_path = RESULTS_DIR / filename
    if image_path.exists():
        story.append(Spacer(1, 0.2 * cm))
        story.append(Image(str(image_path), width=width, height=width * height_ratio))
        story.append(Paragraph(caption, styles["CaptionCustom"]))
        story.append(Spacer(1, 0.25 * cm))


def load_metrics_table():
    table = [["Modelo", "Acurácia", "Precisão", "Recall", "F1-score"]]
    if not METRICS_PATH.exists():
        return table

    with METRICS_PATH.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            table.append([
                MODEL_LABELS.get(row["modelo"], row["modelo"]),
                f"{float(row['acuracia']):.4f}",
                f"{float(row['precisao']):.4f}",
                f"{float(row['recall']):.4f}",
                f"{float(row['f1_score']):.4f}",
            ])
    return table


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
    )

    base = getSampleStyleSheet()
    styles = {
        "TitleCustom": ParagraphStyle(
            "TitleCustom",
            parent=base["Title"],
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            spaceAfter=12,
        ),
        "HeadingCustom": ParagraphStyle(
            "HeadingCustom",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#111827"),
        ),
        "BodyCustom": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            alignment=0,
            spaceAfter=4,
        ),
        "CaptionCustom": ParagraphStyle(
            "CaptionCustom",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4B5563"),
        ),
    }

    story = []

    add_body(story, "Disciplina de Inteligência Artificial , Professor Munif , Unicesumar 2026", styles)
    add_title(story, "Trabalho Final - Predição de Popularidade de Músicas Estilo Spotify", styles)

    add_heading(story, "Integrantes", styles)
    add_body(story, "Thiago Poliseli Silva - RA: PREENCHER", styles)
    add_body(story, "Integrante 2 - RA: PREENCHER", styles)
    add_body(story, "Integrante 3 - RA: PREENCHER", styles)
    add_body(story, "Integrante 4 - RA: PREENCHER", styles)

    add_heading(story, "Resumo do Projeto", styles)
    add_body(story, "O projeto aplica Inteligência Artificial para prever se uma música será popular ou não popular com base em características musicais inspiradas em datasets do Spotify. Foram utilizados atributos como danceability, energy, loudness, acousticness, valence, tempo, duration_ms e track_genre.", styles)
    add_body(story, "O problema foi tratado como classificação binária. A variável alvo popularidade_alta recebe valor 1 quando popularity >= 70 e valor 0 quando popularity < 70.", styles)

    add_heading(story, "Contextualização, Problema e Hipótese", styles)
    add_body(story, "Plataformas de streaming armazenam grandes volumes de dados sobre músicas. A hipótese da equipe é que músicas com maior energia, dançabilidade, valência e intensidade sonora tendem a ter maior popularidade. Assim, modelos de IA podem identificar padrões nos atributos musicais e classificar músicas como populares ou não populares.", styles)

    add_heading(story, "Dataset", styles)
    add_body(story, "O dataset foi criado pela equipe por meio do script src/generate_dataset.py, com 5.000 registros e 15 colunas. A base é sintética, mas segue a estrutura de datasets musicais estilo Spotify. O enunciado permite a criação de dataset próprio, o que torna a execução reprodutível e sem dependência de download externo.", styles)
    add_body(story, "Principais atributos: track_genre, popularity, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo e duration_ms.", styles)

    add_heading(story, "Preparação dos Dados", styles)
    add_body(story, "As etapas realizadas foram: carregamento do CSV, validação das colunas obrigatórias, remoção de registros ausentes, criação da variável alvo, padronização dos atributos numéricos com StandardScaler, codificação do gênero musical com OneHotEncoder e divisão estratificada entre treino e teste.", styles)
    add_body(story, "Divisão: 75% para treino, totalizando 3.750 amostras, e 25% para teste, totalizando 1.250 amostras.", styles)

    add_heading(story, "Métodos de IA Utilizados", styles)
    add_body(story, "Método da Parte 1: Árvore de Decisão, usando DecisionTreeClassifier(max_depth=6, random_state=42). O modelo é interpretável e permite visualizar regras de decisão.", styles)
    add_body(story, "Método da Parte 2: Rede Neural MLP, usando MLPClassifier com camadas ocultas (32, 16), ativação ReLU, otimizador Adam, early stopping e random_state=42. O modelo consegue aprender relações não lineares entre os atributos.", styles)

    add_heading(story, "Avaliação dos Modelos", styles)
    add_body(story, "As métricas utilizadas foram acurácia, precisão, recall, F1-score e matriz de confusão. Também foram gerados gráficos de distribuição da base e comparação dos modelos.", styles)

    add_image(story, "distribuicao_popularidade.png", "Figura 1 - Distribuição da popularidade das músicas.", styles)
    add_image(story, "distribuicao_classes.png", "Figura 2 - Distribuição das classes popular e não popular.", styles)

    story.append(PageBreak())
    add_heading(story, "Matrizes de Confusão", styles)
    add_image(story, "matriz_confusao_arvore_decisao.png", "Figura 3 - Matriz de confusão da Árvore de Decisão.", styles)
    add_image(story, "matriz_confusao_rede_neural_mlp.png", "Figura 4 - Matriz de confusão da Rede Neural MLP.", styles)

    add_heading(story, "Tabela de Métricas", styles)
    data = load_metrics_table()
    table = Table(data, hAlign="LEFT", colWidths=[5 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9CA3AF")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.25 * cm))

    add_image(story, "comparacao_modelos.png", "Figura 5 - Comparação gráfica entre os modelos.", styles)
    add_image(story, "importancia_atributos_arvore_decisao.png", "Figura 6 - Atributos mais importantes na Árvore de Decisão.", styles, height_ratio=0.66)
    add_image(story, "curva_treinamento_rede_neural_mlp.png", "Figura 7 - Curva de treinamento da Rede Neural MLP.", styles, height_ratio=0.38)

    story.append(PageBreak())
    add_heading(story, "Comparação dos Resultados", styles)
    add_body(story, "A Rede Neural MLP apresentou melhor desempenho geral em comparação com a Árvore de Decisão. O modelo MLP obteve maior acurácia, precisão e F1-score. A Árvore de Decisão teve desempenho inferior, mas possui a vantagem de ser mais interpretável.", styles)
    add_body(story, "Com base no F1-score, que equilibra precisão e recall, o melhor modelo foi a Rede Neural MLP.", styles)

    add_heading(story, "Modelo Treinado", styles)
    add_body(story, "Os modelos treinados foram salvos na pasta models/: arvore_decisao.pkl e rede_neural_mlp.pkl. Eles podem ser carregados com a biblioteca joblib.", styles)

    add_heading(story, "Material de Estudo e Apresentação", styles)
    add_body(story, "A pasta docs/ contém uma checklist do enunciado, um guia de estudo dos algoritmos e um roteiro sugerido para a apresentação.", styles)
    add_body(story, "Arquivos: docs/checklist_enunciado.md, docs/guia_estudo_algoritmos.md e docs/roteiro_apresentacao.md.", styles)

    add_heading(story, "Como Executar", styles)
    add_body(story, "1. Criar ambiente virtual: python -m venv .venv", styles)
    add_body(story, "2. Instalar dependências: pip install -r requirements.txt", styles)
    add_body(story, "3. Gerar dataset: python src/generate_dataset.py", styles)
    add_body(story, "4. Treinar e avaliar: python src/main.py", styles)
    add_body(story, "5. Testar predição: python src/predict.py", styles)
    add_body(story, "6. Gerar PDF do relatório: python src/generate_pdf.py", styles)

    add_heading(story, "Conclusão", styles)
    add_body(story, "O projeto demonstrou o processo completo de criação de uma solução baseada em IA: definição do problema, criação e preparação dos dados, treinamento dos modelos, avaliação com métricas e gráficos, comparação e conclusão.", styles)
    add_body(story, "A Rede Neural MLP foi o modelo mais adequado para este dataset, alcançando F1-score de 0.6127 contra 0.5556 da Árvore de Decisão. Apesar disso, a Árvore de Decisão é útil para fins de interpretação e explicação acadêmica.", styles)

    doc.build(story)
    print(f"PDF gerado em: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
