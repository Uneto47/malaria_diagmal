# Detecção Automática de Malária em Esfregaços Sanguíneos

**Trabalho Final - Processamento de Imagens**

**Autores:** Unaldo Neto e Rueslei  
**Ano:** 2026

---

## Resumo

Este trabalho apresenta um sistema automatizado para **detecção de objetos e classificação** de células em esfregaços sanguíneos corados com Giemsa. O sistema realiza a detecção e classificação de células infectadas por malária e hemácias normais, utilizando técnicas de processamento digital de imagens (detecção de bordas com Sobel, segmentação HSV, operações morfológicas e Transformada de Hough Circular).

**Dataset:** [Malaria DiagMal - Roboflow](https://universe.roboflow.com/malariasystem/malaria_diagmal)

**Palavras-chave:** Processamento de Imagens, Malária, Detecção de Objetos, Classificação, Transformada de Hough, Segmentação HSV

---

## 1. Apresentação do Trabalho

### Objetivo

Desenvolver um sistema computacional para **detecção de objetos e classificação automática** de células em imagens de esfregaços sanguíneos, separando:
- **Células infectadas por malária** (classe 1)
- **Hemácias normais** (classe 2)

### Motivação

O diagnóstico microscópico manual de malária:
- Requer profissionais especializados
- É demorado (20-30 minutos por amostra)
- Apresenta variabilidade entre observadores
- Dificulta rastreamento em larga escala

Nossa solução automatiza esse processo, oferecendo:
- Diagnóstico mais rápido
- Padronização dos resultados
- Possibilidade de uso em regiões remotas

---

## 2. Metodologia do Sistema

### Pipeline de Processamento

O sistema processa as imagens em 5 etapas principais:

**1. Pré-processamento**
- Conversão para escala de cinza
- Detecção de bordas com operador Sobel
- Binarização adaptativa
- Operações morfológicas (fechamento e abertura)

**2. Segmentação HSV**
- Conversão RGB → HSV
- Filtragem por tonalidade (Hue): 0.55-0.95 (roxo/azul dos parasitas)
- Filtragem por saturação (Saturation): 0.15-1.0
- Filtragem por valor (Value): 0.1-0.75
- Combinação das máscaras

**3. Classificação e Separação de Classes**
- **Classe 1 - Células infectadas:** isoladas pela máscara HSV (coloração roxa/azulada)
- **Classe 2 - Hemácias normais:** obtidas por subtração das células infectadas

**4. Detecção de Objetos (Transformada de Hough)**
- Células infectadas: raios 75-80 pixels
- Hemácias normais: raios 70-100 pixels
- Remoção de sobreposições entre objetos detectados

**5. Visualização e Quantificação**
- Células infectadas: marcadas com X ciano
- Hemácias normais: marcadas com círculos verdes
- Cálculo da taxa de parasitemia

---

## 3. Conteúdo do Repositório

Este repositório contém todos os arquivos e documentação do trabalho:

### Estrutura dos Arquivos

```
malaria_diagmal/
├── README.md                    # Esta apresentação
├── deteccao_malaria.ipynb       # Notebook Jupyter com implementação completa
│
├── images/
│   ├── input/                   # Imagens do dataset (sem classificação)
│   └── results/                 # Resultados de processamento
│
└── docs/                        # Documentação adicional
```

### Arquivo Principal: Notebook Jupyter

O arquivo `deteccao_malaria.ipynb` contém:

1. **Importação de bibliotecas** (NumPy, Matplotlib, scikit-image)
2. **Definição de parâmetros** configuráveis do sistema
3. **Funções auxiliares** para processamento
4. **Pipeline completo** de detecção com visualizações
5. **Análise de resultados** e métricas
6. **Documentação** em Markdown intercalada

Cada etapa do processamento gera visualizações que são salvas na pasta `images/results/`.

---

## 4. Tecnologias Utilizadas

**Linguagem:** Python 3.8+

**Bibliotecas principais:**
- **NumPy:** manipulação de arrays e operações numéricas
- **Matplotlib:** visualização de resultados
- **scikit-image:** algoritmos de processamento de imagens
  - Filtro Sobel
  - Transformada de Hough
  - Operações morfológicas
  - Conversões de espaço de cor

**Ambiente:** Jupyter Notebook

---

## 5. Resultados

### Visualizações do Pipeline

O processamento gera 6 figuras que documentam cada etapa:

**Figura 1: Imagem Original**
![Original](images/results/01_imagem_original.png)

**Figura 2: Detecção de Bordas (Sobel)**
![Bordas](images/results/02_deteccao_bordas.png)

**Figura 3: Segmentação HSV**
![HSV](images/results/03_segmentacao_hsv.png)

**Figura 4: Processamento de Células**
![Células](images/results/04_processamento_celulas.png)

**Figura 5: Resultado Final**
![Final](images/results/05_resultado_final.png)
*Células infectadas (X ciano) e hemácias normais (círculos verdes)*

**Figura 6: Análise Hough**
![Hough](images/results/06_analise_hough.png)

### Análise de Desempenho

**Células Infectadas (X Ciano):**
- ✅ **Acurácia: ~100%** em quase todos os casos testados
- Falsos positivos: geralmente **+1 célula** por imagem
- Alta precisão devido à segmentação HSV específica

**Hemácias Normais (Círculos Verdes):**
- ⚠️ **Taxa de falsos positivos: ~14%** (média no dataset)
- Exemplo: 105 detecções → 22 falsos positivos + 83 verdadeiros positivos
- Causa: artefatos de iluminação e bordas residuais

**Conclusão:** O sistema é altamente confiável para detectar células infectadas, mas apresenta falsos positivos moderados na identificação de hemácias normais, requerendo refinamento futuro.

---

## 6. Vídeo de Apresentação

**Link:** [A ser adicionado após upload no YouTube]

**Conteúdo:** Demonstração do sistema funcionando, explicação da metodologia e discussão dos resultados.


*Projeto final da disciplina de Processamento de Imagens - 2026*
