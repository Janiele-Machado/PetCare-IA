# Classificador Hierárquico de Espécies e Raças com YOLOv8

Sistema de Inteligência Artificial desenvolvido em Python para classificação automática de espécies e raças de animais, com **detecção visual por bounding box**, utilizando a arquitetura **YOLOv8** da Ultralytics.

O projeto utiliza três modelos:

- **Modelo 1** → detecção do animal na imagem (bounding box)
- **Modelo 2** → classificação de espécie (cachorro ou gato)
- **Modelo 3** → classificação de raça

---

## Objetivo do Projeto

Identificar automaticamente animais em imagens, classificando a espécie e a raça, e **plotar cada imagem testada** com:

- Bounding box desenhado ao redor do animal detectado
- Gráfico Top-5 de espécies com percentuais de confiança
- Gráfico Top-5 de raças com percentuais de confiança

---

## Tecnologias Utilizadas

- Python 3.12
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- PyTorch
- NumPy
- Matplotlib
- Pillow

### Instalação das dependências

```bash
pip install ultralytics matplotlib numpy pillow
```

---

## Conceitos Aplicados

- Deep Learning
- Redes Neurais Convolucionais (CNN)
- Transfer Learning (a partir de `yolov8n-cls.pt` pré-treinado no ImageNet)
- Object Detection com bounding box
- Classificação Hierárquica
- Data Augmentation

---

## Estrutura do Projeto

```
dataset/
├── train/
│   ├── cachorro/
│   │   ├── chihuahua/
│   │   ├── golden_retriever/
│   │   └── ...
│   ├── gato/
│   │   ├── siamese/
│   │   ├── sphynx/
│   │   └── ...
│   └── coelho/
│       └── ...
└── test/
    ├── cachorro/
    │   └── chihuahua/
    │       └── imagem.jpg
    └── (ou diretamente imagens na raiz)

runs/classify/                  ← gerado automaticamente pelo Ultralytics
├── modelo_especie_yolo/
│   └── treino/weights/best.pt
└── modelo_raca_yolo/
    └── treino/weights/best.pt

modelo_especie_info.json
modelo_raca_info.json
resultados_classificacao.json
resultado_001_<nome>.png        ← plots gerados por imagem testada
```

---

## Funcionamento do Sistema

### Fluxo completo

```
Imagem
  └─▶ yolov8n.pt (detecção geral)
          └─▶ Bounding box desenhado
  └─▶ modelo_especie_yolo (classificação)
          └─▶ Espécie + Top-5 com confiança
  └─▶ modelo_raca_yolo (classificação)
          └─▶ Raça + Top-5 com confiança
  └─▶ Plot salvo em PNG
```

### Modelo de Detecção (`yolov8n.pt`)

Utiliza o modelo geral do YOLOv8 treinado no COCO para **localizar o animal na imagem** e desenhar o bounding box com o label e a confiança — exatamente como visto em exemplos clássicos de detecção de objetos.

### Modelo de Espécie

Classifica a espécie do animal identificado (ex.: cachorro, gato, coelho).

### Modelo de Raça

Classifica a raça dentro das 37 raças disponíveis no dataset.

---

## Como Executar

1. Clone este repositório
2. Instale as dependências:
```bash
pip install ultralytics matplotlib numpy pillow
```
3. Execute o script principal:
```bash
python classificador_yolov8.py
```
4. Escolha uma opção no menu:
```
1 - Treinar modelos
2 - Testar e plotar TODAS as imagens da pasta TEST
3 - Ver métricas do último treinamento
4 - Sair
```

---

## Treinamento

Durante o treinamento o sistema:

- Reorganiza o dataset em pastas temporárias separadas por espécie e por raça
- Aplica Data Augmentation (rotação, zoom, flip horizontal, shear, deslocamento)
- Treina dois modelos YOLOv8 de classificação com Transfer Learning
- Salva os pesos do melhor modelo (`best.pt`) automaticamente
- Gera gráficos de acurácia e loss (`results.png`)
- Salva os metadados de cada modelo em JSON

### Configurações padrão

| Parâmetro | Valor |
|---|---|
| Épocas | 30 |
| Tamanho da imagem | 224 × 224 |
| Batch size | 16 |
| Learning rate | 1e-3 |
| Divisão treino/val | 80% / 20% |
| Backbone | yolov8n-cls.pt |

---

## Arquivos Gerados

| Arquivo | Descrição |
|---|---|
| `runs/classify/modelo_especie_yolo/treino/weights/best.pt` | Pesos do modelo de espécie |
| `runs/classify/modelo_raca_yolo/treino/weights/best.pt` | Pesos do modelo de raça |
| `modelo_especie_info.json` | Metadados do modelo de espécie |
| `modelo_raca_info.json` | Metadados do modelo de raça |
| `resultado_001_<nome>.png` | Plot de cada imagem testada |
| `resultados_classificacao.json` | Relatório completo das classificações |

---

## Exemplo de Plot Gerado

Cada imagem testada gera um plot com três painéis lado a lado:

- **Esquerda:** imagem original com bounding box verde e label de detecção
- **Centro:** gráfico horizontal Top-5 de espécies com percentuais de confiança
- **Direita:** gráfico horizontal Top-5 de raças com percentuais de confiança

---

## Dataset Utilizado

O modelo foi treinado com o **Oxford-IIIT Pet Dataset**, amplamente utilizado em pesquisas de Visão Computacional.

| Característica | Detalhe |
|---|---|
| Total de imagens | 7.387 |
| Raças | 37 (25 de cachorro, 12 de gato) |
| Variações | poses, iluminações, fundos, ângulos e tamanhos distintos |

---

## Aplicações do Projeto

- Sistemas veterinários
- Aplicativos de adoção de animais
- Monitoramento animal
- Estudos acadêmicos de Visão Computacional
- Projetos de Inteligência Artificial

---

Projeto desenvolvido para estudos e aplicações de Deep Learning e Visão Computacional utilizando Python e YOLOv8.

