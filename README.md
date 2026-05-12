# Classificador Hierárquico de Espécies e Raças com Deep Learning

Sistema de Inteligência Artificial desenvolvido em Python para classificação automática de espécies e raças de animais utilizando Deep Learning com a arquitetura ResNet50 e TensorFlow.

O projeto utiliza dois modelos separados:

Modelo 1 → classificação de espécie (cachorro ou gato)
Modelo 2 → classificação de raça

## Objetivo do Projeto

O objetivo deste projeto é criar um sistema capaz de identificar automaticamente animais em imagens, classificando:

A espécie do animal
A raça correspondente

O sistema foi desenvolvido utilizando técnicas modernas de Visão Computacional e Deep Learning, aplicando Transfer Learning com ResNet50.

- Tecnologias Utilizadas
- Python
- TensorFlow
- Keras
- ResNet50
- NumPy
- Matplotlib
- Scikit-Learn
  
## Conceitos Aplicados

Este projeto utiliza diversos conceitos importantes da área de Inteligência Artificial:

- Deep Learning
- Redes Neurais Convolucionais (CNN)
- Transfer Learning
- Data Augmentation
- Classificação Hierárquica
- Batch Normalization
- Dropout
- Early Stopping
- Reduce Learning Rate on Plateau
## Estrutura do Projeto
    dataset/
    │
    ├── train/
    │   ├── cachorro/
    │   │  
    │   ├── coelho/
    │   │   
    │   │
    │   └── gato/
    │  │
    └── test/
        (Arquivos que serão testados)

## Funcionamento do Sistema
Modelo de Espécie

- O primeiro modelo identifica a espécie do animal

Modelo de Raça

- O segundo modelo identifica a raça do animal.

## Fluxo do Sistema
    Imagem →
        Modelo de Espécie →
              Modelo de Raça →
                  Resultado Final

## Arquitetura Utilizada

  O projeto utiliza a arquitetura pré-treinada ResNet50 com Transfer Learning.

  A ResNet50 foi treinada originalmente no dataset ImageNet e reutilizada como extratora de características.

## O modelo utiliza:

- Camadas Dense
- Dropout
- BatchNormalization
- Softmax para classificação
- Data Augmentation

## Para melhorar a capacidade de generalização da IA, o projeto aplica:

- Rotação
- Zoom
- Espelhamento horizontal
- Deslocamento horizontal e vertical
= Shear transformation

### Isso ajuda a reduzir overfitting.

## Como Executar o Projeto
1. Clone este repositório.
3. Instale as dependências

       pip install tensorflow matplotlib numpy scikit-learn pillow


## Treinamento

Durante o treinamento o sistema:

- Cria os modelos
- Carrega as imagens
- Aplica Data Augmentation
- Treina a IA
- Salva os modelos
- Gera gráficos de desempenho
- Modelos Gerados

## Após o treinamento:

- modelo_especie.h5
- modelo_raca.h5

## Também são gerados arquivos JSON contendo:

- classes
- quantidade de classes
- data do treinamento
- tamanho das imagens
- número de épocas
- Relatórios Gerados

## O projeto salva:

- Resultados do treinamento
resultados_treinamento.png
- Relatório de classificação
resultados_classificacao.json
- Técnicas de Otimização Utilizadas
EarlyStopping

### Interrompe o treinamento automaticamente caso o modelo pare de melhorar.

ReduceLROnPlateau

Reduz automaticamente a taxa de aprendizado quando necessário.

## Dataset Utilizado

 O modelo foi treinado utilizando o dataset Oxford-IIIT Pet Dataset, amplamente utilizado em pesquisas acadêmicas e projetos de Visão Computacional e Deep Learning.

## O dataset contém:

- 7.387 imagens
- 37 raças diferentes
   - 25 raças de cachorros
  - 12 raças de gatos

As imagens possuem diferentes:

- poses
- iluminações
- fundos
- ângulos
- tamanhos

  Isso torna o treinamento mais robusto e ajuda o modelo a aprender características reais dos animais.

## O sistema é capaz de:

- Identificar espécies automaticamente
- Classificar raças
- Calcular acurácia
- Gerar relatórios automáticos
- Aplicações do Projeto

## Este projeto pode ser utilizado em:

- Sistemas veterinários
- Aplicativos de adoção
- Monitoramento animal
- Sistemas acadêmicos
- Estudos de Visão Computacional
- Projetos de Inteligência Artificial

### Projeto desenvolvido para estudos e aplicações de Deep Learning e Visão Computacional utilizando Python e TensorFlow.
