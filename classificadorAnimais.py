"""
PROJETO: CLASSIFICADOR HIERÁRQUICO 
Dois modelos separados: um para espécie, outro para raça
"""

# ============================================
# Versão do Python: 3.12 para ser compatível com TensorFlow 2.13
# ============================================

# ============================================
# INSTALAR A DEPENDÊNCIA (TENSORFLOW)
# pip install tensorflow matplotlib numpy pillow
# py -m pip install scipy
# py -m pip install scikit-learn
# ============================================

# ============================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
import shutil
import tempfile
from sklearn.model_selection import train_test_split

# Configurações
DATASET_PATH = "dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

IMG_SIZE = 224
BATCH_SIZE = 16
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 15
NUM_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_AT = 100

MODEL_ESPECIE_PATH = "modelo_especie.h5"
MODEL_RACA_PATH = "modelo_raca.h5"

# ============================================
# FUNÇÃO 1: CRIAR MODELO (GENÉRICO)
# ============================================

def criar_modelo(num_classes, nome_modelo):
    """Cria um modelo ResNet50 para classificação"""
    
    print(f"\nCriando modelo para {nome_modelo} com {num_classes} classes...")
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    # Congelar a base inicialmente e treinar apenas a cabeça
    base_model.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================
# FUNÇÃO 2: PREPARAR DADOS PARA ESPÉCIE
# ============================================

def preparar_dados_especie(train_path):
    """Prepara dados para classificação de espécie  """
    
    print("\n" + "="*60)
    print("PREPARANDO DADOS PARA ESPÉCIE")
    print("="*60)
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% para validação
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    especies = list(train_generator.class_indices.keys())
    
    print(f"✓ Espécies: {especies}")
    print(f"✓ Treino: {train_generator.samples} imagens")
    print(f"✓ Validação: {val_generator.samples} imagens")
    
    return train_generator, val_generator, especies

# ============================================
# FUNÇÃO 3: PREPARAR DADOS PARA RAÇA
# ============================================

def preparar_dados_raca(train_path):
    """Prepara dados para classificação de raça (todas as subpastas)"""
    
    print("\n" + "="*60)
    print("PREPARANDO DADOS PARA RAÇA")
    print("="*60)
    
    # Criar estrutura temporária com todas as raças (ignorando espécie)
    temp_path = os.path.join(tempfile.gettempdir(), "dataset_racas")
    
    # Limpar pasta temporária se existir
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.makedirs(temp_path)
    
    # Mapear todas as raças para pastas temporárias
    racas = []
    
    for especie in os.listdir(train_path):
        especie_path = os.path.join(train_path, especie)
        if not os.path.isdir(especie_path):
            continue
        
        for raca in os.listdir(especie_path):
            raca_path = os.path.join(especie_path, raca)
            if not os.path.isdir(raca_path):
                continue
            
            racas.append(raca)
            
            # Criar pasta da raça no diretório temporário
            temp_raca_path = os.path.join(temp_path, raca)
            os.makedirs(temp_raca_path, exist_ok=True)
            
            # Copiar imagens
            for img in os.listdir(raca_path):
                if img.endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(raca_path, img)
                    dst = os.path.join(temp_raca_path, img)
                    shutil.copy2(src, dst)
    
    # Dividir em treino/validação
    imagens = []
    for raca in racas:
        raca_path = os.path.join(temp_path, raca)
        for img in os.listdir(raca_path):
            imagens.append((os.path.join(raca_path, img), raca))
    
    # Dividir 80/20
    train_imgs, val_imgs = train_test_split(imagens, test_size=0.2, random_state=42, stratify=[r for _, r in imagens])
    
    # Criar estrutura temporária de treino
    train_temp = os.path.join(tempfile.gettempdir(), "dataset_racas_train")
    val_temp = os.path.join(tempfile.gettempdir(), "dataset_racas_val")
    
    for path in [train_temp, val_temp]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    
    for img_path, raca in train_imgs:
        raca_train_path = os.path.join(train_temp, raca)
        os.makedirs(raca_train_path, exist_ok=True)
        shutil.copy2(img_path, os.path.join(raca_train_path, os.path.basename(img_path)))
    
    for img_path, raca in val_imgs:
        raca_val_path = os.path.join(val_temp, raca)
        os.makedirs(raca_val_path, exist_ok=True)
        shutil.copy2(img_path, os.path.join(raca_val_path, os.path.basename(img_path)))
    
    # Criar geradores COM data augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    train_generator = train_datagen.flow_from_directory(
        train_temp,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_temp,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    racas_lista = list(train_generator.class_indices.keys())
    
    print(f"✓ Raças: {len(racas_lista)}")
    print(f"✓ Treino: {train_generator.samples} imagens")
    print(f"✓ Validação: {val_generator.samples} imagens")
    
    return train_generator, val_generator, racas_lista

# ============================================
# FUNÇÃO 4: TREINAR MODELO
# ============================================

def treinar_modelo(model, train_gen, val_gen, nome):
    """Treina um modelo em duas fases: cabeça + fine-tuning."""
    
    print(f"\n{'='*60}")
    print(f"TREINANDO MODELO: {nome}")
    print(f"{'='*60}")
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6, verbose=1)
    ]
    
    print(f"\nFase 1: treinando apenas a cabeça por {INITIAL_EPOCHS} épocas...")
    history_head = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning: descongelar as últimas camadas da base e treinar com learning rate menor
    print(f"\nFase 2: fine-tuning das últimas {FINE_TUNE_AT} camadas por {FINE_TUNE_EPOCHS} épocas...")
    model.trainable = True
    for layer in model.layers[:-FINE_TUNE_AT]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_finetune = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=NUM_EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Unir histórico das duas fases
    class CombinedHistory:
        def __init__(self, h1, h2):
            self.history = {}
            for key in h1.history:
                self.history[key] = h1.history[key] + h2.history[key]
    
    return CombinedHistory(history_head, history_finetune)

# ============================================
# FUNÇÃO 5: SALVAR MODELO
# ============================================

def salvar_modelo(model, path, classes, class_indices=None):
    """Salva modelo e metadados"""
    
    model.save(path)
    
    # Se class_indices for fornecido, criar mapeamento correto (índice -> classe)
    if class_indices is not None:
        # Criar dicionário invertido: índice -> nome_classe
        classes_ordenadas = [''] * len(class_indices)
        for classe_nome, idx in class_indices.items():
            classes_ordenadas[idx] = classe_nome
        classes = classes_ordenadas
    else:
        classes = list(classes)
    
    info = {
        'classes': classes,
        'num_classes': len(classes),
        'data': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'tamanho_imagem': IMG_SIZE,
        'epocas': NUM_EPOCHS
    }
    
    info_path = path.replace('.h5', '_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"✓ Modelo salvo: {path}")
    print(f"✓ Info salva: {info_path}")

# ============================================
# FUNÇÃO 6: CLASSIFICAR TODAS IMAGENS DA PASTA TEST
# ============================================

def classificar_todas_imagens_test():
    """
    Classifica automaticamente TODAS as imagens da pasta test
    Mostra o resultado de cada uma individualmente
    Suporta duas estruturas: 
    1. dataset/test/especie/raca/imagem.jpg
    2. dataset/test/imagem.jpg (extrai raça do nome)
    """
    
    print("\n" + "="*60)
    print("CLASSIFICANDO TODAS AS IMAGENS DA PASTA TEST")
    print("="*60)
    
    # Verificar se os modelos existem
    if not os.path.exists(MODEL_ESPECIE_PATH) or not os.path.exists(MODEL_RACA_PATH):
        print(" Modelos não encontrados. Treine primeiro (opção 1)!")
        return
    
    # Verificar se a pasta test existe
    if not os.path.exists(TEST_PATH):
        print(f" Pasta de teste não encontrada: {TEST_PATH}")
        return
    
    # Carregar modelos uma vez
    print("\nCarregando modelos...")
    model_especie = load_model(MODEL_ESPECIE_PATH)
    model_raca = load_model(MODEL_RACA_PATH)
    
    with open(MODEL_ESPECIE_PATH.replace('.h5', '_info.json'), 'r') as f:
        info_especie = json.load(f)
    with open(MODEL_RACA_PATH.replace('.h5', '_info.json'), 'r') as f:
        info_raca = json.load(f)
    
    # Coletar todas as imagens da pasta test
    imagens = []
    
    # Verificar se há subpastas (estrutura esperada)
    tem_subpastas = any(os.path.isdir(os.path.join(TEST_PATH, item)) for item in os.listdir(TEST_PATH))
    
    if tem_subpastas:
        # Estrutura: dataset/test/especie/raca/imagem.jpg
        for especie in os.listdir(TEST_PATH):
            especie_path = os.path.join(TEST_PATH, especie)
            if not os.path.isdir(especie_path):
                continue
            
            for raca in os.listdir(especie_path):
                raca_path = os.path.join(especie_path, raca)
                if not os.path.isdir(raca_path):
                    continue
                
                for imagem in os.listdir(raca_path):
                    if imagem.lower().endswith(('.jpg', '.jpeg', '.png')):
                        imagens.append({
                            'caminho': os.path.join(raca_path, imagem),
                            'especie_verdadeira': especie,
                            'raca_verdadeira': raca,
                            'nome': imagem
                        })
    else:
        # Estrutura: dataset/test/imagem.jpg (extrai raça do nome)
        for imagem in os.listdir(TEST_PATH):
            imagem_path = os.path.join(TEST_PATH, imagem)
            if os.path.isfile(imagem_path) and imagem.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Tentar extrair raça do nome (assumindo padrão: raca_numero.jpg)
                nome_sem_ext = os.path.splitext(imagem)[0]
                partes = nome_sem_ext.rsplit('_', 1)
                raca_do_nome = partes[0] if len(partes) > 1 else nome_sem_ext
                
                imagens.append({
                    'caminho': imagem_path,
                    'especie_verdadeira': 'desconhecido',
                    'raca_verdadeira': raca_do_nome,
                    'nome': imagem
                })
    
    if len(imagens) == 0:
        print("\n Nenhuma imagem encontrada na pasta test!")
        print("   Estrutura esperada (opção 1):")
        print("      dataset/test/cachorro/pitbull/imagem.jpg")
        print("   ou (opção 2):")
        print("      dataset/test/imagem.jpg (será extraída a raça do nome)")
        return
    
    print(f"\n Encontradas {len(imagens)} imagem(ns) para classificar")
    print("="*60)
    
    # Classificar cada imagem
    resultados = []
    contador = 0
    
    for img_info in imagens:
        contador += 1
        print(f"\n{'#'*50}")
        print(f" IMAGEM {contador}/{len(imagens)}: {img_info['nome']}")
        print(f"   Caminho: {img_info['especie_verdadeira']}/{img_info['raca_verdadeira']}/{img_info['nome']}")
        print(f"{'#'*50}")
        
        # Processar imagem (mesmo pré-processamento do treinamento)
        img = load_img(img_info['caminho'], target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Predizer espécie
        pred_especie = model_especie.predict(img_array, verbose=0)
        especie_idx = np.argmax(pred_especie[0])
        especie_predita = info_especie['classes'][especie_idx]
        conf_especie = pred_especie[0][especie_idx]
        
        # Predizer raça
        pred_raca = model_raca.predict(img_array, verbose=0)
        raca_idx = np.argmax(pred_raca[0])
        raca_predita = info_raca['classes'][raca_idx]
        conf_raca = pred_raca[0][raca_idx]
        
        # Mostrar resultado formatado
        print("\n" + "-"*40)
        print("RESULTADO DA CLASSIFICAÇÃO")
        print("-"*40)
        print(f"\nESPÉCIE PREVISTA: {especie_predita}")
        print(f"  Probabilidade de acerto da espécie: {conf_especie*100:.2f}%")
        print(f"\nRAÇA PREVISTA: {raca_predita}")
        print(f"  Probabilidade de acerto da raça: {conf_raca*100:.2f}%")
        print("-"*40)
        
        # Salvar resultado
        resultados.append({
            'imagem': img_info['nome'],
            'caminho': img_info['caminho'],
            'especie_predita': especie_predita,
            'raca_predita': raca_predita,
            'confianca_especie': float(conf_especie),
            'confianca_raca': float(conf_raca)
        })
    
    # Resumo final
    print("\n" + "="*60)
    print("RESUMO FINAL")
    print("="*60)
    
    total = len(resultados)
    print(f"\n Total de imagens classificadas: {total}")
    print(f"\n Classificações concluídas com sucesso!")
    
    # Salvar relatório
    relatorio_path = "resultados_classificacao.json"
    with open(relatorio_path, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
    print(f"\n Relatório detalhado salvo em: {relatorio_path}")
    print("="*60)

# ============================================
# FUNÇÃO 7: VISUALIZAR TREINAMENTO
# ============================================

def visualizar_treinamento(history_especie, history_raca):
    """Plota gráficos dos treinamentos"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Espécie
    axes[0, 0].plot(history_especie.history['accuracy'], label='Treino')
    axes[0, 0].plot(history_especie.history['val_accuracy'], label='Validação')
    axes[0, 0].set_title('Acurácia - Espécie')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history_especie.history['loss'], label='Treino')
    axes[0, 1].plot(history_especie.history['val_loss'], label='Validação')
    axes[0, 1].set_title('Loss - Espécie')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Raça
    axes[1, 0].plot(history_raca.history['accuracy'], label='Treino')
    axes[1, 0].plot(history_raca.history['val_accuracy'], label='Validação')
    axes[1, 0].set_title('Acurácia - Raça')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history_raca.history['loss'], label='Treino')
    axes[1, 1].plot(history_raca.history['val_loss'], label='Validação')
    axes[1, 1].set_title('Loss - Raça')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('resultados_treinamento.png')
    plt.show()

# ============================================
# FUNÇÃO PRINCIPAL
# ============================================

def main():
    print("\n" + "="*60)
    print("CLASSIFICADOR HIERÁRQUICO - 2 MODELOS SEPARADOS")
    
    while True:
        print("\n" + "="*60)
        print("MENU PRINCIPAL")
        print("="*60)
        print("1 - Treinar modelos")
        print("2 - Testar TODAS as imagens da pasta TEST")
        print("3 - Sair")
        
        opcao = input("\nEscolha (1-3): ")
        
        if opcao == '1':
            try:
                # Treinar modelo de ESPÉCIE
                print("\n" + "-"*30)
                print("TREINANDO CLASSIFICADOR DE ESPÉCIE")
                print("-"*30)
                
                train_gen_esp, val_gen_esp, especies = preparar_dados_especie(TRAIN_PATH)
                model_esp = criar_modelo(len(especies), "ESPÉCIE")
                history_esp = treinar_modelo(model_esp, train_gen_esp, val_gen_esp, "ESPÉCIE")
                salvar_modelo(model_esp, MODEL_ESPECIE_PATH, especies, train_gen_esp.class_indices)
                
                # Treinar modelo de RAÇA
                print("\n" + "-"*30)
                print("TREINANDO CLASSIFICADOR DE RAÇA")
                print("-"*30)
                
                train_gen_raca, val_gen_raca, racas = preparar_dados_raca(TRAIN_PATH)
                model_raca = criar_modelo(len(racas), "RAÇA")
                history_raca = treinar_modelo(model_raca, train_gen_raca, val_gen_raca, "RAÇA")
                salvar_modelo(model_raca, MODEL_RACA_PATH, racas, train_gen_raca.class_indices)
                
                # Visualizar resultados
                visualizar_treinamento(history_esp, history_raca)
                
                print("\n" + "="*60)
                print(" MODELOS TREINADOS COM SUCESSO!")
                print(f"   - Espécies: {len(especies)}")
                print(f"   - Raças: {len(racas)}")
                print(f"   - Épocas: {NUM_EPOCHS}")
                print("="*60)
                
            except Exception as e:
                print(f"\n Erro durante treinamento: {e}")
                import traceback
                traceback.print_exc()
        
        elif opcao == '2':
            classificar_todas_imagens_test()
        
        elif opcao == '3':
            print("\nEncerrando programa. Até mais!")
            break
        
        else:
            print(" Opção inválida! Escolha 1, 2 ou 3.")

if __name__ == "__main__":
    main()