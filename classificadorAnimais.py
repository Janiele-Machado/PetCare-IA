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

"""
PROJETO: CLASSIFICADOR HIERÁRQUICO 
Dois modelos separados: um para espécie, outro para raça
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
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

# Configurações
DATASET_PATH = "dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001

MODEL_ESPECIE_PATH = "modelo_especie.h5"
MODEL_RACA_PATH = "modelo_raca.h5"

# ============================================
# FUNÇÃO 1: CRIAR MODELO (GENÉRICO)
# ============================================

def criar_modelo(num_classes, nome_modelo):
    """Cria um modelo ResNet50 para classificação"""
    
    print(f"\nCriando modelo para {nome_modelo} com {num_classes} classes...")
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
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

def preparar_dados_especie(train_path, test_path):
    """Prepara dados para classificação de espécie (cachorro vs gato)"""
    
    print("\n" + "="*60)
    print("PREPARANDO DADOS PARA ESPÉCIE")
    print("="*60)
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    especies = list(train_generator.class_indices.keys())
    
    print(f"✓ Espécies: {especies}")
    print(f"✓ Treino: {train_generator.samples} imagens")
    print(f"✓ Teste: {test_generator.samples} imagens")
    
    return train_generator, test_generator, especies

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
    from sklearn.model_selection import train_test_split
    
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
    
    # Criar geradores
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = datagen.flow_from_directory(
        train_temp,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = datagen.flow_from_directory(
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
    """Treina um modelo"""
    
    print(f"\n{'='*60}")
    print(f"TREINANDO MODELO: {nome}")
    print(f"{'='*60}")
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# ============================================
# FUNÇÃO 5: SALVAR MODELO
# ============================================

def salvar_modelo(model, path, classes):
    """Salva modelo e metadados"""
    
    model.save(path)
    
    info = {
        'classes': classes,
        'num_classes': len(classes),
        'data': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'tamanho_imagem': IMG_SIZE
    }
    
    info_path = path.replace('.h5', '_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"✓ Modelo salvo: {path}")
    print(f"✓ Info salva: {info_path}")

# ============================================
# FUNÇÃO 6: CLASSIFICAR IMAGEM
# ============================================

def classificar_imagem(imagem_path):
    """Classifica uma imagem usando ambos os modelos"""
    
    # Carregar modelos
    if not os.path.exists(MODEL_ESPECIE_PATH) or not os.path.exists(MODEL_RACA_PATH):
        print("Modelos não encontrados. Treine primeiro!")
        return None
    
    model_especie = load_model(MODEL_ESPECIE_PATH)
    model_raca = load_model(MODEL_RACA_PATH)
    
    # Carregar metadados
    with open(MODEL_ESPECIE_PATH.replace('.h5', '_info.json'), 'r') as f:
        info_especie = json.load(f)
    with open(MODEL_RACA_PATH.replace('.h5', '_info.json'), 'r') as f:
        info_raca = json.load(f)
    
    # Processar imagem
    img = load_img(imagem_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Predizer espécie
    pred_especie = model_especie.predict(img_array, verbose=0)
    especie_idx = np.argmax(pred_especie[0])
    especie = info_especie['classes'][especie_idx]
    conf_especie = pred_especie[0][especie_idx]
    
    # Predizer raça
    pred_raca = model_raca.predict(img_array, verbose=0)
    raca_idx = np.argmax(pred_raca[0])
    raca = info_raca['classes'][raca_idx]
    conf_raca = pred_raca[0][raca_idx]
    
    return {
        'especie': especie,
        'confianca_especie': conf_especie,
        'raca': raca,
        'confianca_raca': conf_raca
    }

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
    print("="*60)
    
    while True:
        print("\nMENU:")
        print("1 - Treinar modelos")
        print("2 - Classificar imagem")
        print("3 - Sair")
        
        opcao = input("\nEscolha: ")
        
        if opcao == '1':
            try:
                # Treinar modelo de ESPÉCIE
                print("\n" + "-"*30)
                print("TREINANDO CLASSIFICADOR DE ESPÉCIE")
                print("-"*30)
                
                train_gen_esp, val_gen_esp, especies = preparar_dados_especie(TRAIN_PATH, TEST_PATH)
                model_esp = criar_modelo(len(especies), "ESPÉCIE")
                history_esp = treinar_modelo(model_esp, train_gen_esp, val_gen_esp, "ESPÉCIE")
                salvar_modelo(model_esp, MODEL_ESPECIE_PATH, especies)
                
                # Treinar modelo de RAÇA
                print("\n" + "-"*30)
                print("TREINANDO CLASSIFICADOR DE RAÇA")
                print("-"*30)
                
                train_gen_raca, val_gen_raca, racas = preparar_dados_raca(TRAIN_PATH)
                model_raca = criar_modelo(len(racas), "RAÇA")
                history_raca = treinar_modelo(model_raca, train_gen_raca, val_gen_raca, "RAÇA")
                salvar_modelo(model_raca, MODEL_RACA_PATH, racas)
                
                # Visualizar resultados
                visualizar_treinamento(history_esp, history_raca)
                
                print("\n✅ MODELOS TREINADOS COM SUCESSO!")
                
            except Exception as e:
                print(f"\n Erro: {e}")
                import traceback
                traceback.print_exc()
        
        elif opcao == '2':
            imagem_path = input("Caminho da imagem: ")
            
            if not os.path.exists(imagem_path):
                print(" Imagem não encontrada!")
                continue
            
            resultado = classificar_imagem(imagem_path)
            
            if resultado:
                print("\n" + "="*40)
                print("RESULTADO DA CLASSIFICAÇÃO")
                print("="*40)
                print(f" ESPÉCIE: {resultado['especie'].upper()}")
                print(f"   Confiança: {resultado['confianca_especie']*100:.2f}%")
                print(f"\n RAÇA: {resultado['raca']}")
                print(f"   Confiança: {resultado['confianca_raca']*100:.2f}%")
        
        elif opcao == '3':
            print("Encerrando...")
            break

if __name__ == "__main__":
    main()