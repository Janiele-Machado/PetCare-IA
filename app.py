"""
PROJETO: CLASSIFICADOR HIERÁRQUICO COM YOLOv8
Dois modelos separados: um para espécie, outro para raça
Com plot automático das imagens classificadas
 
Dependências:
    pip install ultralytics matplotlib numpy pillow
"""
 
# ============================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================
 
import os
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from ultralytics import YOLO
 
# ============================================
# CONFIGURAÇÕES
# ============================================
 
DATASET_PATH   = "dataset"
TRAIN_PATH     = os.path.join(DATASET_PATH, "train")
TEST_PATH      = os.path.join(DATASET_PATH, "test")
 
IMG_SIZE       = 224
EPOCHS         = 30           # YOLOv8 converge mais rápido que ResNet
BATCH_SIZE     = 16
LEARNING_RATE  = 1e-3
 
# O Ultralytics sempre salva em runs/classify/<name>/
RUNS_DIR           = "runs/classify"
MODEL_ESPECIE_NAME = "modelo_especie_yolo"
MODEL_RACA_NAME    = "modelo_raca_yolo"
 
# Caminhos completos dos pesos (onde o Ultralytics realmente salva)
MODEL_ESPECIE_PATH = os.path.join(RUNS_DIR, MODEL_ESPECIE_NAME, "treino", "weights", "best.pt")
MODEL_RACA_PATH    = os.path.join(RUNS_DIR, MODEL_RACA_NAME,    "treino", "weights", "best.pt")
 
INFO_ESPECIE_PATH  = "modelo_especie_info.json"
INFO_RACA_PATH     = "modelo_raca_info.json"
 
# ============================================
# FUNÇÃO 1: PREPARAR DATASET DE ESPÉCIE
# ============================================
 
def preparar_dataset_especie(train_path: str) -> tuple[str, str, list[str]]:
    """
    Reorganiza o dataset para classificação por espécie.
    Estrutura esperada em train_path:
        train_path/
            especie_a/
                raca_x/
                    img1.jpg ...
            especie_b/
                ...
    Retorna os caminhos de treino/val e lista de espécies.
    """
    print("\n" + "=" * 60)
    print("PREPARANDO DATASET PARA ESPÉCIE")
    print("=" * 60)
 
    base = os.path.join(tempfile.gettempdir(), "yolo_especie")
    train_dir = os.path.join(base, "train")
    val_dir   = os.path.join(base, "val")
 
    for d in [train_dir, val_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
 
    especies = sorted([
        e for e in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, e))
    ])
 
    for especie in especies:
        especie_path = os.path.join(train_path, especie)
        imagens = _coletar_imagens(especie_path, recursivo=True)
 
        split = max(1, int(len(imagens) * 0.2))
        val_imgs   = imagens[:split]
        train_imgs = imagens[split:]
 
        for subset, imgs in [("train", train_imgs), ("val", val_imgs)]:
            dest = os.path.join(base, subset, especie)
            os.makedirs(dest, exist_ok=True)
            for src in imgs:
                shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
 
    print(f"✓ Espécies encontradas: {especies}")
    print(f"✓ Treino/Val prontos em: {base}")
    return train_dir, val_dir, especies
 
 
# ============================================
# FUNÇÃO 2: PREPARAR DATASET DE RAÇA
# ============================================
 
def preparar_dataset_raca(train_path: str) -> tuple[str, str, list[str]]:
    """
    Reorganiza o dataset para classificação por raça (ignora espécie).
    Retorna os caminhos de treino/val e lista de raças.
    """
    print("\n" + "=" * 60)
    print("PREPARANDO DATASET PARA RAÇA")
    print("=" * 60)
 
    base = os.path.join(tempfile.gettempdir(), "yolo_raca")
    train_dir = os.path.join(base, "train")
    val_dir   = os.path.join(base, "val")
 
    for d in [train_dir, val_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
 
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
            imagens = _coletar_imagens(raca_path, recursivo=False)
 
            split = max(1, int(len(imagens) * 0.2))
            val_imgs   = imagens[:split]
            train_imgs = imagens[split:]
 
            for subset, imgs in [("train", train_imgs), ("val", val_imgs)]:
                dest = os.path.join(base, subset, raca)
                os.makedirs(dest, exist_ok=True)
                for src in imgs:
                    shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
 
    racas = sorted(set(racas))
    print(f"✓ Raças encontradas: {len(racas)}")
    print(f"✓ Treino/Val prontos em: {base}")
    return train_dir, val_dir, racas
 
 
# ============================================
# FUNÇÃO 3: TREINAR MODELO YOLOV8
# ============================================
 
def treinar_modelo(data_dir: str, nome_projeto: str, classes: list[str]) -> YOLO:
    """
    Treina um modelo YOLOv8 de classificação.
    Usa yolov8n-cls como ponto de partida (transfer learning).
    """
    print(f"\n{'=' * 60}")
    print(f"TREINANDO MODELO: {nome_projeto}")
    print(f"{'=' * 60}")
 
    model = YOLO("yolov8n-cls.pt")   # backbone leve; troque por yolov8s-cls.pt para mais acurácia
 
    # O Ultralytics salva em: <project>/<name>/weights/best.pt
    # Queremos: runs/classify/<nome_projeto>/treino/weights/best.pt
    model.train(
        data=data_dir,               # pasta com subpastas train/ e val/
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        project=os.path.join(RUNS_DIR, nome_projeto),
        name="treino",
        exist_ok=True,
        verbose=True,
        plots=True,                  # salva gráficos de loss/acurácia automaticamente
    )
 
    pesos = os.path.join(RUNS_DIR, nome_projeto, "treino", "weights", "best.pt")
    print(f"\n✓ Modelo treinado. Pesos em: {pesos}")
    return model
 
 
# ============================================
# FUNÇÃO 4: SALVAR INFORMAÇÕES DO MODELO
# ============================================
 
def salvar_info(classes: list[str], info_path: str, nome_projeto: str) -> None:
    """Salva metadados do modelo em JSON."""
    info = {
        "classes":        classes,
        "num_classes":    len(classes),
        "data":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tamanho_imagem": IMG_SIZE,
        "epocas":         EPOCHS,
        "modelo_path":    os.path.join(RUNS_DIR, nome_projeto, "treino", "weights", "best.pt"),
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    print(f"✓ Info salva: {info_path}")
 
 
# ============================================
# FUNÇÃO 5: CLASSIFICAR E PLOTAR IMAGENS
# ============================================
 
def classificar_e_plotar(test_path: str) -> None:
    """
    Classifica todas as imagens em test_path usando os dois modelos YOLO
    e plota cada imagem com os resultados de espécie e raça sobrepostos.
    """
    print("\n" + "=" * 60)
    print("CLASSIFICANDO TODAS AS IMAGENS DA PASTA TEST")
    print("=" * 60)
 
    # --- Verificar modelos e info ---
    # MODEL_ESPECIE_PATH e MODEL_RACA_PATH já apontam para o best.pt diretamente
    for p in [MODEL_ESPECIE_PATH, MODEL_RACA_PATH, INFO_ESPECIE_PATH, INFO_RACA_PATH]:
        if not os.path.exists(p):
            print(f"❌ Arquivo não encontrado: {p}")
            print("   Treine os modelos primeiro (opção 1).")
            return
 
    if not os.path.exists(test_path):
        print(f"❌ Pasta de teste não encontrada: {test_path}")
        return
 
    # --- Carregar modelos e metadados ---
    print("\nCarregando modelos...")
    model_especie  = YOLO(MODEL_ESPECIE_PATH)
    model_raca     = YOLO(MODEL_RACA_PATH)
    # Modelo de detecção geral (baixa ~6 MB automaticamente na 1ª execução)
    model_deteccao = YOLO("yolov8n.pt")
 
    with open(INFO_ESPECIE_PATH, "r", encoding="utf-8") as f:
        info_especie = json.load(f)
    with open(INFO_RACA_PATH, "r", encoding="utf-8") as f:
        info_raca = json.load(f)
 
    # --- Coletar imagens ---
    imagens = _coletar_imagens_test(test_path)
 
    if not imagens:
        print("\n❌ Nenhuma imagem encontrada na pasta test!")
        print("   Estrutura esperada: dataset/test/especie/raca/imagem.jpg")
        print("   Ou simplesmente:    dataset/test/imagem.jpg")
        return
 
    print(f"\n✓ {len(imagens)} imagem(ns) encontrada(s)")
 
    # --- Classificar e plotar cada imagem ---
    resultados = []
 
    for idx, img_info in enumerate(imagens, start=1):
        caminho = img_info["caminho"]
        print(f"\n[{idx}/{len(imagens)}] {img_info['nome']}")
 
        # Predições
        res_esp  = model_especie.predict(caminho, imgsz=IMG_SIZE, verbose=False)[0]
        res_raca = model_raca.predict(caminho,    imgsz=IMG_SIZE, verbose=False)[0]
 
        # Índice e confiança da classe vencedora
        esp_idx   = int(res_esp.probs.top1)
        raca_idx  = int(res_raca.probs.top1)
        conf_esp  = float(res_esp.probs.top1conf)
        conf_raca = float(res_raca.probs.top1conf)
 
        esp_nome  = info_especie["classes"][esp_idx]
        raca_nome = info_raca["classes"][raca_idx]
 
        print(f"   Espécie : {esp_nome} ({conf_esp*100:.1f}%)")
        print(f"   Raça    : {raca_nome} ({conf_raca*100:.1f}%)")
 
        # Detecção com bounding box (yolov8n geral)
        res_det = model_deteccao.predict(caminho, imgsz=640, verbose=False)[0]
 
        # Plot da imagem com resultados
        _plotar_resultado(
            caminho=caminho,
            esp_nome=esp_nome,
            conf_esp=conf_esp,
            raca_nome=raca_nome,
            conf_raca=conf_raca,
            res_esp=res_esp,
            res_raca=res_raca,
            res_det=res_det,
            info_especie=info_especie,
            info_raca=info_raca,
            idx=idx,
            total=len(imagens),
        )
 
        resultados.append({
            "imagem":           img_info["nome"],
            "caminho":          caminho,
            "especie_predita":  esp_nome,
            "raca_predita":     raca_nome,
            "confianca_especie":conf_esp,
            "confianca_raca":   conf_raca,
        })
 
    # --- Resumo e relatório ---
    print("\n" + "=" * 60)
    print(f"✓ {len(resultados)} imagem(ns) classificada(s) com sucesso!")
 
    relatorio = "resultados_classificacao.json"
    with open(relatorio, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
    print(f"✓ Relatório salvo em: {relatorio}")
    print("=" * 60)
 
 
# ============================================
# FUNÇÃO 6: PLOTAR RESULTADO DE UMA IMAGEM
# ============================================
 
def _plotar_resultado(
    caminho, esp_nome, conf_esp, raca_nome, conf_raca,
    res_esp, res_raca, res_det, info_especie, info_raca, idx, total
):
    """
    Gera um plot com 3 painéis:
      - Esquerda : imagem com bounding box do animal detectado
      - Centro   : top-5 espécies (barras horizontais)
      - Direita  : top-5 raças   (barras horizontais)
    """
    import matplotlib.patches as patches
    img = np.array(Image.open(caminho).convert("RGB"))
    h_img, w_img = img.shape[:2]
 
    # Top-5 espécies
    top5_esp_idx  = res_esp.probs.top5
    top5_esp_conf = res_esp.probs.top5conf.tolist()
    top5_esp_nome = [info_especie["classes"][i] for i in top5_esp_idx]
 
    # Top-5 raças
    top5_raca_idx  = res_raca.probs.top5
    top5_raca_conf = res_raca.probs.top5conf.tolist()
    top5_raca_nome = [info_raca["classes"][i] for i in top5_raca_idx]
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a2e")
 
    # ---- Painel 1: imagem com bounding box ----
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title(
        f"Imagem {idx}/{total} — {os.path.basename(caminho)}",
        color="white", fontsize=9, pad=8
    )
 
    # Desenhar bounding boxes retornados pelo modelo de detecção
    bbox_desenhado = False
    if res_det.boxes is not None and len(res_det.boxes) > 0:
        # Filtrar apenas animais (COCO: cat=15, dog=16) se existirem; senão usar todos
        ANIMAL_IDS = {15, 16}
        boxes_animais = [b for b in res_det.boxes if int(b.cls) in ANIMAL_IDS]
        boxes_usar = boxes_animais if boxes_animais else list(res_det.boxes)
 
        for box in boxes_usar:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf_box = float(box.conf[0])
            cls_id   = int(box.cls[0])
            label    = f"{res_det.names[cls_id]}: {conf_box:.3f}"
 
            rect = patches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor="#00ff88", facecolor="none",
                boxstyle="round,pad=2"
            )
            axes[0].add_patch(rect)
 
            # Fundo do label
            axes[0].text(
                x1, y1 - 8, label,
                color="black", fontsize=9, fontweight="bold",
                bbox=dict(facecolor="#00ff88", edgecolor="none", pad=3, alpha=0.9)
            )
            bbox_desenhado = True
 
    if not bbox_desenhado:
        # Sem detecção: só mostrar aviso discreto
        axes[0].text(
            0.5, 0.02, "⚠ sem detecção de bounding box",
            transform=axes[0].transAxes, ha="center", va="bottom",
            color="#ffaa00", fontsize=8
        )
 
    # Label de resultado abaixo da imagem
    axes[0].text(
        0.5, -0.04,
        f"🐾  {esp_nome}  •  {raca_nome}",
        transform=axes[0].transAxes,
        ha="center", va="top", fontsize=12, color="#00d4ff", fontweight="bold"
    )
 
    # ---- Painel 2: top-5 espécies ----
    _barras_horizontais(
        ax=axes[1],
        nomes=top5_esp_nome,
        confs=top5_esp_conf,
        titulo="Top-5 Espécies",
        cor_destaque="#00d4ff",
        cor_resto="#2a6090",
    )
 
    # ---- Painel 3: top-5 raças ----
    _barras_horizontais(
        ax=axes[2],
        nomes=top5_raca_nome,
        confs=top5_raca_conf,
        titulo="Top-5 Raças",
        cor_destaque="#ff6b6b",
        cor_resto="#7a3030",
    )
 
    plt.tight_layout(pad=2.5)
 
    # Salvar e exibir
    nome_saida = f"resultado_{idx:03d}_{os.path.splitext(os.path.basename(caminho))[0]}.png"
    plt.savefig(nome_saida, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"   📊 Plot salvo: {nome_saida}")
    plt.show()
    plt.close(fig)
 
 
def _barras_horizontais(ax, nomes, confs, titulo, cor_destaque, cor_resto):
    """
    Barras horizontais com nome à esquerda (fora da barra)
    e percentual à direita — sem sobreposição.
    """
    ax.set_facecolor("#16213e")
    n = len(nomes)
    # y: 4 no topo, 0 na base  →  maior confiança no topo
    y_pos = list(range(n - 1, -1, -1))
 
    cores = [cor_destaque if i == 0 else cor_resto for i in range(n)]
    bar_height = 0.55
 
    bars = ax.barh(y_pos, confs, color=cores, height=bar_height,
                   edgecolor="none", left=0)
 
    for bar, nome, conf, y in zip(bars, nomes, confs, y_pos):
        # Percentual: sempre à direita da barra, fora dela
        ax.text(
            conf + 0.02,
            y,
            f"{conf * 100:.1f}%",
            va="center", ha="left",
            color="white", fontsize=9, fontweight="bold"
        )
        # Nome: dentro da barra se couber (conf > 25%), senão à direita do %
        if conf > 0.25:
            ax.text(
                0.015, y,
                nome,
                va="center", ha="left",
                color="white", fontsize=8.5, fontweight="bold",
                clip_on=True
            )
        else:
            ax.text(
                conf + 0.09, y,
                nome,
                va="center", ha="left",
                color="#aaaaaa", fontsize=8,
                clip_on=False
            )
 
    ax.set_xlim(0, 1.30)          # espaço extra à direita para os rótulos
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                       color="#888888", fontsize=8)
    ax.tick_params(axis="x", colors="#555577", length=3)
    ax.tick_params(axis="y", left=False)
    for spine in ax.spines.values():
        spine.set_color("#2a2a4a")
    ax.set_title(titulo, color="white", fontsize=11, pad=10, fontweight="bold")
 
 
# ============================================
# FUNÇÃO 7: VISUALIZAR MÉTRICAS DO TREINAMENTO
# ============================================
 
def visualizar_treinamento() -> None:
    """
    Exibe os gráficos de loss/acurácia gerados automaticamente pelo Ultralytics
    (results.png) para ambos os modelos.
    """
    for nome, projeto in [("Espécie", MODEL_ESPECIE_NAME), ("Raça", MODEL_RACA_NAME)]:
        results_img = os.path.join(RUNS_DIR, projeto, "treino", "results.png")
        if os.path.exists(results_img):
            img = Image.open(results_img)
            plt.figure(figsize=(14, 6))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Métricas de Treinamento — {nome}", fontsize=14)
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠️  Gráfico não encontrado para {nome}: {results_img}")
 
 
# ============================================
# AUXILIARES INTERNOS
# ============================================
 
def _coletar_imagens(pasta: str, recursivo: bool) -> list[str]:
    """Retorna lista de caminhos de imagens em uma pasta."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = []
    if recursivo:
        for root, _, files in os.walk(pasta):
            for f in files:
                if Path(f).suffix.lower() in exts:
                    imgs.append(os.path.join(root, f))
    else:
        for f in os.listdir(pasta):
            if Path(f).suffix.lower() in exts:
                imgs.append(os.path.join(pasta, f))
    return sorted(imgs)
 
 
def _coletar_imagens_test(test_path: str) -> list[dict]:
    """
    Coleta imagens da pasta test.
    Suporta:
      - dataset/test/especie/raca/img.jpg
      - dataset/test/img.jpg
    """
    imagens = []
    tem_subpastas = any(
        os.path.isdir(os.path.join(test_path, item))
        for item in os.listdir(test_path)
    )
 
    if tem_subpastas:
        for especie in os.listdir(test_path):
            ep = os.path.join(test_path, especie)
            if not os.path.isdir(ep):
                continue
            for raca in os.listdir(ep):
                rp = os.path.join(ep, raca)
                if not os.path.isdir(rp):
                    continue
                for img in _coletar_imagens(rp, recursivo=False):
                    imagens.append({
                        "caminho":           img,
                        "especie_verdadeira": especie,
                        "raca_verdadeira":    raca,
                        "nome":              os.path.basename(img),
                    })
    else:
        for img in _coletar_imagens(test_path, recursivo=False):
            nome_sem_ext = Path(img).stem
            partes = nome_sem_ext.rsplit("_", 1)
            raca = partes[0] if len(partes) > 1 else nome_sem_ext
            imagens.append({
                "caminho":           img,
                "especie_verdadeira": "desconhecido",
                "raca_verdadeira":    raca,
                "nome":              os.path.basename(img),
            })
 
    return imagens
 
 
# ============================================
# FUNÇÃO PRINCIPAL
# ============================================
 
def main():
    print("\n" + "=" * 60)
    print("  CLASSIFICADOR HIERÁRQUICO — YOLOv8")
    print("=" * 60)
 
    while True:
        print("\n" + "=" * 60)
        print("MENU PRINCIPAL")
        print("=" * 60)
        print("1 - Treinar modelos")
        print("2 - Testar e plotar TODAS as imagens da pasta TEST")
        print("3 - Ver métricas do último treinamento")
        print("4 - Sair")
 
        opcao = input("\nEscolha (1-4): ").strip()
 
        if opcao == "1":
            try:
                # --- Espécie ---
                train_dir_esp, _, especies = preparar_dataset_especie(TRAIN_PATH)
                model_esp = treinar_modelo(
                    data_dir=os.path.dirname(train_dir_esp),   # pasta pai com train/ e val/
                    nome_projeto=MODEL_ESPECIE_NAME,
                    classes=especies,
                )
                salvar_info(especies, INFO_ESPECIE_PATH, MODEL_ESPECIE_NAME)
 
                # --- Raça ---
                train_dir_raca, _, racas = preparar_dataset_raca(TRAIN_PATH)
                model_raca = treinar_modelo(
                    data_dir=os.path.dirname(train_dir_raca),
                    nome_projeto=MODEL_RACA_NAME,
                    classes=racas,
                )
                salvar_info(racas, INFO_RACA_PATH, MODEL_RACA_NAME)
 
                print("\n" + "=" * 60)
                print("MODELOS TREINADOS COM SUCESSO!")
                print(f"   • Espécies : {len(especies)}")
                print(f"   • Raças    : {len(racas)}")
                print(f"   • Épocas   : {EPOCHS}")
                print("=" * 60)
 
            except Exception as e:
                import traceback
                print(f"\nErro durante treinamento: {e}")
                traceback.print_exc()
 
        elif opcao == "2":
            classificar_e_plotar(TEST_PATH)
 
        elif opcao == "3":
            visualizar_treinamento()
 
        elif opcao == "4":
            print("\nEncerrando. Até mais! ")
            break
 
        else:
            print("Opção inválida! Escolha entre 1 e 4.")
 
 
if __name__ == "__main__":
    main()