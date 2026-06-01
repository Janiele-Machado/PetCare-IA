"""
PROJETO: CLASSIFICADOR HIERÁRQUICO COM YOLOv8
Interface Web com Flask — Rosa Bebê Edition 
Suporta: cachorro, gato, cobra, coelho

Dependências:
    pip install ultralytics flask matplotlib numpy pillow
"""

# ============================================
# IMPORTAÇÕES
# ============================================

import os
import sys
import json
import shutil
import tempfile
import threading
import io
import base64
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend não-interativo (obrigatório para Flask)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO
from flask import Flask, render_template, request, jsonify

# ============================================
# CONFIGURAÇÕES
# ============================================

DATASET_PATH   = "dataset"
TRAIN_PATH     = os.path.join(DATASET_PATH, "train")
TEST_PATH      = os.path.join(DATASET_PATH, "test")

IMG_SIZE       = 224
EPOCHS         = 30
BATCH_SIZE     = 16
LEARNING_RATE  = 1e-3

RUNS_DIR           = "runs/classify"
MODEL_ESPECIE_NAME = "modelo_especie_yolo"
MODEL_RACA_NAME    = "modelo_raca_yolo"

# Onde o Ultralytics salva os pesos
MODEL_ESPECIE_PATH = os.path.join(RUNS_DIR, MODEL_ESPECIE_NAME, "treino", "weights", "best.pt")
MODEL_RACA_PATH    = os.path.join(RUNS_DIR, MODEL_RACA_NAME,    "treino", "weights", "best.pt")

INFO_ESPECIE_PATH  = "modelo_especie_info.json"
INFO_RACA_PATH     = "modelo_raca_info.json"

# Espécies suportadas (devem ser as pastas em dataset/train/)
# cachorro | gato | cobras | coelho
ESPECIES_SUPORTADAS = ["cachorro", "gato", "cobras", "coelho"]

# ============================================
# FLASK APP
# ============================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

# ============================================
# ESTADO GLOBAL DE TREINAMENTO
# ============================================

training_state = {
    "running": False,
    "log":     [],
    "done":    False,
    "error":   None,
    "especies": 0,
    "racas":    0,
}


def _log(msg: str) -> None:
    """Registra mensagem no log de treinamento E no console."""
    print(msg, flush=True)
    training_state["log"].append(str(msg))


# ============================================
# FUNÇÕES DE ML — PREPARAR DATASETS
# ============================================

def preparar_dataset_especie(train_path: str):
    _log("=" * 60)
    _log("PREPARANDO DATASET → ESPÉCIE")
    _log("=" * 60)

    base      = os.path.join(tempfile.gettempdir(), "yolo_especie")
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
        imagens = _coletar_imagens(os.path.join(train_path, especie), recursivo=True)
        split   = max(1, int(len(imagens) * 0.2))

        for subset, imgs in [("train", imagens[split:]), ("val", imagens[:split])]:
            dest = os.path.join(base, subset, especie)
            os.makedirs(dest, exist_ok=True)
            for src in imgs:
                shutil.copy2(src, os.path.join(dest, os.path.basename(src)))

    _log(f"✓ Espécies: {especies}")
    _log(f"✓ Dataset pronto em: {base}")
    return train_dir, val_dir, especies


def preparar_dataset_raca(train_path: str):
    _log("=" * 60)
    _log("PREPARANDO DATASET → RAÇA")
    _log("=" * 60)

    base      = os.path.join(tempfile.gettempdir(), "yolo_raca")
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
            split   = max(1, int(len(imagens) * 0.2))

            for subset, imgs in [("train", imagens[split:]), ("val", imagens[:split])]:
                dest = os.path.join(base, subset, raca)
                os.makedirs(dest, exist_ok=True)
                for src in imgs:
                    shutil.copy2(src, os.path.join(dest, os.path.basename(src)))

    racas = sorted(set(racas))
    _log(f"✓ Raças encontradas: {len(racas)}")
    _log(f"✓ Dataset pronto em: {base}")
    return train_dir, val_dir, racas


# ============================================
# FUNÇÕES DE ML — TREINAR E SALVAR
# ============================================

def treinar_modelo(data_dir: str, nome_projeto: str, classes: list) -> None:
    _log(f"\n{'=' * 60}")
    _log(f"TREINANDO: {nome_projeto}  ({len(classes)} classes)")
    _log(f"{'=' * 60}")

    model = YOLO("yolov8n-cls.pt")
    model.train(
        data      = data_dir,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH_SIZE,
        lr0       = LEARNING_RATE,
        project   = os.path.join(RUNS_DIR, nome_projeto),
        name      = "treino",
        exist_ok  = True,
        verbose   = True,
        plots     = True,
    )
    _log(f"✓ Pesos salvos em: {os.path.join(RUNS_DIR, nome_projeto, 'treino', 'weights', 'best.pt')}")


def salvar_info(classes: list, info_path: str, nome_projeto: str) -> None:
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
    _log(f"✓ Info salva: {info_path}")


# ============================================
# THREAD DE TREINAMENTO EM BACKGROUND
# ============================================

def _executar_treinamento():
    training_state.update(running=True, done=False, error=None, log=[],
                          especies=0, racas=0)
    try:
        # ── Espécie ──────────────────────────────────────────────────
        train_dir_esp, _, especies = preparar_dataset_especie(TRAIN_PATH)
        treinar_modelo(os.path.dirname(train_dir_esp), MODEL_ESPECIE_NAME, especies)
        salvar_info(especies, INFO_ESPECIE_PATH, MODEL_ESPECIE_NAME)

        # ── Raça ─────────────────────────────────────────────────────
        train_dir_raca, _, racas = preparar_dataset_raca(TRAIN_PATH)
        treinar_modelo(os.path.dirname(train_dir_raca), MODEL_RACA_NAME, racas)
        salvar_info(racas, INFO_RACA_PATH, MODEL_RACA_NAME)

        training_state["especies"] = len(especies)
        training_state["racas"]    = len(racas)
        _log("\n" + "=" * 60)
        _log("✅  MODELOS TREINADOS COM SUCESSO!")
        _log(f"   • Espécies : {len(especies)}")
        _log(f"   • Raças    : {len(racas)}")
        _log(f"   • Épocas   : {EPOCHS}")
        _log("=" * 60)

    except Exception as exc:
        import traceback
        training_state["error"] = str(exc)
        _log(f"❌  Erro: {exc}")
        _log(traceback.format_exc())
    finally:
        training_state["running"] = False
        training_state["done"]    = True


# ============================================
# CLASSIFICAÇÃO POR UPLOAD
# ============================================

def classificar_upload(file_storage):
    """
    Recebe um FileStorage do Flask, classifica com os dois modelos YOLO
    e retorna um dicionário com espécie, raça, top-5 e gráfico em base64.
    """
    # Verificar modelos
    ausentes = [p for p in [MODEL_ESPECIE_PATH, MODEL_RACA_PATH,
                             INFO_ESPECIE_PATH,  INFO_RACA_PATH]
                if not os.path.exists(p)]
    if ausentes:
        return {"success": False,
                "error": "Modelos não encontrados. Treine primeiro! "
                         f"(faltando: {', '.join(ausentes)})"}

    suffix = Path(file_storage.filename).suffix or ".jpg"
    tmp    = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    file_storage.save(tmp.name)
    tmp_path = tmp.name
    tmp.close()

    try:
        model_esp = YOLO(MODEL_ESPECIE_PATH)
        model_rac = YOLO(MODEL_RACA_PATH)
        model_det = YOLO("yolov8n.pt")         # detecção geral (bounding box)

        with open(INFO_ESPECIE_PATH, encoding="utf-8") as f:
            info_esp = json.load(f)
        with open(INFO_RACA_PATH, encoding="utf-8") as f:
            info_rac = json.load(f)

        res_esp = model_esp.predict(tmp_path, imgsz=IMG_SIZE, verbose=False)[0]
        res_rac = model_rac.predict(tmp_path, imgsz=IMG_SIZE, verbose=False)[0]
        res_det = model_det.predict(tmp_path, imgsz=640,      verbose=False)[0]

        esp_nome  = info_esp["classes"][int(res_esp.probs.top1)]
        rac_nome  = info_rac["classes"][int(res_rac.probs.top1)]
        conf_esp  = float(res_esp.probs.top1conf)
        conf_rac  = float(res_rac.probs.top1conf)

        # Gerar gráfico matplotlib → base64
        fig = _gerar_grafico(tmp_path, esp_nome, conf_esp, rac_nome, conf_rac,
                             res_esp, res_rac, res_det, info_esp, info_rac)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "success":           True,
            "especie":           esp_nome,
            "raca":              rac_nome,
            "confianca_especie": round(conf_esp * 100, 1),
            "confianca_raca":    round(conf_rac * 100, 1),
            "grafico":           img_b64,
            "top5_especies": [
                {"nome": info_esp["classes"][i], "conf": round(float(c) * 100, 1)}
                for i, c in zip(res_esp.probs.top5, res_esp.probs.top5conf.tolist())
            ],
            "top5_racas": [
                {"nome": info_rac["classes"][i], "conf": round(float(c) * 100, 1)}
                for i, c in zip(res_rac.probs.top5, res_rac.probs.top5conf.tolist())
            ],
        }

    finally:
        os.unlink(tmp_path)


# ============================================
# PLOTAGEM (matplotlib) — usada no upload
# ============================================

def _gerar_grafico(caminho, esp_nome, conf_esp, rac_nome, conf_rac,
                   res_esp, res_rac, res_det, info_esp, info_rac):
    img = np.array(Image.open(caminho).convert("RGB"))

    top5_esp_nome = [info_esp["classes"][i] for i in res_esp.probs.top5]
    top5_esp_conf = res_esp.probs.top5conf.tolist()
    top5_rac_nome = [info_rac["classes"][i] for i in res_rac.probs.top5]
    top5_rac_conf = res_rac.probs.top5conf.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a2e")

    # ── Painel 1: imagem + bounding box ──
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title(os.path.basename(caminho), color="white", fontsize=9, pad=8)

    bbox_ok = False
    if res_det.boxes is not None and len(res_det.boxes) > 0:
        ANIMAL_IDS = {15, 16}
        boxes = [b for b in res_det.boxes if int(b.cls) in ANIMAL_IDS] or list(res_det.boxes)
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = f"{res_det.names[int(box.cls[0])]}: {float(box.conf[0]):.2f}"
            rect  = patches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor="#00ff88", facecolor="none",
                boxstyle="round,pad=2"
            )
            axes[0].add_patch(rect)
            axes[0].text(x1, y1 - 8, label, color="black", fontsize=9,
                         fontweight="bold",
                         bbox=dict(facecolor="#00ff88", edgecolor="none", pad=3, alpha=0.9))
            bbox_ok = True

    if not bbox_ok:
        axes[0].text(0.5, 0.02, "⚠ sem bbox detectado",
                     transform=axes[0].transAxes, ha="center",
                     color="#ffaa00", fontsize=8)

    axes[0].text(0.5, -0.04, f"🐾  {esp_nome}  •  {rac_nome}",
                 transform=axes[0].transAxes, ha="center", va="top",
                 fontsize=12, color="#00d4ff", fontweight="bold")

    _barras(axes[1], top5_esp_nome, top5_esp_conf, "Top-5 Espécies", "#ff85a1", "#7a3050")
    _barras(axes[2], top5_rac_nome, top5_rac_conf, "Top-5 Raças",    "#ffb6c1", "#7a4058")

    plt.tight_layout(pad=2.5)
    return fig


def _barras(ax, nomes, confs, titulo, cor_destaque, cor_resto):
    ax.set_facecolor("#16213e")
    n     = len(nomes)
    y_pos = list(range(n - 1, -1, -1))
    cores = [cor_destaque if i == 0 else cor_resto for i in range(n)]
    bars  = ax.barh(y_pos, confs, color=cores, height=0.55, edgecolor="none")

    for _, nome, conf, y in zip(bars, nomes, confs, y_pos):
        ax.text(conf + 0.02, y, f"{conf * 100:.1f}%",
                va="center", ha="left", color="white", fontsize=9, fontweight="bold")
        if conf > 0.25:
            ax.text(0.015, y, nome, va="center", ha="left",
                    color="white", fontsize=8.5, fontweight="bold", clip_on=True)
        else:
            ax.text(conf + 0.09, y, nome, va="center", ha="left",
                    color="#aaaaaa", fontsize=8, clip_on=False)

    ax.set_xlim(0, 1.30)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color="#888888", fontsize=8)
    ax.tick_params(axis="x", colors="#555577", length=3)
    ax.tick_params(axis="y", left=False)
    for spine in ax.spines.values():
        spine.set_color("#2a2a4a")
    ax.set_title(titulo, color="white", fontsize=11, pad=10, fontweight="bold")


# ============================================
# AUXILIARES
# ============================================

def _coletar_imagens(pasta: str, recursivo: bool) -> list:
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


def _modelos_prontos() -> bool:
    return all(os.path.exists(p) for p in [MODEL_ESPECIE_PATH, MODEL_RACA_PATH])


# ============================================
# ROTAS FLASK
# ============================================

@app.route("/")
def index():
    return render_template("index.html", modelos_prontos=_modelos_prontos())


@app.route("/treinar", methods=["GET", "POST"])
def treinar():
    if request.method == "POST":
        if training_state["running"]:
            return jsonify({"status": "already_running"})
        t = threading.Thread(target=_executar_treinamento, daemon=True)
        t.start()
        return jsonify({"status": "started"})
    return render_template("treinar.html",
                           epochs=EPOCHS, img_size=IMG_SIZE,
                           batch=BATCH_SIZE, lr=LEARNING_RATE)


@app.route("/status_treino")
def status_treino():
    return jsonify(training_state)


@app.route("/testar", methods=["GET", "POST"])
def testar():
    if request.method == "POST":
        if "imagem" not in request.files:
            return jsonify({"success": False, "error": "Nenhum arquivo enviado."})
        f = request.files["imagem"]
        if not f or f.filename == "":
            return jsonify({"success": False, "error": "Arquivo vazio."})
        return jsonify(classificar_upload(f))
    return render_template("testar.html")


@app.route("/metricas")
def metricas():
    graficos = []
    for nome, projeto in [("Espécie", MODEL_ESPECIE_NAME), ("Raça", MODEL_RACA_NAME)]:
        p = os.path.join(RUNS_DIR, projeto, "treino", "results.png")
        if os.path.exists(p):
            with open(p, "rb") as f:
                graficos.append({"nome": nome,
                                  "imagem": base64.b64encode(f.read()).decode("utf-8")})
    return render_template("metricas.html", graficos=graficos)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("\n🌸  PetClassifier — Interface Web")
    print("   Acesse: http://localhost:5000\n")
    app.run(debug=True, port=5000, threaded=True)
