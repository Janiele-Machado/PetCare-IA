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
import contextlib
import json
import random
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

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH   = os.path.join(BASE_DIR, "dataset")
TRAIN_PATH     = os.path.join(DATASET_PATH, "train")
TEST_PATH      = os.path.join(DATASET_PATH, "test")

IMG_SIZE       = 224
EPOCHS         = 30          
BATCH_SIZE     = 16          
LEARNING_RATE  = 1e-3
PATIENCE       = 10          # para o treino se não melhorar por N épocas
MODEL_BACKBONE = "yolov8s-cls.pt"   # 's' (small) é 3x mais rápido que 'm' com acurácia muito próxima
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RUNS_DIR           = os.path.join(BASE_DIR, "runs", "classify")
MODEL_ESPECIE_NAME = "modelo_especie_yolo"

# Onde o Ultralytics salva os pesos
MODEL_ESPECIE_PATH = os.path.join(RUNS_DIR, MODEL_ESPECIE_NAME, "treino", "weights", "best.pt")

INFO_ESPECIE_PATH  = os.path.join(BASE_DIR, "modelo_especie_info.json")

# Espécies suportadas (devem ser as pastas em dataset/train/)
# cachorro | gato | cobras | coelho
ESPECIES_SUPORTADAS = ["cachorro", "gato", "cobras", "coelho"]

MODEL_RACA_MODEL_NAMES = {
    especie: f"modelo_raca_{especie}_yolo"
    for especie in ESPECIES_SUPORTADAS
}
MODEL_RACA_INFO_PATHS = {
    especie: os.path.join(BASE_DIR, f"modelo_raca_{especie}_info.json")
    for especie in ESPECIES_SUPORTADAS
}
MODEL_RACA_PATHS = {
    especie: os.path.join(RUNS_DIR, model_name, "treino", "weights", "best.pt")
    for especie, model_name in MODEL_RACA_MODEL_NAMES.items()
}
MODEL_RACA_FALLBACK_PATH = os.path.join(RUNS_DIR, "modelo_raca_yolo", "treino", "weights", "best.pt")
MODEL_RACA_FALLBACK_INFO = os.path.join(BASE_DIR, "modelo_raca_info.json")


def _find_weights_in_runs(model_name: str) -> str | None:
    """Procura por `treino/weights/best.pt` dentro de `RUNS_DIR` considerando
    possíveis subpastas aninhadas (ex: runs/classify/runs/...). Retorna o caminho
    completo se encontrado, ou `None` caso contrário.
    """
    # caminho direto esperado
    expected = os.path.join(RUNS_DIR, model_name, "treino", "weights", "best.pt")
    if os.path.exists(expected):
        return expected

    # possível pasta aninhada `runs/classify/runs/<model_name>/...`
    nested = os.path.join(RUNS_DIR, "runs", model_name, "treino", "weights", "best.pt")
    if os.path.exists(nested):
        return nested

    # busca recursiva dentro de RUNS_DIR por uma pasta com o nome do modelo
    for root, dirs, files in os.walk(RUNS_DIR):
        if os.path.basename(root) == model_name:
            candidate = os.path.join(root, "treino", "weights", "best.pt")
            if os.path.exists(candidate):
                return candidate

    # fallback: busca por qualquer best.pt dentro de RUNS_DIR (último recurso)
    for root, dirs, files in os.walk(RUNS_DIR):
        if "best.pt" in files:
            return os.path.join(root, "best.pt")

    return None


def _resolve_model_path(path: str | None, model_name: str | None = None) -> str | None:
    """Retorna `path` se existir, caso contrário tenta localizar o modelo
    usando o `model_name` dentro de `RUNS_DIR`.
    """
    if path and os.path.exists(path):
        return path
    if model_name:
        found = _find_weights_in_runs(model_name)
        if found:
            _log(f"🔎 Modelo '{model_name}' encontrado em: {found}")
            return found
    return None


def _find_results_png_in_runs(model_name: str) -> str | None:
    """Procura por `treino/results.png` dentro de `RUNS_DIR` considerando
    possíveis subpastas aninhadas (ex: runs/classify/runs/...). Retorna o caminho
    completo se encontrado, ou `None` caso contrário.
    """
    # caminho direto esperado
    expected = os.path.join(RUNS_DIR, model_name, "treino", "results.png")
    if os.path.exists(expected):
        return expected

    # possível pasta aninhada `runs/classify/runs/<model_name>/...`
    nested = os.path.join(RUNS_DIR, "runs", model_name, "treino", "results.png")
    if os.path.exists(nested):
        return nested

    # busca recursiva dentro de RUNS_DIR por uma pasta com o nome do modelo
    for root, dirs, files in os.walk(RUNS_DIR):
        if os.path.basename(root) == model_name:
            candidate = os.path.join(root, "treino", "results.png")
            if os.path.exists(candidate):
                return candidate

    return None

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

def preparar_dataset_especie(train_path: str, force: bool = False):
    _log("=" * 60)
    _log("PREPARANDO DATASET → ESPÉCIE")
    _log("=" * 60)

    base      = os.path.join(tempfile.gettempdir(), "petcare_especie")
    train_dir = os.path.join(base, "train")
    val_dir   = os.path.join(base, "val")
    yaml_path = os.path.join(base, "data.yaml")

    if force and os.path.exists(base):
        _log(f"⚠️  Forçando reconstrução do dataset temporário em: {base}")
        shutil.rmtree(base)

    if not force and _dataset_temporario_pronto(base):
        especies = sorted([
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ])
        _log(f"✓ Dataset já pronto em: {base} (pulando preparação)")
        _log(f"✓ Espécies: {especies}")
        return base, yaml_path, especies

    for d in [train_dir, val_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    especies = sorted([
        e for e in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, e)) and e in ESPECIES_SUPORTADAS
    ])

    imagens_por_especie = {}
    for especie in especies:
        imagens = _coletar_imagens(os.path.join(train_path, especie), recursivo=True)
        if not imagens:
            _log(f"⚠️  Espécie '{especie}' não possui imagens. Ignorando.")
            continue
        imagens_por_especie[especie] = imagens

    if not imagens_por_especie:
        _log("❌ Nenhuma espécie com imagens encontrada.")
        return base, yaml_path, []

    target_count = min(len(imgs) for imgs in imagens_por_especie.values())
    _log(f"⚠️  Balanceando espécies com {target_count} imagens por classe.")

    balanced_especies = []
    for especie, imagens in imagens_por_especie.items():
        random.shuffle(imagens)
        if len(imagens) > target_count:
            imagens = imagens[:target_count]

        split = max(1, int(len(imagens) * 0.2))
        val_images = imagens[:split]
        train_images = imagens[split:]

        _log(
            f"Processando espécie '{especie}': {len(imagens)} imagens "
            f"(train={len(train_images)}, val={len(val_images)})"
        )

        for subset, imgs in [("train", train_images), ("val", val_images)]:
            dest = os.path.join(base, subset, especie)
            os.makedirs(dest, exist_ok=True)
            for src in imgs:
                try:
                    shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
                except Exception as e:
                    _log(f"⚠️  Falha copiando {src}: {e}")

        balanced_especies.append(especie)

    _salvar_yaml_dataset(train_dir, val_dir, balanced_especies, yaml_path)
    _log(f"✓ Espécies: {balanced_especies}")
    _log(f"✓ Dataset pronto em: {base}")
    return base, yaml_path, balanced_especies


def preparar_dataset_raca(train_path: str, especie: str, force: bool = False):
    _log("=" * 60)
    _log(f"PREPARANDO DATASET → RAÇA ({especie})")
    _log("=" * 60)

    base      = os.path.join(tempfile.gettempdir(), f"petcare_raca_{especie}")
    train_dir = os.path.join(base, "train")
    val_dir   = os.path.join(base, "val")
    yaml_path = os.path.join(base, "data.yaml")

    if force and os.path.exists(base):
        _log(f"⚠️  Forçando reconstrução do dataset temporário em: {base}")
        shutil.rmtree(base)

    if not force and _dataset_temporario_pronto(base):
        racas = sorted([
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ])
        _log(f"✓ Dataset de raças já pronto em: {base} (pulando preparação)")
        _log(f"✓ Raças encontradas: {len(racas)}")
        return base, yaml_path, racas

    for d in [train_dir, val_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    especie_path = os.path.join(train_path, especie)
    if not os.path.isdir(especie_path):
        _log(f"⚠️  Espécie '{especie}' não encontrada em {train_path}.")
        return base, yaml_path, []

    racas = []
    for raca in sorted(os.listdir(especie_path)):
        raca_path = os.path.join(especie_path, raca)
        if not os.path.isdir(raca_path):
            continue

        imagens = _coletar_imagens(raca_path, recursivo=False)
        if not imagens:
            _log(f"⚠️  Raça '{raca}' não possui imagens. Ignorando.")
            continue

        random.shuffle(imagens)
        split = max(1, int(len(imagens) * 0.2))
        val_images = imagens[:split]
        train_images = imagens[split:]

        _log(
            f"Processando raça '{raca}' (espécie {especie}): {len(imagens)} imagens "
            f"(train={len(train_images)}, val={len(val_images)})"
        )

        for subset, imgs in [("train", train_images), ("val", val_images)]:
            dest = os.path.join(base, subset, raca)
            os.makedirs(dest, exist_ok=True)
            for src in imgs:
                try:
                    shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
                except Exception as e:
                    _log(f"⚠️  Falha copiando {src}: {e}")

        racas.append(raca)

    racas = sorted(racas)
    if racas:
        _salvar_yaml_dataset(train_dir, val_dir, racas, yaml_path)
    _log(f"✓ Raças encontradas: {len(racas)}")
    _log(f"✓ Dataset pronto em: {base}")
    return base, yaml_path, racas


# ============================================
# FUNÇÕES DE ML — TREINAR E SALVAR
# ============================================

def treinar_modelo(data_dir: str, nome_projeto: str, classes: list) -> None:
    _log(f"\n{'=' * 60}")
    _log(f"TREINANDO: {nome_projeto}  ({len(classes)} classes)")
    _log(f"{'=' * 60}")
    try:
        _log(f"✓ Usando diretório de dados: {data_dir}")
        model = YOLO(MODEL_BACKBONE)
        # Ajustes para execução em background: desativar plots e reduzir workers
        # Redirecionar stdout/stderr do Ultralytics para o `training_state['log']`
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr

        class TeeWriter:
            def __init__(self, orig):
                self.orig = orig
                self._buf = ""

            def write(self, s):
                try:
                    self.orig.write(s)
                except Exception:
                    pass
                if not s:
                    return
                self._buf += s
                while "\n" in self._buf:
                    line, self._buf = self._buf.split("\n", 1)
                    if line.strip():
                        training_state.setdefault("log", []).append(line)

            def flush(self):
                try:
                    self.orig.flush()
                except Exception:
                    pass

        with contextlib.redirect_stdout(TeeWriter(orig_stdout)), contextlib.redirect_stderr(TeeWriter(orig_stderr)):
            model.train(
                data      = data_dir,
                epochs    = EPOCHS,
                imgsz     = IMG_SIZE,
                batch     = BATCH_SIZE,
                lr0       = LEARNING_RATE,
                augment   = True,
                project   = os.path.join(RUNS_DIR, nome_projeto),
                name      = "treino",
                exist_ok  = True,
                verbose   = True,
                plots     = False,
                workers   = 4,         # paralelismo no carregamento de dados (era 0 = sem paralelismo!)
                patience  = PATIENCE,  # early stopping: para se não melhorar por N épocas
                cache     = False,     # não tenta cachear 20k imgs na RAM
                optimizer = "AdamW",   # AdamW converge melhor que SGD padrão
                cos_lr    = True,      # cosine LR schedule melhora a convergência
            )

        _log(f"✓ Pesos salvos em: {os.path.join(RUNS_DIR, nome_projeto, 'treino', 'weights', 'best.pt')}")

    except Exception as e:
        _log(f"❌ Erro no treinamento ({nome_projeto}): {e}")
        raise


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

def _executar_treinamento(force_dataset: bool = False):
    training_state.update(running=True, done=False, error=None, log=[],
                          especies=0, racas=0)
    if force_dataset:
        _log("⚠️  Reconstrução forçada do dataset ativada.")
    try:
        # ── Espécie ──────────────────────────────────────────────────
        train_base_esp, _, especies = preparar_dataset_especie(TRAIN_PATH, force=force_dataset)
        treinar_modelo(train_base_esp, MODEL_ESPECIE_NAME, especies)
        salvar_info(especies, INFO_ESPECIE_PATH, MODEL_ESPECIE_NAME)

        # ── Raça por espécie ──────────────────────────────────────────
        total_racas = 0
        for especie in ESPECIES_SUPORTADAS:
            dataset_base, _, racas = preparar_dataset_raca(TRAIN_PATH, especie, force=force_dataset)
            if not racas:
                _log(f"⚠️  Ignorando treinamento de raças para '{especie}' (sem imagens ou classes).")
                continue

            modelo_nome = MODEL_RACA_MODEL_NAMES[especie]
            treinar_modelo(dataset_base, modelo_nome, racas)
            salvar_info(racas, MODEL_RACA_INFO_PATHS[especie], modelo_nome)
            total_racas += len(racas)

        training_state["especies"] = len(especies)
        training_state["racas"]    = total_racas
        _log("\n" + "=" * 60)
        _log("✅  MODELOS TREINADOS COM SUCESSO!")
        _log(f"   • Espécies : {len(especies)}")
        _log(f"   • Raças    : {total_racas}")
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


def _executar_treinamento_especies(force_dataset: bool = False):
    """Treina somente o modelo de espécies, sem tocar nos modelos de raças.
    Útil para testes rápidos quando já existem modelos de raças treinados.
    """
    training_state.update(running=True, done=False, error=None, log=[], especies=0, racas=0)
    if force_dataset:
        _log("⚠️  Reconstrução forçada do dataset ativada (só espécies).")
    try:
        train_base_esp, _, especies = preparar_dataset_especie(TRAIN_PATH, force=force_dataset)
        treinar_modelo(train_base_esp, MODEL_ESPECIE_NAME, especies)
        salvar_info(especies, INFO_ESPECIE_PATH, MODEL_ESPECIE_NAME)
        training_state["especies"] = len(especies)
        _log("✅  Treino de espécies concluído (raças preservadas).")
    except Exception as exc:
        import traceback
        training_state["error"] = str(exc)
        _log(f"❌  Erro no treino de espécies: {exc}")
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
    # Verificar modelos (considera possíveis paths aninhados em RUNS_DIR)
    resolved_esp = _resolve_model_path(MODEL_ESPECIE_PATH, MODEL_ESPECIE_NAME)
    if not resolved_esp or not os.path.exists(INFO_ESPECIE_PATH):
        missing = []
        if not resolved_esp:
            missing.append(MODEL_ESPECIE_PATH)
        if not os.path.exists(INFO_ESPECIE_PATH):
            missing.append(INFO_ESPECIE_PATH)
        return {"success": False,
                "error": "Modelos não encontrados. Treine primeiro! "
                         f"(faltando: {', '.join(missing)})"}

    suffix = Path(file_storage.filename).suffix or ".jpg"
    tmp    = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    file_storage.save(tmp.name)
    tmp_path = tmp.name
    tmp.close()

    try:
        resolved_esp_path = _resolve_model_path(MODEL_ESPECIE_PATH, MODEL_ESPECIE_NAME)
        if not resolved_esp_path:
            _log(f"❌ Modelo de espécie não encontrado em '{MODEL_ESPECIE_PATH}' nem em subpastas de '{RUNS_DIR}'")
            return {"error": "Modelo de espécie indisponível"}
        model_esp = YOLO(resolved_esp_path)
        model_det = YOLO("yolov8n.pt")         # detecção geral (bounding box)

        with open(INFO_ESPECIE_PATH, encoding="utf-8") as f:
            info_esp = json.load(f)

        res_esp = model_esp.predict(tmp_path, imgsz=IMG_SIZE, verbose=False)[0]
        esp_nome = info_esp["classes"][int(res_esp.probs.top1)]
        conf_esp = float(res_esp.probs.top1conf)

        info_rac = None
        res_rac = None
        rac_nome = None
        conf_rac = 0.0
        top5_racas = []

        modelo_raca_path = MODEL_RACA_PATHS.get(esp_nome)
        modelo_raca_info = MODEL_RACA_INFO_PATHS.get(esp_nome)
        modelo_raca_path = _resolve_model_path(modelo_raca_path, MODEL_RACA_MODEL_NAMES.get(esp_nome))
        if not modelo_raca_path or not os.path.exists(modelo_raca_path) or not modelo_raca_info or not os.path.exists(modelo_raca_info):
            if os.path.exists(MODEL_RACA_FALLBACK_PATH) and os.path.exists(MODEL_RACA_FALLBACK_INFO):
                _log(f"⚠️  Modelo de raças por espécie não encontrado para '{esp_nome}'. Usando modelo genérico de raças.")
                modelo_raca_path = MODEL_RACA_FALLBACK_PATH
                modelo_raca_info = MODEL_RACA_FALLBACK_INFO
            else:
                _log(f"⚠️  Modelo de raças não encontrado para espécie '{esp_nome}'.")
                info_rac = {"classes": []}
                modelo_raca_path = None
                modelo_raca_info = None

        if modelo_raca_path and modelo_raca_info and os.path.exists(modelo_raca_path) and os.path.exists(modelo_raca_info):
            with open(modelo_raca_info, encoding="utf-8") as f:
                info_rac = json.load(f)

            model_rac = YOLO(modelo_raca_path)
            res_rac = model_rac.predict(tmp_path, imgsz=IMG_SIZE, verbose=False)[0]
            rac_nome = info_rac["classes"][int(res_rac.probs.top1)]
            conf_rac = float(res_rac.probs.top1conf)
            top5_racas = [
                {"nome": info_rac["classes"][i], "conf": round(float(c) * 100, 1)}
                for i, c in zip(res_rac.probs.top5, res_rac.probs.top5conf.tolist())
            ]
        else:
            if info_rac is None:
                info_rac = {"classes": []}

        res_det = model_det.predict(tmp_path, imgsz=640, verbose=False)[0]

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
            "top5_racas": top5_racas,
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
    top5_rac_nome = []
    top5_rac_conf = []
    if res_rac is not None and info_rac is not None:
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

    breed_label = rac_nome if rac_nome else "(sem modelo de raça)"
    axes[0].text(0.5, -0.04, f"🐾  {esp_nome}  •  {breed_label}",
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
    imgs = []
    if recursivo:
        for root, _, files in os.walk(pasta):
            for f in files:
                if Path(f).suffix.lower() in IMG_EXTS:
                    imgs.append(os.path.join(root, f))
    else:
        for f in os.listdir(pasta):
            if Path(f).suffix.lower() in IMG_EXTS:
                imgs.append(os.path.join(pasta, f))
    return sorted(imgs)


def _salvar_yaml_dataset(train_dir: str, val_dir: str, classes: list, yaml_path: str):
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"train: {train_dir}\n")
        f.write(f"val: {val_dir}\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("names:\n")
        for cls in classes:
            f.write(f"  - {json.dumps(cls, ensure_ascii=False)}\n")


def _dir_has_images(pasta: str) -> bool:
    if not os.path.isdir(pasta):
        return False
    for root, _, files in os.walk(pasta):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXTS:
                return True
    return False


def _dataset_temporario_pronto(base: str) -> bool:
    return _dir_has_images(os.path.join(base, "train")) and _dir_has_images(os.path.join(base, "val"))


def _modelos_prontos() -> bool:
    # resolve species model
    if not _resolve_model_path(MODEL_ESPECIE_PATH, MODEL_ESPECIE_NAME):
        return False

    # resolve any breed model or fallback
    for esp, p in MODEL_RACA_PATHS.items():
        if _resolve_model_path(p, MODEL_RACA_MODEL_NAMES.get(esp)):
            return True
    if _resolve_model_path(MODEL_RACA_FALLBACK_PATH, "modelo_raca_yolo"):
        return True
    return False


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
        force = request.json.get("force_dataset", False) if request.is_json else False
        t = threading.Thread(target=_executar_treinamento, args=(force,), daemon=True)
        t.start()
        return jsonify({"status": "started"})
    return render_template("treinar.html",
                           epochs=EPOCHS, img_size=IMG_SIZE,
                           batch=BATCH_SIZE, lr=LEARNING_RATE,
                           backbone=MODEL_BACKBONE)


@app.route("/treinar_especies", methods=["POST"])
def treinar_especies():
    """Inicia treino apenas do modelo de espécies. Recebe JSON opcional: {"force_dataset": true}.
    Use quando quiser preservar os modelos de raças existentes.
    """
    if training_state["running"]:
        return jsonify({"status": "already_running"})
    force = request.json.get("force_dataset", False) if request.is_json else False
    t = threading.Thread(target=_executar_treinamento_especies, args=(force,), daemon=True)
    t.start()
    return jsonify({"status": "started"})


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
    
    # Modelo de Espécies
    p_esp = _find_results_png_in_runs(MODEL_ESPECIE_NAME)
    if p_esp and os.path.exists(p_esp):
        with open(p_esp, "rb") as f:
            graficos.append({"nome": "Espécie",
                             "imagem": base64.b64encode(f.read()).decode("utf-8")})
    
    # Modelo de Raças (genérico - serve para todas as espécies)
    p_raca = _find_results_png_in_runs("modelo_raca_yolo")
    if p_raca and os.path.exists(p_raca):
        with open(p_raca, "rb") as f:
            graficos.append({"nome": "Raças (todas as espécies)",
                             "imagem": base64.b64encode(f.read()).decode("utf-8")})
    
    return render_template("metricas.html", graficos=graficos)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("\n🌸  PetClassifier — Interface Web")
    print("   Acesse: http://localhost:5000\n")
    app.run(debug=True, port=5000, threaded=True)
