"""
CLASSIFICADOR HIERÁRQUICO COM YOLOv8 — Interface Gráfica
Paleta: rosa claro/pastel + branco + preto
"""

import os
import json
import shutil
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageTk
from ultralytics import YOLO

# ============================================
# PALETA DE CORES
# ============================================

COLORS = {
    "bg":           "#1a1118",      # fundo principal escuro (quase preto com toque rosado)
    "bg2":          "#231520",      # fundo painel secundário
    "bg3":          "#2e1b2a",      # fundo card
    "pink1":        "#f4a7c3",      # rosa pastel principal
    "pink2":        "#e8799e",      # rosa médio
    "pink3":        "#f9c9d9",      # rosa bem claro
    "pink4":        "#d4547a",      # rosa mais saturado (acento)
    "pink5":        "#fde8f0",      # quase branco rosado
    "white":        "#ffffff",
    "off_white":    "#f7eef3",
    "gray":         "#8a7085",
    "gray2":        "#5a4558",
    "border":       "#3d2438",
    "success":      "#c8f5c8",
    "error":        "#ffb3b3",
    "warning":      "#ffd9a8",
    "text":         "#f7eef3",
    "text_dim":     "#9a7d90",
}

FONT_TITLE  = ("Georgia", 22, "bold")
FONT_SUB    = ("Georgia", 13, "italic")
FONT_LABEL  = ("Trebuchet MS", 11)
FONT_LABEL_B= ("Trebuchet MS", 11, "bold")
FONT_MONO   = ("Courier New", 10)
FONT_BTN    = ("Trebuchet MS", 12, "bold")
FONT_SMALL  = ("Trebuchet MS", 9)

# ============================================
# CONFIGURAÇÕES
# ============================================

DATASET_PATH        = "dataset"
TRAIN_PATH          = os.path.join(DATASET_PATH, "train")
TEST_PATH           = os.path.join(DATASET_PATH, "test")
IMG_SIZE            = 224
EPOCHS              = 30
BATCH_SIZE          = 16
LEARNING_RATE       = 1e-3
RUNS_DIR            = "runs/classify"
MODEL_ESPECIE_NAME  = "modelo_especie_yolo"
MODEL_RACA_NAME     = "modelo_raca_yolo"
MODEL_ESPECIE_PATH  = os.path.join(RUNS_DIR, MODEL_ESPECIE_NAME, "treino", "weights", "best.pt")
MODEL_RACA_PATH     = os.path.join(RUNS_DIR, MODEL_RACA_NAME,    "treino", "weights", "best.pt")
INFO_ESPECIE_PATH   = "modelo_especie_info.json"
INFO_RACA_PATH      = "modelo_raca_info.json"

# ============================================
# UTILITÁRIOS DE IMAGEM
# ============================================

def _coletar_imagens(pasta, recursivo):
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


def _coletar_imagens_test(test_path):
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
                        "caminho": img,
                        "especie_verdadeira": especie,
                        "raca_verdadeira": raca,
                        "nome": os.path.basename(img),
                    })
    else:
        for img in _coletar_imagens(test_path, recursivo=False):
            nome_sem_ext = Path(img).stem
            partes = nome_sem_ext.rsplit("_", 1)
            raca = partes[0] if len(partes) > 1 else nome_sem_ext
            imagens.append({
                "caminho": img,
                "especie_verdadeira": "desconhecido",
                "raca_verdadeira": raca,
                "nome": os.path.basename(img),
            })
    return imagens

# ============================================
# WIDGET: BARRA DE PROGRESSO PERSONALIZADA
# ============================================

class PinkProgressBar(tk.Canvas):
    def __init__(self, parent, width=400, height=14, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=COLORS["bg3"], highlightthickness=0, **kwargs)
        self._w = width
        self._h = height
        self._pct = 0
        self._draw()

    def _draw(self):
        self.delete("all")
        r = self._h // 2
        # trilha
        self.create_rounded_rect(0, 0, self._w, self._h, r, fill=COLORS["bg2"], outline=COLORS["border"])
        # preenchimento
        fill_w = int(self._w * self._pct)
        if fill_w > 4:
            self.create_rounded_rect(0, 0, fill_w, self._h, r,
                                     fill=COLORS["pink2"], outline="")
            # brilho
            self.create_rounded_rect(2, 2, fill_w - 2, self._h // 2, r - 1,
                                     fill=COLORS["pink3"], outline="", stipple="gray50")

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kw):
        points = [
            x1+r, y1, x2-r, y1, x2, y1, x2, y1+r,
            x2, y2-r, x2, y2, x2-r, y2, x1+r, y2,
            x1, y2, x1, y2-r, x1, y1+r, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kw)

    def set(self, value):
        self._pct = max(0.0, min(1.0, value))
        self._draw()

# ============================================
# WIDGET: CONSOLE DE LOG
# ============================================

class LogConsole(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS["bg2"], **kwargs)
        self._build()

    def _build(self):
        header = tk.Frame(self, bg=COLORS["bg2"])
        header.pack(fill="x", pady=(0, 4))
        tk.Label(header, text="◈  Console de Saída", font=FONT_LABEL_B,
                 bg=COLORS["bg2"], fg=COLORS["pink1"]).pack(side="left")
        tk.Button(header, text="limpar", font=FONT_SMALL,
                  bg=COLORS["bg3"], fg=COLORS["gray"], relief="flat",
                  activebackground=COLORS["border"], activeforeground=COLORS["pink3"],
                  cursor="hand2", command=self.clear).pack(side="right", padx=4)

        self.text = scrolledtext.ScrolledText(
            self, font=FONT_MONO, bg="#110d10", fg=COLORS["pink3"],
            insertbackground=COLORS["pink1"], relief="flat",
            wrap="word", state="disabled", height=14,
        )
        self.text.pack(fill="both", expand=True)

        # tags de cor
        self.text.tag_config("ok",      foreground=COLORS["success"])
        self.text.tag_config("err",     foreground=COLORS["error"])
        self.text.tag_config("warn",    foreground=COLORS["warning"])
        self.text.tag_config("pink",    foreground=COLORS["pink1"])
        self.text.tag_config("dim",     foreground=COLORS["text_dim"])
        self.text.tag_config("white",   foreground=COLORS["white"])

    def log(self, msg, tag="white"):
        self.text.config(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        self.text.insert("end", f"[{ts}] {msg}\n", tag)
        self.text.see("end")
        self.text.config(state="disabled")

    def clear(self):
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.config(state="disabled")

# ============================================
# TELA: MENU PRINCIPAL
# ============================================

class MenuPrincipal(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=COLORS["bg"])
        self.controller = controller
        self._build()

    def _build(self):
        # Decoração de fundo — círculos difusos
        canvas = tk.Canvas(self, bg=COLORS["bg"], highlightthickness=0)
        canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        for cx, cy, cr, alpha_color in [
            (100, 80,  260, "#2d1525"),
            (680, 500, 200, "#2a1322"),
            (800, 50,  160, "#231220"),
        ]:
            canvas.create_oval(cx-cr, cy-cr, cx+cr, cy+cr, fill=alpha_color, outline="")

        # Conteúdo centralizado
        center = tk.Frame(self, bg=COLORS["bg"])
        center.place(relx=0.5, rely=0.5, anchor="center")

        # Ícone / logo
        tk.Label(center, text="🐾", font=("Segoe UI Emoji", 48),
                 bg=COLORS["bg"]).pack(pady=(0, 4))

        tk.Label(center, text="Classificador Hierárquico",
                 font=FONT_TITLE, bg=COLORS["bg"], fg=COLORS["pink1"]).pack()
        tk.Label(center, text="YOLOv8  ·  Espécie & Raça",
                 font=FONT_SUB, bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(pady=(2, 40))

        # Botões principais
        btns = [
            ("🏋️  Treinar Modelos",              "TreinoPage",   COLORS["pink2"],  COLORS["white"]),
            ("🔍  Testar & Classificar Imagens", "TestePage",    COLORS["pink4"],  COLORS["white"]),
            ("📊  Ver Métricas do Treinamento",  "MetricasPage", COLORS["bg3"],    COLORS["pink1"]),
        ]
        for label, page, bg, fg in btns:
            self._make_btn(center, label, page, bg, fg)

        tk.Label(center, text="v1.0  ·  Interface Gráfica",
                 font=FONT_SMALL, bg=COLORS["bg"], fg=COLORS["gray2"]).pack(pady=(30, 0))

    def _make_btn(self, parent, text, page, bg, fg):
        frame = tk.Frame(parent, bg=COLORS["bg"])
        frame.pack(fill="x", pady=7)

        btn = tk.Button(
            frame, text=text, font=FONT_BTN,
            bg=bg, fg=fg, relief="flat", cursor="hand2",
            activebackground=COLORS["pink3"], activeforeground=COLORS["bg"],
            padx=40, pady=14,
            command=lambda p=page: self.controller.show(p)
        )
        btn.pack(fill="x")

        # hover suave
        btn.bind("<Enter>", lambda e, b=btn: b.config(bg=COLORS["pink3"], fg=COLORS["bg"]))
        btn.bind("<Leave>", lambda e, b=btn, c=bg, f=fg: b.config(bg=c, fg=f))

# ============================================
# TELA: TREINO
# ============================================

class TreinoPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=COLORS["bg"])
        self.controller = controller
        self._running = False
        self._build()

    def _build(self):
        self._header("🏋️  Treinar Modelos")

        # Config card
        card = self._card(self)
        card.pack(fill="x", padx=30, pady=(0, 12))

        self._section(card, "Configurações de Treino")
        grid = tk.Frame(card, bg=COLORS["bg3"])
        grid.pack(fill="x", padx=16, pady=(0, 12))

        labels   = ["Épocas", "Batch Size", "Img Size", "Learning Rate"]
        defaults = [str(EPOCHS), str(BATCH_SIZE), str(IMG_SIZE), str(LEARNING_RATE)]
        self._entries = {}

        for col, (lbl, dflt) in enumerate(zip(labels, defaults)):
            f = tk.Frame(grid, bg=COLORS["bg3"])
            f.grid(row=0, column=col, padx=8, sticky="w")
            tk.Label(f, text=lbl, font=FONT_SMALL, bg=COLORS["bg3"],
                     fg=COLORS["text_dim"]).pack(anchor="w")
            e = tk.Entry(f, font=FONT_LABEL, bg=COLORS["bg2"], fg=COLORS["pink3"],
                         insertbackground=COLORS["pink1"], relief="flat",
                         width=12, justify="center")
            e.insert(0, dflt)
            e.pack(pady=2)
            self._entries[lbl] = e

        # Pasta dataset
        self._section(card, "Pasta de Treino")
        path_row = tk.Frame(card, bg=COLORS["bg3"])
        path_row.pack(fill="x", padx=16, pady=(0, 12))
        self._path_var = tk.StringVar(value=TRAIN_PATH)
        tk.Entry(path_row, textvariable=self._path_var, font=FONT_LABEL,
                 bg=COLORS["bg2"], fg=COLORS["pink3"], relief="flat",
                 insertbackground=COLORS["pink1"], width=40).pack(side="left", ipady=6, padx=(0,8))
        tk.Button(path_row, text="Procurar…", font=FONT_SMALL,
                  bg=COLORS["border"], fg=COLORS["pink1"], relief="flat",
                  cursor="hand2", command=self._browse).pack(side="left")

        # Barra de progresso
        prog_card = self._card(self)
        prog_card.pack(fill="x", padx=30, pady=(0, 12))
        self._section(prog_card, "Progresso")
        prog_inner = tk.Frame(prog_card, bg=COLORS["bg3"])
        prog_inner.pack(fill="x", padx=16, pady=(0, 12))

        self._status_label = tk.Label(prog_inner, text="Aguardando início…",
                                      font=FONT_LABEL, bg=COLORS["bg3"], fg=COLORS["text_dim"])
        self._status_label.pack(anchor="w", pady=(0, 6))
        self._pbar = PinkProgressBar(prog_inner, width=640, height=16)
        self._pbar.pack(anchor="w")

        # Console
        log_card = self._card(self)
        log_card.pack(fill="both", expand=True, padx=30, pady=(0, 12))
        self._log = LogConsole(log_card)
        self._log.pack(fill="both", expand=True, padx=16, pady=12)

        # Botões de ação
        btn_row = tk.Frame(self, bg=COLORS["bg"])
        btn_row.pack(fill="x", padx=30, pady=(0, 20))
        self._btn_train = self._action_btn(btn_row, "🚀  Iniciar Treinamento",
                                           self._start_training, COLORS["pink2"])
        self._btn_train.pack(side="left")
        self._back_btn(btn_row)

    def _browse(self):
        path = filedialog.askdirectory(title="Selecione a pasta de treino")
        if path:
            self._path_var.set(path)

    def _start_training(self):
        if self._running:
            return
        self._running = True
        self._btn_train.config(state="disabled", text="⏳  Treinando…")
        self._log.clear()
        self._pbar.set(0)
        self._status_label.config(text="Preparando dataset…", fg=COLORS["pink1"])
        t = threading.Thread(target=self._run_training, daemon=True)
        t.start()

    def _run_training(self):
        try:
            global EPOCHS, BATCH_SIZE, IMG_SIZE, LEARNING_RATE
            EPOCHS        = int(self._entries["Épocas"].get())
            BATCH_SIZE    = int(self._entries["Batch Size"].get())
            IMG_SIZE      = int(self._entries["Img Size"].get())
            LEARNING_RATE = float(self._entries["Learning Rate"].get())
            train_path = self._path_var.get()

            self._log_ui("Preparando dataset de espécie…", "pink")
            self._set_status("Preparando dataset de espécie…", 0.05)
            train_dir_esp, _, especies = preparar_dataset_especie(train_path, self._log_ui)

            self._log_ui(f"✓ Espécies: {especies}", "ok")
            self._set_status("Treinando modelo de espécie…", 0.20)
            model_esp = treinar_modelo(
                data_dir=os.path.dirname(train_dir_esp),
                nome_projeto=MODEL_ESPECIE_NAME,
                classes=especies,
                log_fn=self._log_ui,
            )
            salvar_info(especies, INFO_ESPECIE_PATH, MODEL_ESPECIE_NAME)
            self._set_status("Modelo de espécie concluído!", 0.55)

            self._log_ui("Preparando dataset de raça…", "pink")
            self._set_status("Preparando dataset de raça…", 0.60)
            train_dir_raca, _, racas = preparar_dataset_raca(train_path, self._log_ui)

            self._log_ui(f"✓ Raças: {len(racas)} encontradas", "ok")
            self._set_status("Treinando modelo de raça…", 0.65)
            model_raca = treinar_modelo(
                data_dir=os.path.dirname(train_dir_raca),
                nome_projeto=MODEL_RACA_NAME,
                classes=racas,
                log_fn=self._log_ui,
            )
            salvar_info(racas, INFO_RACA_PATH, MODEL_RACA_NAME)

            self._set_status("✅  Treinamento concluído com sucesso!", 1.0)
            self._log_ui(f"Espécies: {len(especies)}  |  Raças: {len(racas)}  |  Épocas: {EPOCHS}", "ok")

        except Exception as e:
            import traceback
            self._log_ui(f"Erro: {e}", "err")
            self._log_ui(traceback.format_exc(), "err")
            self._set_status("❌  Erro durante o treinamento.", 0)
        finally:
            self._running = False
            self.after(0, lambda: self._btn_train.config(
                state="normal", text="🚀  Iniciar Treinamento"))

    def _log_ui(self, msg, tag="white"):
        self.after(0, lambda: self._log.log(msg, tag))

    def _set_status(self, msg, pct):
        self.after(0, lambda: self._status_label.config(text=msg))
        self.after(0, lambda: self._pbar.set(pct))

    # ---- helpers de layout ----
    def _header(self, title):
        h = tk.Frame(self, bg=COLORS["bg2"], pady=0)
        h.pack(fill="x")
        inner = tk.Frame(h, bg=COLORS["bg2"])
        inner.pack(fill="x", padx=30, pady=16)
        tk.Label(inner, text=title, font=("Georgia", 17, "bold"),
                 bg=COLORS["bg2"], fg=COLORS["pink1"]).pack(side="left")

    def _card(self, parent):
        f = tk.Frame(parent, bg=COLORS["bg3"],
                     highlightthickness=1, highlightbackground=COLORS["border"])
        return f

    def _section(self, parent, text):
        tk.Label(parent, text=text, font=FONT_LABEL_B,
                 bg=COLORS["bg3"], fg=COLORS["pink2"]).pack(anchor="w", padx=16, pady=(12, 4))

    def _action_btn(self, parent, text, cmd, bg):
        b = tk.Button(parent, text=text, font=FONT_BTN,
                      bg=bg, fg=COLORS["white"], relief="flat",
                      cursor="hand2", padx=28, pady=10, command=cmd,
                      activebackground=COLORS["pink3"], activeforeground=COLORS["bg"])
        b.bind("<Enter>", lambda e: b.config(bg=COLORS["pink3"], fg=COLORS["bg"]))
        b.bind("<Leave>", lambda e: b.config(bg=bg, fg=COLORS["white"]))
        return b

    def _back_btn(self, parent):
        b = tk.Button(parent, text="← Voltar", font=FONT_LABEL,
                      bg=COLORS["bg3"], fg=COLORS["text_dim"], relief="flat",
                      cursor="hand2", padx=18, pady=10,
                      command=lambda: self.controller.show("MenuPrincipal"))
        b.pack(side="right")

# ============================================
# TELA: TESTE / CLASSIFICAÇÃO
# ============================================

class TestePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=COLORS["bg"])
        self.controller = controller
        self._running = False
        self._build()

    def _build(self):
        self._header("🔍  Testar & Classificar Imagens")

        # Painel superior: pasta + botão
        card_top = self._card(self)
        card_top.pack(fill="x", padx=30, pady=(0, 12))
        self._section(card_top, "Pasta de Teste")

        row = tk.Frame(card_top, bg=COLORS["bg3"])
        row.pack(fill="x", padx=16, pady=(0, 14))
        self._path_var = tk.StringVar(value=TEST_PATH)
        tk.Entry(row, textvariable=self._path_var, font=FONT_LABEL,
                 bg=COLORS["bg2"], fg=COLORS["pink3"], relief="flat",
                 insertbackground=COLORS["pink1"], width=44).pack(side="left", ipady=6, padx=(0,8))
        tk.Button(row, text="Procurar…", font=FONT_SMALL,
                  bg=COLORS["border"], fg=COLORS["pink1"], relief="flat",
                  cursor="hand2", command=self._browse).pack(side="left")

        # Status + progresso
        card_prog = self._card(self)
        card_prog.pack(fill="x", padx=30, pady=(0, 12))
        self._section(card_prog, "Progresso")
        inner = tk.Frame(card_prog, bg=COLORS["bg3"])
        inner.pack(fill="x", padx=16, pady=(0, 14))
        self._status = tk.Label(inner, text="Aguardando…", font=FONT_LABEL,
                                bg=COLORS["bg3"], fg=COLORS["text_dim"])
        self._status.pack(anchor="w", pady=(0, 6))
        self._pbar = PinkProgressBar(inner, width=640, height=16)
        self._pbar.pack(anchor="w")

        # Preview da última imagem classificada
        self._preview_card = self._card(self)
        self._preview_card.pack(fill="x", padx=30, pady=(0, 12))
        self._section(self._preview_card, "Última Classificação")
        self._preview_frame = tk.Frame(self._preview_card, bg=COLORS["bg3"])
        self._preview_frame.pack(fill="x", padx=16, pady=(0, 14))
        self._preview_label = tk.Label(self._preview_frame,
                                       text="Nenhuma imagem classificada ainda.",
                                       font=FONT_LABEL, bg=COLORS["bg3"], fg=COLORS["text_dim"])
        self._preview_label.pack()

        # Console
        log_card = self._card(self)
        log_card.pack(fill="both", expand=True, padx=30, pady=(0, 12))
        self._log = LogConsole(log_card)
        self._log.pack(fill="both", expand=True, padx=16, pady=12)

        # Botões
        btn_row = tk.Frame(self, bg=COLORS["bg"])
        btn_row.pack(fill="x", padx=30, pady=(0, 20))
        self._btn_run = self._action_btn(btn_row, "🔍  Classificar Todas as Imagens",
                                         self._start, COLORS["pink4"])
        self._btn_run.pack(side="left")
        self._back_btn(btn_row)

    def _browse(self):
        p = filedialog.askdirectory(title="Selecione a pasta de teste")
        if p:
            self._path_var.set(p)

    def _start(self):
        if self._running:
            return
        self._running = True
        self._btn_run.config(state="disabled", text="⏳  Processando…")
        self._log.clear()
        self._pbar.set(0)
        self._status.config(text="Iniciando…", fg=COLORS["pink1"])
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        try:
            test_path = self._path_var.get()
            for p in [MODEL_ESPECIE_PATH, MODEL_RACA_PATH, INFO_ESPECIE_PATH, INFO_RACA_PATH]:
                if not os.path.exists(p):
                    self._log_ui(f"❌ Não encontrado: {p}", "err")
                    self._log_ui("Treine os modelos primeiro.", "warn")
                    self._set_status("❌  Modelos não encontrados.", 0)
                    return

            self._log_ui("Carregando modelos…", "pink")
            model_especie  = YOLO(MODEL_ESPECIE_PATH)
            model_raca     = YOLO(MODEL_RACA_PATH)
            model_deteccao = YOLO("yolov8n.pt")
            self._log_ui("✓ Modelos carregados.", "ok")

            with open(INFO_ESPECIE_PATH, "r", encoding="utf-8") as f:
                info_especie = json.load(f)
            with open(INFO_RACA_PATH, "r", encoding="utf-8") as f:
                info_raca = json.load(f)

            imagens = _coletar_imagens_test(test_path)
            if not imagens:
                self._log_ui("❌ Nenhuma imagem encontrada.", "err")
                self._set_status("❌  Sem imagens.", 0)
                return

            self._log_ui(f"✓ {len(imagens)} imagem(ns) encontrada(s).", "ok")
            resultados = []

            for idx, img_info in enumerate(imagens, 1):
                caminho = img_info["caminho"]
                self._set_status(f"Classificando {idx}/{len(imagens)}: {img_info['nome']}",
                                 (idx - 1) / len(imagens))

                res_esp  = model_especie.predict(caminho, imgsz=IMG_SIZE, verbose=False)[0]
                res_raca = model_raca.predict(caminho,   imgsz=IMG_SIZE, verbose=False)[0]
                res_det  = model_deteccao.predict(caminho, imgsz=640, verbose=False)[0]

                esp_idx   = int(res_esp.probs.top1)
                raca_idx  = int(res_raca.probs.top1)
                conf_esp  = float(res_esp.probs.top1conf)
                conf_raca = float(res_raca.probs.top1conf)
                esp_nome  = info_especie["classes"][esp_idx]
                raca_nome = info_raca["classes"][raca_idx]

                self._log_ui(f"[{idx}] {img_info['nome']}", "white")
                self._log_ui(f"     Espécie : {esp_nome} ({conf_esp*100:.1f}%)", "ok")
                self._log_ui(f"     Raça    : {raca_nome} ({conf_raca*100:.1f}%)", "pink")

                nome_saida = plotar_resultado_pink(
                    caminho=caminho,
                    esp_nome=esp_nome, conf_esp=conf_esp,
                    raca_nome=raca_nome, conf_raca=conf_raca,
                    res_esp=res_esp, res_raca=res_raca, res_det=res_det,
                    info_especie=info_especie, info_raca=info_raca,
                    idx=idx, total=len(imagens),
                )
                self._log_ui(f"     📊 Plot salvo: {nome_saida}", "dim")
                self._update_preview(caminho, esp_nome, conf_esp, raca_nome, conf_raca)

                resultados.append({
                    "imagem": img_info["nome"],
                    "especie_predita": esp_nome,
                    "raca_predita": raca_nome,
                    "confianca_especie": conf_esp,
                    "confianca_raca": conf_raca,
                })

            self._pbar.set(1.0)
            with open("resultados_classificacao.json", "w", encoding="utf-8") as f:
                json.dump(resultados, f, indent=4, ensure_ascii=False)
            self._set_status(f"✅  {len(resultados)} imagem(ns) classificada(s)!", 1.0)
            self._log_ui("Relatório salvo em resultados_classificacao.json", "ok")

        except Exception as e:
            import traceback
            self._log_ui(f"Erro: {e}", "err")
            self._log_ui(traceback.format_exc(), "err")
            self._set_status("❌  Erro durante a classificação.", 0)
        finally:
            self._running = False
            self.after(0, lambda: self._btn_run.config(
                state="normal", text="🔍  Classificar Todas as Imagens"))

    def _update_preview(self, caminho, esp_nome, conf_esp, raca_nome, conf_raca):
        try:
            img = Image.open(caminho).convert("RGB")
            img.thumbnail((160, 160))
            photo = ImageTk.PhotoImage(img)

            def _update():
                for w in self._preview_frame.winfo_children():
                    w.destroy()
                img_lbl = tk.Label(self._preview_frame, image=photo, bg=COLORS["bg3"])
                img_lbl.image = photo
                img_lbl.pack(side="left", padx=(0, 16))
                info = tk.Frame(self._preview_frame, bg=COLORS["bg3"])
                info.pack(side="left", anchor="center")
                tk.Label(info, text=os.path.basename(caminho), font=FONT_LABEL_B,
                         bg=COLORS["bg3"], fg=COLORS["white"]).pack(anchor="w")
                tk.Label(info, text=f"🐾  Espécie:  {esp_nome}  ({conf_esp*100:.1f}%)",
                         font=FONT_LABEL, bg=COLORS["bg3"], fg=COLORS["pink1"]).pack(anchor="w", pady=2)
                tk.Label(info, text=f"🦴  Raça:     {raca_nome}  ({conf_raca*100:.1f}%)",
                         font=FONT_LABEL, bg=COLORS["bg3"], fg=COLORS["pink3"]).pack(anchor="w")

            self.after(0, _update)
        except Exception:
            pass

    def _log_ui(self, msg, tag="white"):
        self.after(0, lambda: self._log.log(msg, tag))

    def _set_status(self, msg, pct):
        self.after(0, lambda: self._status.config(text=msg))
        self.after(0, lambda: self._pbar.set(pct))

    # ---- helpers ----
    def _header(self, title):
        h = tk.Frame(self, bg=COLORS["bg2"])
        h.pack(fill="x")
        tk.Label(h, text=title, font=("Georgia", 17, "bold"),
                 bg=COLORS["bg2"], fg=COLORS["pink1"],
                 padx=30, pady=16).pack(anchor="w")

    def _card(self, parent):
        return tk.Frame(parent, bg=COLORS["bg3"],
                        highlightthickness=1, highlightbackground=COLORS["border"])

    def _section(self, parent, text):
        tk.Label(parent, text=text, font=FONT_LABEL_B,
                 bg=COLORS["bg3"], fg=COLORS["pink2"]).pack(anchor="w", padx=16, pady=(12, 4))

    def _action_btn(self, parent, text, cmd, bg):
        b = tk.Button(parent, text=text, font=FONT_BTN,
                      bg=bg, fg=COLORS["white"], relief="flat",
                      cursor="hand2", padx=28, pady=10, command=cmd,
                      activebackground=COLORS["pink3"], activeforeground=COLORS["bg"])
        b.bind("<Enter>", lambda e: b.config(bg=COLORS["pink3"], fg=COLORS["bg"]))
        b.bind("<Leave>", lambda e: b.config(bg=bg, fg=COLORS["white"]))
        return b

    def _back_btn(self, parent):
        tk.Button(parent, text="← Voltar", font=FONT_LABEL,
                  bg=COLORS["bg3"], fg=COLORS["text_dim"], relief="flat",
                  cursor="hand2", padx=18, pady=10,
                  command=lambda: self.controller.show("MenuPrincipal")).pack(side="right")

# ============================================
# TELA: MÉTRICAS
# ============================================

class MetricasPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=COLORS["bg"])
        self.controller = controller
        self._build()

    def _build(self):
        h = tk.Frame(self, bg=COLORS["bg2"])
        h.pack(fill="x")
        tk.Label(h, text="📊  Métricas do Treinamento",
                 font=("Georgia", 17, "bold"),
                 bg=COLORS["bg2"], fg=COLORS["pink1"],
                 padx=30, pady=16).pack(anchor="w")

        content = tk.Frame(self, bg=COLORS["bg"])
        content.pack(fill="both", expand=True, padx=30, pady=20)

        for nome, projeto in [("Espécie", MODEL_ESPECIE_NAME), ("Raça", MODEL_RACA_NAME)]:
            card = tk.Frame(content, bg=COLORS["bg3"],
                            highlightthickness=1, highlightbackground=COLORS["border"])
            card.pack(fill="x", pady=8)

            tk.Label(card, text=f"Modelo de {nome}", font=FONT_LABEL_B,
                     bg=COLORS["bg3"], fg=COLORS["pink2"]).pack(anchor="w", padx=16, pady=(12, 4))

            results_img = os.path.join(RUNS_DIR, projeto, "treino", "results.png")
            row = tk.Frame(card, bg=COLORS["bg3"])
            row.pack(fill="x", padx=16, pady=(0, 14))

            if os.path.exists(results_img):
                try:
                    img = Image.open(results_img)
                    img.thumbnail((700, 250))
                    photo = ImageTk.PhotoImage(img)
                    lbl = tk.Label(row, image=photo, bg=COLORS["bg3"])
                    lbl.image = photo
                    lbl.pack(side="left")
                except Exception as ex:
                    tk.Label(row, text=f"Erro ao carregar imagem: {ex}",
                             font=FONT_LABEL, bg=COLORS["bg3"], fg=COLORS["error"]).pack()
                btn_open = tk.Button(
                    row, text="🔎 Abrir em tela cheia", font=FONT_SMALL,
                    bg=COLORS["border"], fg=COLORS["pink1"], relief="flat", cursor="hand2",
                    command=lambda p=results_img, n=nome: self._open_full(p, n)
                )
                btn_open.pack(side="left", padx=16, anchor="center")
            else:
                tk.Label(row,
                         text=f"⚠  Gráfico não encontrado.\n({results_img})",
                         font=FONT_LABEL, bg=COLORS["bg3"], fg=COLORS["warning"]).pack()

        btn_row = tk.Frame(self, bg=COLORS["bg"])
        btn_row.pack(fill="x", padx=30, pady=(0, 20))
        tk.Button(btn_row, text="← Voltar", font=FONT_LABEL,
                  bg=COLORS["bg3"], fg=COLORS["text_dim"], relief="flat",
                  cursor="hand2", padx=18, pady=10,
                  command=lambda: self.controller.show("MenuPrincipal")).pack(side="right")

    def _open_full(self, path, nome):
        img = Image.open(path)
        plt.figure(figsize=(14, 6), facecolor=COLORS["bg"])
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Métricas — {nome}", color=COLORS["pink1"], fontsize=15,
                  fontfamily="serif")
        plt.tight_layout()
        plt.show()

# ============================================
# FUNÇÕES DE ML (adaptadas para aceitar log_fn)
# ============================================

def preparar_dataset_especie(train_path, log_fn=print):
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
        val_imgs, train_imgs = imagens[:split], imagens[split:]
        for subset, imgs in [("train", train_imgs), ("val", val_imgs)]:
            dest = os.path.join(base, subset, especie)
            os.makedirs(dest, exist_ok=True)
            for src in imgs:
                shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
    log_fn(f"✓ Espécies: {especies}", "ok")
    return train_dir, val_dir, especies


def preparar_dataset_raca(train_path, log_fn=print):
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
            val_imgs, train_imgs = imagens[:split], imagens[split:]
            for subset, imgs in [("train", train_imgs), ("val", val_imgs)]:
                dest = os.path.join(base, subset, raca)
                os.makedirs(dest, exist_ok=True)
                for src in imgs:
                    shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
    racas = sorted(set(racas))
    log_fn(f"✓ Raças: {len(racas)} encontradas", "ok")
    return train_dir, val_dir, racas


def treinar_modelo(data_dir, nome_projeto, classes, log_fn=print):
    log_fn(f"Treinando: {nome_projeto} ({len(classes)} classes)", "pink")
    model = YOLO("yolov8n-cls.pt")
    model.train(
        data=data_dir,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        project=os.path.join(RUNS_DIR, nome_projeto),
        name="treino",
        exist_ok=True,
        verbose=True,
        plots=True,
    )
    pesos = os.path.join(RUNS_DIR, nome_projeto, "treino", "weights", "best.pt")
    log_fn(f"✓ Pesos salvos: {pesos}", "ok")
    return model


def salvar_info(classes, info_path, nome_projeto):
    info = {
        "classes": classes,
        "num_classes": len(classes),
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tamanho_imagem": IMG_SIZE,
        "epocas": EPOCHS,
        "modelo_path": os.path.join(RUNS_DIR, nome_projeto, "treino", "weights", "best.pt"),
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

# ============================================
# PLOT COM PALETA ROSA/PASTEL
# ============================================

def plotar_resultado_pink(
    caminho, esp_nome, conf_esp, raca_nome, conf_raca,
    res_esp, res_raca, res_det, info_especie, info_raca, idx, total
):
    """Plot com tema rosa pastel."""
    BG      = "#1a1118"
    BG2     = "#231520"
    PINK1   = "#f4a7c3"
    PINK2   = "#e8799e"
    PINK3   = "#f9c9d9"
    PINK4   = "#d4547a"
    ACCENT  = "#fde8f0"
    DIM     = "#8a7085"

    img = np.array(Image.open(caminho).convert("RGB"))

    top5_esp_idx  = res_esp.probs.top5
    top5_esp_conf = res_esp.probs.top5conf.tolist()
    top5_esp_nome = [info_especie["classes"][i] for i in top5_esp_idx]

    top5_raca_idx  = res_raca.probs.top5
    top5_raca_conf = res_raca.probs.top5conf.tolist()
    top5_raca_nome = [info_raca["classes"][i] for i in top5_raca_idx]

    fig = plt.figure(figsize=(19, 7), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # Título geral
    fig.suptitle(
        f"🐾  {esp_nome}  ·  {raca_nome}   —   Imagem {idx}/{total}",
        fontsize=14, color=PINK1, fontfamily="serif", y=0.98
    )

    gs = fig.add_gridspec(1, 3, wspace=0.35, left=0.04, right=0.97,
                           top=0.91, bottom=0.06)
    ax_img   = fig.add_subplot(gs[0, 0])
    ax_esp   = fig.add_subplot(gs[0, 1])
    ax_raca  = fig.add_subplot(gs[0, 2])

    # ---- imagem ----
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_facecolor(BG2)
    ax_img.set_title(os.path.basename(caminho), color=DIM, fontsize=9, pad=6)

    # bounding boxes
    bbox_ok = False
    if res_det.boxes is not None and len(res_det.boxes) > 0:
        ANIMAL_IDS = {15, 16}
        boxes_usar = [b for b in res_det.boxes if int(b.cls) in ANIMAL_IDS] or list(res_det.boxes)
        for box in boxes_usar:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf_b = float(box.conf[0])
            cls_id = int(box.cls[0])
            label  = f"{res_det.names[cls_id]}: {conf_b:.2f}"
            rect   = patches.FancyBboxPatch(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2.5, edgecolor=PINK1, facecolor="none",
                boxstyle="round,pad=2"
            )
            ax_img.add_patch(rect)
            ax_img.text(x1, max(y1-8, 0), label, color=BG, fontsize=8, fontweight="bold",
                        bbox=dict(facecolor=PINK1, edgecolor="none", pad=3, alpha=0.92))
            bbox_ok = True
    if not bbox_ok:
        ax_img.text(0.5, 0.02, "⚠ sem bounding box",
                    transform=ax_img.transAxes, ha="center", va="bottom",
                    color=PINK2, fontsize=8)

    # badges de resultado sob a imagem
    ax_img.text(0.5, -0.05,
                f"Espécie: {esp_nome}  ({conf_esp*100:.1f}%)",
                transform=ax_img.transAxes, ha="center", va="top",
                fontsize=10, color=PINK3, fontweight="bold")
    ax_img.text(0.5, -0.12,
                f"Raça: {raca_nome}  ({conf_raca*100:.1f}%)",
                transform=ax_img.transAxes, ha="center", va="top",
                fontsize=10, color=PINK1)

    # ---- barras espécie ----
    _barras_pink(ax_esp, top5_esp_nome, top5_esp_conf,
                 "Top-5 Espécies", PINK1, PINK2, BG, BG2, DIM)

    # ---- barras raça ----
    _barras_pink(ax_raca, top5_raca_nome, top5_raca_conf,
                 "Top-5 Raças", PINK2, "#9b3057", BG, BG2, DIM)

    nome_saida = f"resultado_{idx:03d}_{Path(caminho).stem}.png"
    plt.savefig(nome_saida, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.show()
    plt.close(fig)
    return nome_saida


def _barras_pink(ax, nomes, confs, titulo, cor_dest, cor_resto, bg, bg2, dim):
    ax.set_facecolor(bg2)
    n = len(nomes)
    y_pos = list(range(n-1, -1, -1))
    cores = [cor_dest if i == 0 else cor_resto for i in range(n)]

    bars = ax.barh(y_pos, confs, color=cores, height=0.52,
                   edgecolor="none", left=0)

    for bar, nome, conf, y in zip(bars, nomes, confs, y_pos):
        ax.text(conf + 0.02, y, f"{conf*100:.1f}%",
                va="center", ha="left", color="#ffffff", fontsize=9, fontweight="bold")
        if conf > 0.25:
            ax.text(0.015, y, nome, va="center", ha="left",
                    color="#ffffff", fontsize=8.5, fontweight="bold", clip_on=True)
        else:
            ax.text(conf + 0.09, y, nome, va="center", ha="left",
                    color=dim, fontsize=8, clip_on=False)

    ax.set_xlim(0, 1.30)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                       color=dim, fontsize=8)
    ax.tick_params(axis="x", colors=dim, length=3)
    ax.tick_params(axis="y", left=False)
    for spine in ax.spines.values():
        spine.set_color("#3d2438")
    ax.set_title(titulo, color=cor_dest, fontsize=12, pad=10,
                 fontweight="bold", fontfamily="serif")

# ============================================
# APP PRINCIPAL
# ============================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Classificador Hierárquico — YOLOv8")
        self.geometry("900x700")
        self.minsize(820, 620)
        self.configure(bg=COLORS["bg"])

        # Tentar ícone (opcional)
        try:
            self.iconbitmap("")
        except Exception:
            pass

        self._pages = {}
        self._container = tk.Frame(self, bg=COLORS["bg"])
        self._container.pack(fill="both", expand=True)

        for PageClass in (MenuPrincipal, TreinoPage, TestePage, MetricasPage):
            name = PageClass.__name__
            page = PageClass(self._container, self)
            self._pages[name] = page
            page.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.show("MenuPrincipal")

    def show(self, name):
        page = self._pages[name]
        page.tkraise()


if __name__ == "__main__":
    app = App()
    app.mainloop()