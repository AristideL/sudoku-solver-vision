# src/gui_app.py
from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import mss
import numpy as np
import pyautogui
import os
from datetime import datetime

from .config import AppConfig
from .ocr_easy import EasyOCRReader
from .pipeline import run_detection_pipeline
from .resolver import SudokuGrid, solve_sudoku


def screenshot_bgr() -> np.ndarray:
    with mss.mss() as sct:
        mon = sct.monitors[1]
        shot = sct.grab(mon)  # BGRA
        img = np.array(shot, dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


class SudokuBoard(ttk.Frame):
    """Styled Sudoku display (Canvas) + ability to color digits."""
    def __init__(self, master, size=360, padding=16):
        super().__init__(master)
        self.size = size
        self.padding = padding

        self.canvas = tk.Canvas(self, width=size, height=size, highlightthickness=0)
        self.canvas.pack()

        self._cell = size / 9.0
        self._text_ids = [[None for _ in range(9)] for _ in range(9)]
        self._draw_grid()

    def _draw_grid(self):
        c = self.canvas
        s = self.size
        cell = self._cell

        c.delete("grid")
        c.create_rectangle(0, 0, s, s, fill="white", outline="", tags="grid")

        for i in range(10):
            x = i * cell
            y = i * cell
            c.create_line(x, 0, x, s, width=1, fill="#D0D0D0", tags="grid")
            c.create_line(0, y, s, y, width=1, fill="#D0D0D0", tags="grid")

        for i in range(0, 10, 3):
            x = i * cell
            y = i * cell
            c.create_line(x, 0, x, s, width=3, fill="#202020", tags="grid")
            c.create_line(0, y, s, y, width=3, fill="#202020", tags="grid")

    def clear_numbers(self):
        for r in range(9):
            for c in range(9):
                tid = self._text_ids[r][c]
                if tid is not None:
                    self.canvas.delete(tid)
                    self._text_ids[r][c] = None

    def set_grid(self, grid_9x9: list[list[int]], givens: set[tuple[int, int]] | None = None):
        """
        grid_9x9: values 0..9
        givens: coords (r,c) of detected values (displayed in black). Others in blue.
        """
        self.clear_numbers()
        givens = givens or set()

        cell = self._cell
        for r in range(9):
            for c in range(9):
                v = grid_9x9[r][c]
                if v == 0:
                    continue

                x = (c + 0.5) * cell
                y = (r + 0.5) * cell

                if (r, c) in givens:
                    fill = "#101010"
                    font = ("Segoe UI", 18, "bold")
                else:
                    fill = "#1F5EFF"
                    font = ("Segoe UI", 18, "normal")

                self._text_ids[r][c] = self.canvas.create_text(
                    x, y, text=str(v), fill=fill, font=font
                )


class App(tk.Tk):
    def __init__(self, debug_dir: str | None = None):
        super().__init__()
        self.title("Sudoku CV Solver")
        self.geometry("560x520")
        self.minsize(560, 520)

        # OCR persisted (avoid re-init cost each detect)
        self.ocr = EasyOCRReader(gpu=True, min_conf=0.40)
        self.debug_root = debug_dir  # None or base folder for debug artifacts
        self.run_pipeline = run_detection_pipeline

        self.cfg = AppConfig(
            img_path="",
            gamezone_weights="models/gamezone/best.pt",
            cells_weights="models/cells/best.pt",
            gamezone_class_name="gameZone",
            empty_class_name="emptycells",
        )

        self.detected_grid: list[list[int]] | None = None
        self.detected_givens: set[tuple[int, int]] = set()

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", padding=10)
        style.configure("Status.TLabel", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"))

        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x")
        ttk.Label(header, text="Sudoku CV Solver", style="Title.TLabel").pack(side="left")

        self.status = tk.StringVar(value="Prêt.")
        ttk.Label(root, textvariable=self.status, style="Status.TLabel").pack(anchor="w", pady=(8, 6))

        self.board = SudokuBoard(root, size=380)
        self.board.pack(pady=8)

        controls = ttk.Frame(root)
        controls.pack(fill="x", pady=10)

        self.btn_detect = ttk.Button(controls, text="Détecter", command=self.on_detect)
        self.btn_detect.pack(side="left", expand=True, fill="x", padx=(0, 6))

        self.btn_solve = ttk.Button(controls, text="Résoudre", command=self.on_solve, state="disabled")
        self.btn_solve.pack(side="left", expand=True, fill="x", padx=(6, 0))

        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.pack(fill="x", pady=(6, 0))

    def _set_busy(self, busy: bool, msg: str):
        def apply():
            self.status.set(msg)
            if busy:
                self.btn_detect.config(state="disabled")
                self.btn_solve.config(state="disabled")
                self.progress.start(12)
            else:
                self.btn_detect.config(state="normal")
                self.btn_solve.config(state=("normal" if self.detected_grid else "disabled"))
                self.progress.stop()
        self.after(0, apply)

    # --- Fast filling (zig-zag to avoid returning left) ---
    def fill_sudoku_com(self, solved_grid: list[list[int]], givens: set[tuple[int, int]]):
        """
        Requires sudoku.com window focused and the current selection on the grid.
        Strategy: zig-zag rows => avoids 8x LEFT at end of each row.
        Types only non-given cells.
        """
        # Let user focus the browser
        time.sleep(2.0)

        interval = 0.0001

        for r in range(9):
            if r % 2 == 0:
                cols = range(9)
                move = "right"
            else:
                cols = range(8, -1, -1)
                move = "left"

            for i, c in enumerate(cols):
                if (r, c) not in givens:
                    pyautogui.press(str(solved_grid[r][c]), interval=interval)
                if i < 8:
                    pyautogui.press(move, interval=interval)

            if r < 8:
                pyautogui.press("down", interval=interval)

    def on_detect(self):
        threading.Thread(target=self._detect_worker, daemon=True).start()

    def _detect_worker(self):
        self._set_busy(True, "Capture écran…")
        try:
            bgr = screenshot_bgr()

            self._set_busy(True, "Détection en cours…")
            mapping = self.run_pipeline(self.cfg, bgr, self.ocr, debug_dir="debug/pipeline_v1")

            grid = [[int(x) for x in row.split()] for row in mapping.splitlines()]
            givens = {(r, c) for r in range(9) for c in range(9) if grid[r][c] != 0}

            self.detected_grid = grid
            self.detected_givens = givens

            self.after(0, lambda: self.board.set_grid(grid, givens=givens))
            self._set_busy(False, "Grille détectée. Vous pouvez résoudre.")
        except Exception as e:
            self.detected_grid = None
            self.detected_givens = set()
            self.after(0, self.board.clear_numbers)
            self._set_busy(False, f"Erreur détection: {e}")

    def on_solve(self):
        if not self.detected_grid:
            return
        threading.Thread(target=self._solve_worker, daemon=True).start()

    def _solve_worker(self):
        self._set_busy(True, "Résolution…")
        try:
            base = [row[:] for row in self.detected_grid]
            sudoku = SudokuGrid(base)

            solved = solve_sudoku(sudoku)
            out = solved if isinstance(solved, SudokuGrid) else sudoku
            solved_grid = out.grid if hasattr(out, "grid") else out

            # Update Tk board
            self.after(0, lambda: self.board.set_grid(solved_grid, givens=self.detected_givens))

            # Auto-fill sudoku.com
            self._set_busy(False, "Grille résolue. Remplissage sudoku.com…")
            self.fill_sudoku_com(solved_grid, self.detected_givens)

            self._set_busy(False, "Grille résolue + remplie sur sudoku.com.")
        except Exception as e:
            self._set_busy(False, f"Erreur résolution: {e}")


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
