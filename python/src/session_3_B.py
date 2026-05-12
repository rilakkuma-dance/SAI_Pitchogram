"""
Mandarin Tone Identification Experiment — Group A
Plays pitchogram videos (up to 2x) and collects tone responses.

Usage:
    python tone_experiment.py   (no extra libraries needed)

Video files live in a 'stimuli/' folder next to this script.
Filename format:  <Chinese_or_English_word>_<m|f>.mp4
  e.g.  书包_f.mp4 · 好_f.mp4 · 中国_m.mp4

Adjust the video_name field in TRIALS below to match your exact filenames.
Results are saved to: results_<ParticipantID>_<timestamp>.csv
"""

import csv
import os
import sys
import time
import datetime
import subprocess
import platform
import tkinter as tk
from tkinter import messagebox

# ── Group B Trial list ───────────────────────────────────────────────────────
# 5 one-syllable + 5 two-syllable · Tones 1–4 each appear 3–4×
# video_name: stem of the file in stimuli/ (without .mp4)
GROUP = "B"
TRIALS = [
    {"id": 1,  "video_name": "好_f",     "pinyin": "hǎo",      "meaning": "good",          "syllables": 1, "correct": "3"},
    {"id": 2,  "video_name": "电话_f",   "pinyin": "diànhuà",  "meaning": "telephone",     "syllables": 2, "correct": "4-4"},
    {"id": 3,  "video_name": "花_m",     "pinyin": "huā",      "meaning": "flower",        "syllables": 1, "correct": "1"},
    {"id": 4,  "video_name": "地图_m",   "pinyin": "dìtú",     "meaning": "map",           "syllables": 2, "correct": "4-2"},
    {"id": 5,  "video_name": "马_m",     "pinyin": "mǎ",       "meaning": "horse",         "syllables": 1, "correct": "3"},
    {"id": 6,  "video_name": "中文_m",   "pinyin": "zhōngwén", "meaning": "Chinese lang.", "syllables": 2, "correct": "1-2"},
    {"id": 7,  "video_name": "汤_m",     "pinyin": "tāng",     "meaning": "soup",          "syllables": 1, "correct": "1"},
    {"id": 8,  "video_name": "白酒_f",   "pinyin": "báijiǔ",   "meaning": "baijiu",        "syllables": 2, "correct": "2-3"},
    {"id": 9,  "video_name": "费_m",     "pinyin": "fèi",      "meaning": "cost",          "syllables": 1, "correct": "4"},
    {"id": 10, "video_name": "苹果_m",   "pinyin": "píngguǒ",  "meaning": "apple",         "syllables": 2, "correct": "2-3"},
]

TONE_SPECS = {
    1: {"shape": "―", "color": "#1a7a3c", "label": "High Level"},
    2: {"shape": "╱", "color": "#1a5ea8", "label": "Rising"},
    3: {"shape": "∨", "color": "#b85c00", "label": "Falling-Rising"},
    4: {"shape": "╲", "color": "#b81c1c", "label": "Falling"},
}

VIDEO_DIR = r"C:\Users\z5718263\SAI_Pitchogram\python\src\session3_stimuli"
MAX_PLAYS = 2


# ── Video player (cross-platform) ─────────────────────────────────────────────

# Common VLC install paths on Windows
VLC_PATHS = [
    r"C:\Program Files\VideoLAN\VLC\vlc.exe",
    r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe",
]

def find_vlc() -> str | None:
    """Return path to VLC executable, or None if not found."""
    for p in VLC_PATHS:
        if os.path.isfile(p):
            return p
    # Also try if vlc is on PATH
    try:
        result = subprocess.run(["where", "vlc"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return None

def play_video(video_name: str) -> bool:
    path = os.path.join(VIDEO_DIR, f"{video_name}.mp4")
    if not os.path.isfile(path):
        return False
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", path])
        elif system == "Windows":
            vlc = find_vlc()
            if vlc:
                # --play-and-exit closes VLC when video ends
                subprocess.Popen([vlc, "--play-and-exit", "--qt-minimal-view",
                                   "--no-video-title-show", path])
            else:
                # Fallback to default player
                os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path])
        return True
    except Exception:
        return False


# ── GUI ───────────────────────────────────────────────────────────────────────

C = {
    "bg":      "#f4f6f9",
    "surface": "#ffffff",
    "border":  "#c8cdd8",
    "accent":  "#1a5ea8",
    "text":    "#1a1d27",
    "muted":   "#5a6170",
    "correct": "#1a7a3c",
    "wrong":   "#b81c1c",
}
TONE_CLR = {t: TONE_SPECS[t]["color"] for t in range(1, 5)}


class ToneExperiment(tk.Tk):

    def __init__(self, participant_id: str):
        super().__init__()
        self.participant_id = participant_id
        self.results: list[dict] = []
        self.trial_index = 0
        self.plays_used = 0
        self.response_slots: list[tk.StringVar] = []
        self.start_time: float = 0.0

        self.title(f"Mandarin Tone Identification — Group {GROUP}")
        self.configure(bg=C["bg"])
        self.resizable(False, False)
        self.geometry("920x740")

        self._build_reference_bar()
        self._build_trial_card()
        self._build_response_area()
        self._build_nav()
        self._load_trial()

    # ── Reference bar ─────────────────────────────────────────────────────────

    def _build_reference_bar(self):
        outer = tk.Frame(self, bg=C["bg"])
        outer.pack(fill="x", padx=22, pady=(14, 6))

        tk.Label(outer, text="TONE REFERENCE  (always visible)",
                 font=("Arial", 9), bg=C["bg"], fg=C["muted"]).pack(anchor="w")

        bar = tk.Frame(outer, bg=C["surface"],
                       highlightbackground=C["border"], highlightthickness=1)
        bar.pack(fill="x", pady=(4, 0))

        for t in range(1, 5):
            spec = TONE_SPECS[t]
            clr  = spec["color"]
            cell = tk.Frame(bar, bg=C["surface"], padx=14, pady=10)
            cell.pack(side="left", expand=True, fill="both")

            # Big coloured number + shape badge  (e.g.  "1  ―")
            tk.Label(cell,
                     text=f"{t}  {spec['shape']}",
                     font=("Arial", 26, "bold"),
                     bg=C["surface"], fg=clr).pack(anchor="w")

            # Label line
            tk.Label(cell,
                     text=spec["label"],
                     font=("Arial", 10, "bold"),
                     bg=C["surface"], fg=clr).pack(anchor="w", pady=(2, 0))


            if t < 4:
                tk.Frame(bar, bg=C["border"], width=1).pack(side="left", fill="y")

    # ── Trial card ────────────────────────────────────────────────────────────

    def _build_trial_card(self):
        outer = tk.Frame(self, bg=C["bg"])
        outer.pack(fill="x", padx=22, pady=4)

        self.progress_var = tk.StringVar()
        tk.Label(outer, textvariable=self.progress_var,
                 font=("Arial", 10), bg=C["bg"], fg=C["muted"]).pack(anchor="e")

        card = tk.Frame(outer, bg=C["surface"],
                        highlightbackground=C["border"], highlightthickness=1)
        card.pack(fill="x")

        inner = tk.Frame(card, bg=C["surface"], padx=20, pady=14)
        inner.pack(fill="x")

        tk.Label(inner,
                 text="Listen to the word, then answer below.",
                 font=("Arial", 13, "italic"),
                 bg=C["surface"], fg=C["muted"]).pack(anchor="w")

    # ── Response slots ────────────────────────────────────────────────────────

    def _build_response_area(self):
        self.resp_outer = tk.Frame(self, bg=C["bg"])
        self.resp_outer.pack(fill="x", padx=22, pady=8)

        # ── Step 1: syllable count guess ──────────────────────────────────────
        tk.Label(self.resp_outer,
                 text="Step 1 — How many syllables?",
                 font=("Arial", 12, "bold"), bg=C["bg"], fg=C["text"]).pack(anchor="w")

        syl_row = tk.Frame(self.resp_outer, bg=C["bg"])
        syl_row.pack(anchor="w", pady=(6, 0))

        self.syl_guess = tk.IntVar(value=0)  # 0 = not chosen yet

        for n, lbl in [(1, "1  syllable"), (2, "2  syllables")]:
            btn = tk.Button(
                syl_row, text=lbl,
                font=("Arial", 16, "bold"),
                bg=C["surface"], fg=C["accent"],
                activebackground=C["accent"], activeforeground="white",
                relief="flat", padx=16, pady=8,
                highlightbackground=C["border"], highlightthickness=1,
                cursor="hand2",
                command=lambda n=n: self._pick_syllables(n)
            )
            btn.pack(side="left", padx=(0, 10))
        self.syl_buttons = syl_row.winfo_children()

        # ── Step 2: tone slots (hidden until syllable chosen) ─────────────────
        self.tone_section = tk.Frame(self.resp_outer, bg=C["bg"])
        # not packed yet — shown after syllable pick

        tk.Label(self.tone_section,
                 text="Step 2 — Select tone(s):",
                 font=("Arial", 12, "bold"), bg=C["bg"], fg=C["text"]).pack(anchor="w", pady=(14, 0))

        self.slots_frame = tk.Frame(self.tone_section, bg=C["bg"])
        self.slots_frame.pack(anchor="w", pady=(8, 0))

        self.feedback_var = tk.StringVar()
        self.feedback_lbl = tk.Label(self.resp_outer,
                                     textvariable=self.feedback_var,
                                     font=("Arial", 13, "bold"),
                                     bg=C["bg"], fg=C["text"])
        self.feedback_lbl.pack(anchor="w", pady=(8, 0))

    def _pick_syllables(self, n: int):
        self.syl_guess.set(n)
        # Highlight chosen button
        for btn in self.syl_buttons:
            lbl = btn.cget("text")
            chosen = lbl.startswith(str(n))
            btn.config(
                bg=C["accent"] if chosen else C["surface"],
                fg="#0f1117" if chosen else C["accent"]
            )
        # Show tone section with correct number of slots
        self.tone_section.pack(fill="x")
        self._build_slots(n)
        self._update_submit()

    def _build_slots(self, n: int):
        for w in self.slots_frame.winfo_children():
            w.destroy()
        self.response_slots = []

        for s in range(n):
            col = tk.Frame(self.slots_frame, bg=C["bg"])
            col.pack(side="left", padx=(0, 20))

            tk.Label(col, text=f"Syllable {s+1}",
                         font=("Arial", 9), bg=C["bg"], fg=C["muted"]).pack()

            var = tk.StringVar(value="")
            self.response_slots.append(var)

            row = tk.Frame(col, bg=C["bg"])
            row.pack()

            for t in range(1, 5):
                spec = TONE_SPECS[t]
                btn = tk.Button(
                    row, text=f"{t}  {spec['shape']}",
                    font=("Arial", 18, "bold"),
                    bg=C["surface"], fg=spec["color"],
                    activebackground=spec["color"], activeforeground="white",
                    relief="flat", padx=10, pady=6,
                    highlightbackground=C["border"], highlightthickness=1,
                    cursor="hand2",
                    command=lambda v=var, val=str(t): self._pick(v, val)
                )
                btn.pack(side="left", padx=4)

            # shows chosen tone + shape
            def _make_display(col=col, var=var):
                lbl = tk.Label(col, text="",
                               font=("Arial", 22, "bold"),
                               bg=C["bg"], fg=C["accent"], width=6)
                lbl.pack(pady=(6, 0))
                def _update(*_):
                    v = var.get()
                    if v and v.isdigit():
                        t = int(v)
                        lbl.config(text=f"{t}  {TONE_SPECS[t]['shape']}",
                                   fg=TONE_SPECS[t]["color"])
                    else:
                        lbl.config(text="", fg=C["accent"])
                var.trace_add("write", _update)
            _make_display()

    def _pick(self, var: tk.StringVar, val: str):
        var.set(val)
        self._update_submit()

    # ── Nav bar ───────────────────────────────────────────────────────────────

    def _build_nav(self):
        nav = tk.Frame(self, bg=C["bg"])
        nav.pack(fill="x", padx=22, pady=12)

        self.play_btn = tk.Button(
            nav, text="▶  Play Video",
            command=self._play,
            font=("Arial", 12, "bold"),
            bg=C["accent"], fg="white",
            activebackground="#3a82d6",
            relief="flat", padx=20, pady=10, cursor="hand2"
        )
        self.play_btn.pack(side="left", padx=(0, 12))

        self.plays_lbl = tk.Label(nav, text="",
                                  font=("Arial", 10),
                                  bg=C["bg"], fg=C["muted"])
        self.plays_lbl.pack(side="left", padx=(0, 20))

        self.submit_btn = tk.Button(
            nav, text="Submit  →",
            command=self._submit,
            font=("Arial", 12, "bold"),
            bg=C["border"], fg=C["muted"],
            relief="flat", padx=20, pady=10,
            state="disabled"
        )
        self.submit_btn.pack(side="right")

    # ── Trial lifecycle ───────────────────────────────────────────────────────

    def _load_trial(self):
        trial = TRIALS[self.trial_index]
        self.plays_used = 0
        self.feedback_var.set("")
        self.start_time = time.time()

        self.progress_var.set(f"Trial {self.trial_index + 1} / {len(TRIALS)}")

        # Reset syllable guess and hide tone section
        self.syl_guess.set(0)
        for btn in self.syl_buttons:
            btn.config(bg=C["surface"], fg=C["accent"])
        self.tone_section.pack_forget()
        for w in self.slots_frame.winfo_children():
            w.destroy()
        self.response_slots = []

        self._update_play_btn()
        self._update_submit()

    def _update_play_btn(self):
        remaining = MAX_PLAYS - self.plays_used
        if remaining > 0:
            self.play_btn.config(
                state="normal",
                text=f"▶  Play Video  ({remaining} play{'s' if remaining > 1 else ''} left)")
        else:
            self.play_btn.config(state="disabled", text="▶  No plays remaining")
        self.plays_lbl.config(text=f"{self.plays_used}/{MAX_PLAYS} used")

    def _update_submit(self):
        ready = (self.syl_guess.get() != 0 and
                 len(self.response_slots) > 0 and
                 all(v.get() for v in self.response_slots))
        if ready:
            self.submit_btn.config(
                state="normal", bg=C["accent"], fg="white",
                activebackground="#3a82d6")
        else:
            self.submit_btn.config(
                state="disabled", bg=C["border"], fg=C["muted"])

    def _play(self):
        if self.plays_used >= MAX_PLAYS:
            return
        trial = TRIALS[self.trial_index]
        ok = play_video(trial["video_name"])
        if not ok:
            messagebox.showwarning(
                "File not found",
                f"Could not open:\n  {VIDEO_DIR}/{trial['video_name']}.mp4\n\n"
                "Make sure all video files are in the 'stimuli/' folder "
                "and filenames match the video_name field in TRIALS.")
        self.plays_used += 1
        self._update_play_btn()

    def _submit(self):
        trial = TRIALS[self.trial_index]
        response = "-".join(v.get() for v in self.response_slots)
        correct = trial["correct"]
        syl_guess = self.syl_guess.get()
        syl_correct = trial["syllables"]
        syl_right = int(syl_guess == syl_correct)
        tone_right = int(response == correct and syl_right)
        is_correct = bool(tone_right)
        rt = round(time.time() - self.start_time, 3)

        self.results.append({
            "participant_id":           self.participant_id,
            "trial_number":             self.trial_index + 1,
            "trial_id":                 trial["id"],
            "video_name":               trial["video_name"],
            "pinyin":                   trial["pinyin"],
            "meaning":                  trial["meaning"],
            "syllables_correct":        syl_correct,
            "syllables_guessed":        syl_guess,
            "syllables_right":          syl_right,
            "correct_tones":            correct,
            "participant_tones":        response,
            "tones_correct":            int(response == correct and syl_right),
            "fully_correct":            int(is_correct),
            "plays_used":               self.plays_used,
            "response_time_s":          rt,
            "timestamp":                datetime.datetime.now().isoformat(),
        })

        if is_correct:
            self.feedback_var.set("✓  Correct!")
            self.feedback_lbl.config(fg=C["correct"])
        elif not syl_right:
            self.feedback_var.set(
                f"✗  It was {syl_correct} syllable{'s' if syl_correct > 1 else ''}  |  tones: {correct}")
            self.feedback_lbl.config(fg=C["wrong"])
        else:
            self.feedback_var.set(f"✗  Correct tones: {correct}")
            self.feedback_lbl.config(fg=C["wrong"])

        self.submit_btn.config(state="disabled")
        self.play_btn.config(state="disabled")
        self.after(1400, self._next_trial)

    def _next_trial(self):
        self.trial_index += 1
        if self.trial_index >= len(TRIALS):
            self._finish()
        else:
            self._load_trial()

    def _finish(self):
        n_correct = sum(r["fully_correct"] for r in self.results)
        total = len(self.results)
        saved_path = self._save_results()
        messagebox.showinfo(
            "Session complete",
            f"All {total} trials finished!\n"
            f"Score: {n_correct} / {total}  "
            f"({100 * n_correct // total}%)\n\n"
            f"Results saved to:\n  {saved_path}")
        self.destroy()

    def _save_results(self) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save next to the script file so it's always easy to find
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fname = os.path.join(script_dir,
                             f"results_Group{GROUP}_{self.participant_id}_{ts}.csv")
        fields = [
            "participant_id", "trial_number", "trial_id", "video_name",
            "pinyin", "meaning",
            "syllables_correct", "syllables_guessed", "syllables_right",
            "correct_tones", "participant_tones", "tones_correct",
            "fully_correct", "plays_used", "response_time_s", "timestamp",
        ]
        try:
            with open(fname, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(self.results)
                f.flush()
                os.fsync(f.fileno())
            print(f"Saved → {fname}")
            return fname
        except Exception as e:
            messagebox.showerror("Save error",
                                 "Could not save results:\n" + str(e) +
                                 "\n\nAttempted path:\n" + fname)
            return ""


# ── Participant ID dialog ─────────────────────────────────────────────────────

def ask_participant_id() -> str | None:
    root = tk.Tk()
    root.withdraw()

    dlg = tk.Toplevel(root)
    dlg.title("Start Experiment")
    dlg.configure(bg=C["bg"])
    dlg.resizable(False, False)
    dlg.grab_set()

    tk.Label(dlg, text="Participant ID",
             font=("Arial", 14, "bold"),
             bg=C["bg"], fg=C["text"],
             pady=14, padx=24).pack()

    pid_var = tk.StringVar(value="P01")
    entry = tk.Entry(dlg, textvariable=pid_var,
                     font=("Arial", 14), width=10, justify="center",
                     bg=C["surface"], fg=C["text"],
                     insertbackground=C["text"],
                     relief="flat",
                     highlightbackground=C["border"],
                     highlightthickness=1)
    entry.pack(padx=24, pady=(0, 14))

    result = []

    def start():
        pid = pid_var.get().strip()
        if pid:
            result.append(pid)
            dlg.destroy()
            root.destroy()

    tk.Button(dlg, text="Start  ▶",
              command=start,
              font=("Arial", 12, "bold"),
              bg=C["accent"], fg="white",
              activebackground="#3a82d6",
              relief="flat", padx=16, pady=8).pack(pady=(0, 18))

    entry.focus_set()
    entry.bind("<Return>", lambda e: start())
    root.wait_window(dlg)
    return result[0] if result else None


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Warn early if on Windows and VLC is missing
    if platform.system() == "Windows" and find_vlc() is None:
        root = tk.Tk(); root.withdraw()
        messagebox.showwarning(
            "VLC not found",
            "VLC Media Player was not found on this computer.\n\n"
            "Without VLC, .mp4 files may fail to open due to codec issues.\n\n"
            "Please install VLC from:\n  https://www.videolan.org/vlc/\n\n"
            "The experiment will continue but video playback may not work.")
        root.destroy()

    pid = ask_participant_id()
    if pid:
        app = ToneExperiment(pid)
        app.mainloop()