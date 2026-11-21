"""Terminal ANSI Spectrum Visualizer & Real-Time Mood HUD."""

import os
import sys
import numpy as np

BAR_BLOCKS = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
BAR_HEIGHT = 16

def get_gradient_color(ratio: float) -> str:
    """Returns 24-bit TrueColor ANSI escape sequence along an audio-reactive gradient."""
    # Cyan (0, 229, 255) -> Green (0, 230, 118) -> Yellow (255, 214, 0) -> Red/Pink (255, 64, 129)
    if ratio < 0.33:
        r = 0
        g = int(229 + (230 - 229) * (ratio / 0.33))
        b = int(255 * (1.0 - ratio / 0.33))
    elif ratio < 0.66:
        sub = (ratio - 0.33) / 0.33
        r = int(255 * sub)
        g = int(230 - 16 * sub)
        b = 0
    else:
        sub = (ratio - 0.66) / 0.34
        r = 255
        g = int(214 * (1.0 - sub) + 64 * sub)
        b = int(129 * sub)

    return f"\033[38;2;{r};{g};{b}m"

class TerminalVisualizer:
    def __init__(self, num_bands: int = 48):
        self.num_bands = num_bands
        self.active_mood = "Analyzing..."
        self.mood_probs = [0.25, 0.25, 0.25, 0.25]
        self.mood_labels = ["High-Energy", "Chill / Lo-Fi", "Dark / Cinematic", "Upbeat"]

    def update_mood(self, probs: list[float], top_mood: str):
        self.mood_probs = probs
        self.active_mood = top_mood

    def render_frame(self, energies: np.ndarray, peaks: np.ndarray):
        """Renders 48-band spectrum with peak-hold caps and mood confidence HUD."""
        lines = []

        # Cursor home
        lines.append("\033[H")

        # Header
        lines.append(
            "\033[1;36m┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\033[0m\r\n"
            f"\033[1;36m┃\033[0m \033[1mBEAT-LENS\033[0m │ 48-Band Spectrum Analyzer │ Predicted Mood: \033[1;33m{self.active_mood:<18}\033[0m \033[1;36m┃\033[0m\r\n"
        )

        # Mood Confidence Bars
        mood_str = "┃ "
        colors = ["\033[1;31m", "\033[1;36m", "\033[1;35m", "\033[1;32m"]
        for i, (name, p) in enumerate(zip(self.mood_labels, self.mood_probs)):
            bars = "■" * int(p * 8)
            mood_str += f"{name[:5]}: {colors[i]}{int(p*100):2d}%\033[0m {bars:<8} "
        mood_str += "┃\r\n"
        lines.append(mood_str)
        lines.append("\033[1;36m┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\033[0m\r\n")

        # Render Vertical Spectrum Bars (top down from BAR_HEIGHT to 1)
        for h in range(BAR_HEIGHT, 0, -1):
            row = "┃ "
            ratio = h / BAR_HEIGHT
            col_code = get_gradient_color(ratio)

            for i in range(self.num_bands):
                val = energies[i] * BAR_HEIGHT
                peak_val = peaks[i] * BAR_HEIGHT

                if abs(peak_val - h) < 0.6 and peak_val > val:
                    # Peak hold cap
                    row += "\033[1;37m▔\033[0m"
                elif val >= h:
                    row += f"{col_code}█\033[0m"
                elif val >= h - 1:
                    frac = val - (h - 1)
                    idx = int(frac * 8)
                    char = BAR_BLOCKS[max(0, min(8, idx))]
                    row += f"{col_code}{char}\033[0m"
                else:
                    row += " "

            row += " ┃\r\n"
            lines.append(row)

        # Bottom Frequency Scale
        lines.append(
            "\033[1;36m┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\033[0m\r\n"
            "  20 Hz           100 Hz          500 Hz           2 kHz           8 kHz          16 kHz   \033[K\r\n"
        )

        sys.stdout.write("".join(lines))
        sys.stdout.flush()
