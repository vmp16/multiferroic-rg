"""Lightweight console utility."""

__all__ = ["progress_bar"]


def progress_bar(current, total, width=40):
    FILL  = "█"  # █
    EMPTY = "░"  # ░

    filled = int(width * current / total)
    bar = FILL * filled + EMPTY * (width - filled)
    pct = current / total * 100
    print(f"\r {bar}  {pct:.1f}%", end="", flush=True)
