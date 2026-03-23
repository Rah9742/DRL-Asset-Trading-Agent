"""Compatibility wrapper for the full comparison runner."""

from __future__ import annotations

from .run_full_comparison import main, run_full_comparison

__all__ = ["main", "run_full_comparison"]


if __name__ == "__main__":
    main()
