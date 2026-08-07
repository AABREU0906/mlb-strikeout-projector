#!/usr/bin/env python3
"""MLB Pitcher Strikeout Projection System - entrypoint.

Usage:
    python main.py                  # interactive daily workflow
    python main.py project          # same, explicitly
    python main.py update-results   # fetch actual results for completed games
    python main.py evaluate         # model evaluation reports
    python main.py retrain          # retrain the model on stored history
    python main.py history          # browse historical projections
    python main.py backtest --start-date ... --end-date ...
"""
from app.cli.main_app import app

if __name__ == "__main__":
    app()
