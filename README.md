# Mahjong AI Agent

An AI system for decision-making in Mahjong, a 4-player imperfect-information stochastic game.

## Features

- 34-tile vector state representation
- Opponent modeling via discards + melds
- Heuristic AI baseline (shanten minimization)
- Monte Carlo planning (in progress)
- Planned ML model trained on real game logs (Tenhou dataset)

## Core Idea

The AI makes decisions under uncertainty using:
- probabilistic reasoning
- simulation-based planning
- structured feature engineering

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate  # Git Bash
pip install -r requirements.txt