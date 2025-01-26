#!/bin/zsh

cd "$(dirname "$0")"
source .venv/bin/activate

uv pip install -r requirements.txt

python main.py
