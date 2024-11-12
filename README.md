# Installation process

## Install uv package manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Clone this hub repository

```bash
git clone https://github.com/Gianzanti/ia_research
```

## Exec uv sync to install all dependencies

```bash
uv sync
```

## Test the instalation

```bash
uv run src/main.py -m check

## Expected output
Namespace(algo='PPO', mode='check', model='100000')
Checking environment...
Gym version: 1.0.0
Stable Baselines version: 2.4.0a11
Mujoco version: 3.2.5
Environment check successful!

```
