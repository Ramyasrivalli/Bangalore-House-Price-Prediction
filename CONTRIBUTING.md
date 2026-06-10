# Contributing to Bangalore House Price Prediction

Thank you for considering a contribution! Below are the guidelines to ensure smooth collaboration.

## Development Setup

```bash
git clone https://github.com/Ramyasrivalli/Bangalore-House-Price-Prediction.git
cd Bangalore-House-Price-Prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Code Standards

- **Formatter**: `black .`
- **Linter**: `flake8 src/ tests/`
- **Import sorting**: `isort src/ tests/`
- **Type hints** required for all public functions.
- **Docstrings** required (NumPy style preferred).

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

All tests must pass before opening a pull request.

## Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, tagged releases |
| `dev` | Integration branch |
| `feature/<name>` | New features |
| `fix/<name>` | Bug fixes |

## Pull Request Checklist

- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] Code formatted (`black .` and `isort .`)
- [ ] Docstrings added for new public functions
- [ ] `CHANGELOG.md` updated (if applicable)
- [ ] PR title follows pattern: `feat:`, `fix:`, `docs:`, `refactor:`

## Reporting Issues

Please use [GitHub Issues](https://github.com/Ramyasrivalli/Bangalore-House-Price-Prediction/issues)
and include:

1. Exact error message / traceback
2. Python version and OS
3. Steps to reproduce
4. Expected vs actual behaviour
