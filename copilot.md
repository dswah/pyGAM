# GitHub Copilot Instructions for pyGAM

This file provides context and style guidelines for GitHub Copilot or other AI coding assistants when generating code for pyGAM.

## Project Context
- **pyGAM** is a Python library for Generalized Additive Models (GAMs).
- The API is heavily inspired by `scikit-learn` (with methods like `.fit()`, `.predict()`, `.predict_proba()`), but it uses its own class hierarchy for modular additive terms.
- **Dependencies**: Heavily relies on `numpy` and `scipy`. For development, we use `pytest`.

## Architecture & Conventions
- **Model Construction**: GAMs are built by summing Term instances.
  - `s(feature)`: Spline term (for non-linear continuous features).
  - `l(feature)`: Linear term (for linear continuous features).
  - `f(feature)`: Factor term (for categorical features).
  - `te(feature1, feature2)`: Tensor product term (for interactions).
  - Example: `LinearGAM(s(0) + f(1) + te(2, 3))`
- **Hyperparameter Tuning**: Use the built-in `gam.gridsearch(X, y)` rather than `sklearn.model_selection.GridSearchCV` for smooth penalty tuning.
- **Internal State**: Do not mutate internal attributes ending with an underscore (e.g., `gam.coef_` or `term.edge_knots_`). Ensure `get_params` returns independent deep copies to avoid state leakages.

## Code Style & Linting
- **Python Version**: 3.10 to 3.14.
- **Formatter**: Code is formatted using `black` (line-length: 88) and `ruff-format`.
- **Linter**: `ruff` is the primary linter.
- **Docstrings**: We use **NumPy style** docstrings for all modules, classes, and functions (`conventions = "numpy"` in `pyproject.toml`).
- **Pre-commit**: Changes are expected to pass `.pre-commit-config.yaml` checks.

## Testing
- Tests use `pytest` and are located in `pygam/tests/`.
- Ensure new features include assertions for edge cases and exceptions. Use `pytest.raises` for expected exceptions.
- Tests should not have side effects on global states or modify internal estimator states directly.
