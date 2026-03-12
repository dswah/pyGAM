# pyGAM AI Guidance (`agent.md`)

Welcome to pyGAM! If you are using an AI agent (such as ChatGPT, Claude, GitHub Copilot, Cursor, etc.) to help you interact with, contribute to, or debug this repository, please review the following guidelines. Sharing these instructions with your AI, or directing it to read this file, will ensure it generates more accurate, conventional, and contextually appropriate code for pyGAM.

## 1. Core Abstractions & API Style

- **Scikit-Learn like, but distinct**: pyGAM generally follows the estimator API of `scikit-learn` (`fit`, `predict`, `predict_proba`). However, it does not use sklearn's pipeline construction directly; rather, it uses an additive **Term structure**.
- **Model Terms**: AI agents should use `pygam` terms when constructing models:
  - `s(feature_index)` for splines (continuous features).
  - `l(feature_index)` for linear features.
  - `f(feature_index)` for categorical factors.
  - `te(feature_index1, feature_index2)` for tensor products.
  - *Example*: `gam = LinearGAM(s(0) + f(1) + te(2, 3))`
- **Hyperparameter Tuning**: Remind the AI to use pyGAM's built-in `gam.gridsearch(X, y)` rather than `sklearn.model_selection.GridSearchCV`. `gridsearch` handles tuning the smoothing penalty (`lam`) properly.

## 2. Best Practices for Developers using GenAI

- When prompting an LLM about pyGAM bugs or feature requests:
  - **Provide the model summary**: Use `gam.summary()` and supply the output to the AI. This usually helps it identify statistical insignificances or extreme parameter estimations.
  - **State the Python Version**: pyGAM supports Python 3.10 through 3.14. Mentioning this helps avoid deprecated standard library usages.
  - **Reference internal mechanisms precisely**: For parameter cloning, mention that `get_params()` is implemented carefully with `copy.deepcopy()` to avoid reference sharing problems.

## 3. Contributing Code (For the AI)

- **Code Quality & Formatting**: The repository relies heavily on Pre-commit.
  - Formatter: `black` (line length of 88 characters) and `ruff-format`.
  - Linter: `ruff` covers everything from flake8 conventions to docstring definitions.
  - Ensure changes pass `pre-commit run --all-files` locally before committing.
- **Docstrings**: All docstrings *must* follow the NumPy formatting convention. Do not use Sphinx or Google style.
- **Testing**: We use the `pytest` framework. 
  - All new terms or estimator modifications need corresponding cases inside `pygam/tests/`.
  - Do not mutate the internal state of models inappropriately via reference injections. Ensure that parameters fetched via `get_params()` remain independent objects.
