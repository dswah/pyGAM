# Contributing to pyGAM

First off, thanks for taking the time to contribute! All types of contributions are encouraged and valued.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)

## Code of Conduct

This project and everyone participating in it is governed by the [pyGAM Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

- Join our [Discord](https://discord.gg/Rt8By5Jj) for discussions and updates
- Check out the [documentation](https://pygam.readthedocs.io/en/latest/)
- Look at [open issues](https://github.com/dswah/pyGAM/issues) for ways to help

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the problem
- Expected behavior vs actual behavior
- Your Python version and operating system
- Any relevant code snippets or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- A clear description of the enhancement
- Why this would be useful to pyGAM users
- Any examples of similar features in other packages

### Working on Issues

Look for issues labeled:
- `good first issue` - good for newcomers
- `bug` - something isn't working
- `enhancement` - new feature or request
- `help wanted` - extra attention is needed

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pyGAM.git
   cd pyGAM
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

## Pull Request Process

1. Create a new branch from `main`:
   ```bash
   git checkout -b your-feature-name
   ```

2. Make your changes, following our coding standards

3. Add or update tests as needed

4. Run the test suite:
   ```bash
   pytest
   ```

5. Commit your changes with a descriptive message:
   ```bash
   git commit -m "Description of your changes"
   ```

6. Push to your fork and submit a pull request to the `main` branch

7. Ensure all CI checks pass

## Coding Standards

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and modular
- Write code comments for complex logic

## Testing

- All new features should include tests
- Bug fixes should include regression tests
- Run tests before submitting a PR:
  ```bash
  pytest
  ```
- For specific test files:
  ```bash
  pytest pygam/tests/test_your_file.py
  ```

## Questions?

Feel free to:
- Open an issue for questions
- Join our [Discord](https://discord.gg/Rt8By5Jj) for discussions

Thank you for contributing to pyGAM!
