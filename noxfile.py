"""Nox sessions for linting, testing, and type checking."""

import nox

# Default sessions to run when you just type `nox`
nox.options.sessions = ["lint", "tests", "mypy"]

PYTHON_VERSIONS = ["3.11"]
LOCATIONS = "src", "tests", "noxfile.py"


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install("-e", ".[all,dev]")
    session.run("pytest", "--cov=gpt2_framework", *session.posargs)


@nox.session(python=PYTHON_VERSIONS[0])
def lint(session: nox.Session) -> None:
    """Lint with Ruff."""
    session.install("ruff")
    session.run("ruff", "check", *LOCATIONS)
    session.run("ruff", "format", "--check", *LOCATIONS)


@nox.session(python=PYTHON_VERSIONS[0])
def fmt(session: nox.Session) -> None:
    """Auto-format code with Ruff."""
    session.install("ruff")
    session.run("ruff", "check", "--fix", *LOCATIONS)
    session.run("ruff", "format", *LOCATIONS)


@nox.session(python=PYTHON_VERSIONS[0])
def mypy(session: nox.Session) -> None:
    """Type-check with mypy."""
    session.install("-e", ".[all,dev]")
    session.run("mypy", "src/gpt2_framework/")
