# Contributing

Use a separate Git branch and a pull request for each change. Keep contributions
focused and describe the behavior, validation, and remaining limitations. This
repository's existing [MIT license](LICENSE) and copyright notice must remain
intact; document the origin and license of any additional code or dependency.

## Development environment

Use Python 3.10 or newer and a virtual environment named `.venv` in the repository
root. Start from your own clone or an independently authorized server checkout.

```sh
python --version
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS. If activation is unavailable, use the
virtual environment's Python executable directly. Install the development tools:

```sh
python -m pip install -e ".[dev]"
python -m pure_pino doctor
python -m pure_pino example
```

In VS Code, open `TRICO-R-PINO.code-workspace`, then use **Python: Select
Interpreter** to select `.venv`. The checked-in tasks and debug configurations use
that selected interpreter. They do not configure a server connection.

## Branch and review workflow

From an up-to-date, clean checkout of the repository's default branch:

```sh
git pull --ff-only
git switch -c feature/my-change
```

Make the change, select the intended source/documentation files for staging, and
inspect the staged diff. Commit and push your own branch, then open a pull request:

```sh
git diff --cached
git commit -m "Describe the change"
git push -u origin feature/my-change
```

Pushing to this repository requires your own GitHub account with repository write
access. If you do not have it, ask the owner to grant access or use a fork and a
pull request. Do not use another person's credentials. The invitation procedure,
Remote-SSH workflow, and Live Share sessions are described in
[Collaboration](docs/COLLABORATION.md).

## Checks before a pull request

Run these from the repository root with `.venv` selected:

```sh
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

Use `python -m ruff format .` to apply formatting when needed. Report the actual
commands and results in the pull request, including any unavailable checks.
Add a regression test when a change fixes a meaningful behavior or interface.
For changes to scientific code, distinguish manufactured software controls from
real-data or native-physics validation; passing software tests alone does not
establish accuracy, convergence, conservation, or full-population qualification.

## Data and information boundaries

- Keep credentials, SSH configuration, private logs, original lake observations,
  checkpoints, and environment-specific absolute paths out of commits and issues.
- Use generated or explicitly redistributable fixtures. Preserve their provenance
  and state their validation scope.
- Internal evaluation temperatures/depths must not become model features,
  initialization, normalization, or model-selection inputs. Authorized source
  targets belong in the training loss; evaluation readout and scoring have their
  own frozen information boundary.
- Preserve original roles, physical-lake/alias grouping, clock semantics, and
  failure denominators. Disclose any proposed contract change in the pull request.
- Keep the primary inverse-model forward direct. Introducing a legacy temperature
  integrator into that forward requires an explicit architecture discussion.

Keep experiment outputs in a separate local output directory and review the
staged files before every commit. Public bug reports should use a minimal,
sanitized reproduction rather than private run archives.
