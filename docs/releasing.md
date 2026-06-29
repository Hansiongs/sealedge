# Release Process

## Publishing to PyPI

The repository has a `publish.yml` workflow that uses **trusted publishing**
(OIDC) — no API tokens required. You only need to configure the trust
once in the PyPI dashboard.

### One-time PyPI Setup

1. Log in to [pypi.org](https://pypi.org) (create an account if needed).

2. Go to your account settings → "Publishing" →
   "Add a new pending publisher".

3. Fill in:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `quant_lib` |
   | Owner | `Hansiongs` |
   | Repository name | `hans-backtest` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

4. Click "Add".

5. (Optional) Repeat for TestPyPI with the test environment:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `quant_lib` |
   | Owner | `Hansiongs` |
   | Repository name | `hans-backtest` |
   | Workflow name | `publish.yml` |
   | Environment name | `testpypi` |

### Creating a Release

1. Push a tag:
   ```bash
   git tag -a v0.x.x -m "v0.x.x"
   git push origin v0.x.x
   ```

2. On GitHub, create a **Release** from that tag:
   - Go to https://github.com/Hansiongs/hans-backtest/releases
   - Click "Draft a new release"
   - Choose the tag you just pushed
   - Write release notes
   - Publish

3. The `publish.yml` workflow triggers automatically and:
   - Builds sdist + wheel
   - Uploads to `pypi.org/p/quant_lib`
   - No manual steps required

### Testing a Pre-Release (TestPyPI)

1. On the Actions tab, select the `publish` workflow
2. Click "Run workflow"
3. Set environment to `testpypi`
4. Run

This uploads to `test.pypi.org/project/quant_lib` for validation.
Install from TestPyPI with:

```bash
pip install --index-url https://test.pypi.org/simple/ quant_lib
```

## Local Build Verification

Before creating a release, verify the build:

```bash
pip install build wheel
python -m build --sdist --wheel --outdir dist/
# Check the package installs
pip install dist/quant_lib-*.tar.gz
python -c "from quant_lib import run_explore; print('OK')"
```

## Changelog

Every release must update `CHANGELOG.md` under the version header.
Use the same format as existing entries:

```markdown
## [0.x.x] - YYYY-MM-DD

### Added
### Changed
### Fixed
```
