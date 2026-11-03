Create a new release of jax-pme. Optional argument: version bump level (`patch`, `minor`, or `major`).

## Steps

### 1. Pre-flight checks

- Verify you are on the `main` branch with a clean working tree, in sync with `origin/main`
- Check that CI is passing on the latest commit: `gh run list --branch main --limit 3`
- Check that the `release` environment exists: `gh api repos/lab-cosmo/jax-pme/environments/release`
- If anything is off, do NOT proceed — investigate and fix first

### 2. Run full local verification

- Run `ruff format --check . && ruff check .` — must pass
- Run `python -m pytest tests/` — must pass (takes a few minutes)
- Do NOT proceed if either fails

### 3. Determine version

- Find the latest git tag with `git describe --tags --abbrev=0` (or note if there are no tags yet)
- If a bump level was given ($ARGUMENTS), compute the new version following semver (e.g., `0.1.0` → `0.1.1` for patch, `0.2.0` for minor, `1.0.0` for major)
- If NO bump level was given, review the changes (step 4) first, then discuss with the user what the appropriate level should be based on the nature of the changes (breaking → major, new features → minor, fixes/maintenance → patch)
- Confirm the new version with the user before proceeding

### 4. Review changes and finalise the changelog

- Read the `## Unreleased` section of `CHANGELOG.md`, and cross-check it against `git log <last-tag>..HEAD --oneline` for anything user-facing that is missing
- Tidy it into release notes: brief, grouped by category where appropriate (features, fixes, breaking changes, maintenance, etc.), with PR numbers
- Present the changelog to the user for review and approval
- On a `release/v<version>` branch, rename `## Unreleased` to `## v<version> (<YYYY-MM-DD>)` and add a fresh empty `## Unreleased` above it
- Open a PR "Release v<version>", wait for CI, merge it, and update local `main`

### 5. Tag and push

- Tag the merge commit on `main`: `git tag -a v<version> -m "Release v<version>"`
- Push the tag: `git push origin v<version>`
- This triggers the `release.yml` CI workflow which builds and publishes to PyPI
- Watch it: `gh run watch` (or `gh run list --workflow release.yml --limit 1`), and confirm the new version appears at https://pypi.org/project/jax-pme/

### 6. Create GitHub release

- Use `gh release create v<version> --title "v<version>" --notes "<changelog>"` to create a GitHub release with the `CHANGELOG.md` section from step 4

## Notes

- This project uses `setuptools_scm` — the version is derived from git tags, not from pyproject.toml. The only file a release touches is `CHANGELOG.md`.
- The CI release workflow uses PyPI trusted publishing (no tokens needed). The GitHub environment name (`release`) and workflow file name (`release.yml`) must match the trusted publisher configured on PyPI.
- Run this from inside the dev repo (`jax-pme/`), not the workspace root.
