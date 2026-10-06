# Releasing

Releases go to PyPI from GitHub Actions with Trusted Publishing: PyPI verifies
the workflow's identity directly, so no token is stored anywhere. Each file is
also signed with that identity, and PyPI shows the provenance on the release.

## One-time setup

1. On [pypi.org](https://pypi.org): **Your account → Publishing → Add a new
   pending publisher** (GitHub):

   | Field | Value |
   | --- | --- |
   | PyPI project name | `noulxp` |
   | Owner | `systemonemodels` |
   | Repository name | `noulxp` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   "Pending" because the project does not exist on PyPI until the first
   release creates it.

2. On GitHub: **Settings → Environments → New environment → `pypi`**. Under
   *Deployment protection rules*, tick **Required reviewers** and add yourself.
   Under *Deployment branches and tags*, allow only tags matching `v*`. Every
   release then waits for a click, because a version published to PyPI can
   never be replaced.

## The rename (once, for 0.4.0)

OpenDXP became NoulXP in 0.4.0, and `opendxp` on PyPI became a package that
installs `noulxp` ([legacy/opendxp](legacy/opendxp)). Before tagging v0.4.0:

1. GitHub: **Settings → General → Repository name**: `opendxp` → `noulxp`.
   GitHub redirects the old URLs, clones and remotes included.
2. PyPI: the pending publisher for `noulxp` above.
3. PyPI: project **opendxp → Settings → Publishing**: point its GitHub
   publisher at the repository `noulxp` (workflow `release.yml`, environment
   `pypi`), so the same run can publish both.

The tag v0.4.0 then publishes `noulxp` 0.4.0 and `opendxp` 0.4.0 together,
after one approval: the workflow builds the redirect only when its version is
the tag's, and publishes it after noulxp, in a step of its own. Without step 3,
noulxp is published all the same and that step fails: fix the publisher, then
push the tag `v0.4.0-opendxp`, which builds and publishes the redirect alone.

## Each release

```bash
# 1. bump the version in both places, and add a CHANGELOG entry
$EDITOR pyproject.toml src/noulxp/__init__.py CHANGELOG.md
git commit -am "Release 0.1.1"

# 2. tag and push
git tag -a v0.1.1 -m "noulxp 0.1.1"
git push origin main v0.1.1
```

The workflow checks the tag matches the package version, runs the tests on the
oldest and newest supported Python, builds, installs the wheel into a clean
environment to prove it runs, and then waits for approval in the Actions tab.
A minute after approval, `pip install noulxp` gets it.

The build runs in a job of its own, with nothing but uv and the build backend,
and records a digest of what it made. The tests and checks run their
third-party code in other jobs, and the publish job, the only one PyPI trusts,
uploads only files with that digest.

A new version of the standard itself (`noulxp/0.2`) is a change to SPEC.md, the
schemas and the reference runtime together; see CONTRIBUTING.md.

## Pinned tools

Nothing in the workflows moves on its own, so a hijacked release of a tool
cannot reach a published file:

- **Actions** are pinned by commit, with the version in a comment. Dependabot
  proposes new versions once a month, in one pull request.
- **uv** is `PINNED_UV` in each workflow, checked against `PINNED_UV_SHA256`:
  the sha256 of `uv-x86_64-unknown-linux-gnu.tar.gz` on that uv release.
- **The build backend** (hatchling and what it needs) is pinned with hashes in
  [.github/build-constraints.txt](.github/build-constraints.txt). To move it:

  ```bash
  echo hatchling | uv pip compile - --universal --python-version 3.11 \
    --generate-hashes -o .github/build-constraints.txt
  ```

- **twine** is pinned where it runs.

## Try a build locally

```bash
uv build -b .github/build-constraints.txt --require-hashes
uvx twine==7.0.0 check --strict dist/*
```
