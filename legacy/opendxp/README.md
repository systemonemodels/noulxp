# opendxp is now noulxp

OpenDXP was renamed **NoulXP** in version 0.4.0. Nothing else changed: the
format, the answers and the protocol are the same.

```bash
pip install noulxp
noulxp serve ./my-package
```

This package only installs `noulxp` and keeps the old names working:
`import opendxp` (and every `opendxp.*` module) is `noulxp`, and the `opendxp`
command runs `noulxp`. Packages made with opendxp (`odxp.json`, `odxp/0.x`)
run on noulxp as they are.

Code, specification and changelog: https://github.com/systemonemodels/noulxp
