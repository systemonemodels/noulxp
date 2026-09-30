"""Converters: from a model's own checkpoint to a NoulXP package.

Each converter reads the checkpoint with the model's own code or configs,
writes the declarative files (template.json or prompt.json, calibration.json)
and noulxp.json, and puts the weights in the portable format. Conformance is a
separate step (`noulxp conformance generate`), run with the model's own code.
"""
