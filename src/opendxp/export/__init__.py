"""Converters: from a model's own checkpoint to an OpenDXP package.

Each converter reads the checkpoint with the model's own code or configs,
writes the declarative files (template.json or prompt.json, calibration.json)
and odxp.json, and puts the weights in the portable format. Conformance is a
separate step (`opendxp conformance generate`), run with the model's own code.
"""
