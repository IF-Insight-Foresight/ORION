"""Aggregate application callbacks."""

# Import modules that define callbacks so that Dash registers them when this
# package is imported by ``run.py``.
from .. import auth   # noqa: F401
from .. import pages  # noqa: F401
from . import info     # noqa: F401

