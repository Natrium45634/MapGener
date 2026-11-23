"""v5.4 atmosphere: monthly circulation normals, moisture transport, ITCZ, monsoons,
storm climatology and the optional synthetic weather year (TZ 5.4)."""

from __future__ import annotations

from dataclasses import replace


def settings(config):
    """The config's ``AtmosphereParams`` with the domain mode resolved (or None)."""
    atm = getattr(config, "atmosphere", None)
    if atm is None or not atm.enabled:
        return None
    from .stage import resolve_domain

    return replace(atm, domain_mode=resolve_domain(atm, config.mode))
