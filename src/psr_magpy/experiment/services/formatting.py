"""Shared formatting helpers for experiment services."""


def format_duration(seconds: float) -> str:
    """Format duration in appropriate units.

    Returns strings like ``10.5s``, ``2.5min``, ``1.2h``, or ``3.5d``.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}min"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}h"
    days = hours / 24
    return f"{days:.1f}d"
