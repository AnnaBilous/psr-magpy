"""Parameter expansion utilities for experiment series.

This module provides utilities for expanding flexible parameter inputs
into all possible combinations for experiment generation.
"""

from typing import Dict, Any, List
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


def get_parameter_combinations(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters that contain lists.

    Parameters
    ----------
    params : Dict[str, Any]
        Dictionary of parameters, some of which may be lists.

    Returns
    -------
    List[Dict[str, Any]]
        List of parameter dictionaries, one for each combination.

    Examples
    --------
    >>> params = {"a": 1, "b": [2, 3], "c": [4, 5]}
    >>> get_parameter_combinations(params)
        [
            {"a": 1, "b": 2, "c": 4},
            {"a": 1, "b": 2, "c": 5},
            {"a": 1, "b": 3, "c": 4},
            {"a": 1, "b": 3, "c": 5}
        ]
    """
    # Find parameters that need expansion
    expandable_params = {}
    fixed_params = {}

    for key, value in params.items():
        if isinstance(value, list):
            expandable_params[key] = value
        else:
            fixed_params[key] = value

    if not expandable_params:
        return [params]

    # Generate combinations
    combinations = [fixed_params]

    for key, value in expandable_params.items():
        new_combinations = []

        # List: create combinations with each value
        for item in value:
            for combo in combinations:
                new_combo = combo.copy()
                new_combo[key] = item
                new_combinations.append(new_combo)

        combinations = new_combinations

    return combinations
