"""Single source of truth for the variable / pressure-level channel layout.

The model treats pressure levels as part of the channel axis, so the *order* of
features here must match the order of the `feature` coordinate in the zarr and
the order of the normalization arrays (`norm_mean.npy` / `norm_std.npy`).

Channel order:
    [ surface variables ][ pressure_var_0 x all levels ][ pressure_var_1 x ... ]

The number of pressure levels is NOT hardcoded: it is driven by config
(`data.pressure_levels`). This module provides the helpers that every consumer
(expert masks, evaluation metric indices, normalization, channel counting)
derives from, so switching between 13 and 37 levels is a config change only.

This module is intentionally dependency-free (stdlib only) so it can be imported
both from the main training code and from the standalone normalization scripts.
"""

# Surface variables, in channel order.
SURFACE_VARIABLES = [
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "sea_surface_temperature",
    "total_precipitation",
]

# Upper-air variables, in channel order. Each is repeated over every level.
PRESSURE_VARIABLES = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

# Standard ERA5 pressure-level subsets (hPa), descending: surface -> top.
PRESSURE_LEVELS_13 = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# The standard ERA5 37 pressure levels (hPa), descending: surface -> top.
PRESSURE_LEVELS_37 = [
    1000,
    975,
    950,
    925,
    900,
    875,
    850,
    825,
    800,
    775,
    750,
    700,
    650,
    600,
    550,
    500,
    450,
    400,
    350,
    300,
    250,
    225,
    200,
    175,
    150,
    125,
    100,
    70,
    50,
    30,
    20,
    10,
    7,
    5,
    3,
    2,
    1,
]

# Expert name -> the set of *base* variable names that expert "owns".
# Pressure base names match the variable at every level.
EXPERT_VARIABLE_GROUPS = {
    "surface": list(SURFACE_VARIABLES),
    "wind": ["u_component_of_wind", "v_component_of_wind", "vertical_velocity"],
    "thermodynamic": ["geopotential", "temperature", "specific_humidity"],
}

# expert_group.rank() -> expert name. Any rank not listed = "universal" (all vars).
EXPERT_RANK_TO_NAME = {1: "surface", 2: "wind", 3: "thermodynamic"}

_SURFACE_SET = set(SURFACE_VARIABLES)


def _levels_as_str(levels):
    """Normalize a level list to strings for name building.

    Parameters
    ----------
    levels : list of int or str
        Pressure levels (hPa).

    Returns
    -------
    list of str
        The same levels as strings.
    """
    return [str(level) for level in levels]


def build_ordered_variables(
    levels, surface=SURFACE_VARIABLES, pressure=PRESSURE_VARIABLES
):
    """Return the full list of feature names in channel order for ``levels``.

    Parameters
    ----------
    levels : list of int or str
        Pressure levels (hPa), in the order they appear in the data.
    surface : list of str, optional
        Surface variable names (defaults to :data:`SURFACE_VARIABLES`).
    pressure : list of str, optional
        Upper-air variable names (defaults to :data:`PRESSURE_VARIABLES`); each
        is repeated over every level.

    Returns
    -------
    list of str
        Feature names ordered as ``[surface..., var_level...]``.
    """
    str_levels = _levels_as_str(levels)
    ordered = list(surface)
    for var in pressure:
        for level in str_levels:
            ordered.append(f"{var}_{level}")
    return ordered


def num_variables(levels, surface=SURFACE_VARIABLES, pressure=PRESSURE_VARIABLES):
    """Return the total channel count for a level set.

    Parameters
    ----------
    levels : list of int or str
        Pressure levels (hPa).
    surface : list of str, optional
        Surface variable names (defaults to :data:`SURFACE_VARIABLES`).
    pressure : list of str, optional
        Upper-air variable names (defaults to :data:`PRESSURE_VARIABLES`).

    Returns
    -------
    int
        ``len(surface) + len(pressure) * len(levels)``.
    """
    return len(surface) + len(pressure) * len(levels)


def variable_base_name(feature_name):
    """Strip the ``_<level>`` suffix from a pressure feature name.

    Surface names are returned unchanged.

    Parameters
    ----------
    feature_name : str
        A feature name, e.g. ``"geopotential_1000"`` or
        ``"10m_u_component_of_wind"``.

    Returns
    -------
    str
        The base variable name, e.g. ``"geopotential"`` or
        ``"10m_u_component_of_wind"``.
    """
    if feature_name in _SURFACE_SET:
        return feature_name
    return feature_name.rsplit("_", 1)[0]


def feature_index_map(levels):
    """Map each feature name to its global channel index.

    Parameters
    ----------
    levels : list of int or str
        Pressure levels (hPa).

    Returns
    -------
    dict
        Mapping of feature name to its 0-based channel index.
    """
    return {name: i for i, name in enumerate(build_ordered_variables(levels))}


def variable_group_mask(base_names, levels):
    """Build a boolean mask marking features whose base name is in a group.

    Used to construct expert loss-weight masks for any level count.

    Parameters
    ----------
    base_names : iterable of str
        Base variable names that define the group.
    levels : list of int or str
        Pressure levels (hPa).

    Returns
    -------
    list of bool
        Mask of length ``num_variables(levels)``; ``True`` where the feature's
        base name is in ``base_names``.
    """
    base_set = set(base_names)
    ordered = build_ordered_variables(levels)
    return [variable_base_name(name) in base_set for name in ordered]


def expert_variable_mask(expert_rank, levels):
    """Build the global boolean mask of variables owned by an expert.

    Parameters
    ----------
    expert_rank : int
        Rank within the Expert process group. Ranks in
        :data:`EXPERT_RANK_TO_NAME` map to a specific variable group; any other
        rank is the universal expert (owns all variables).
    levels : list of int or str
        Pressure levels (hPa).

    Returns
    -------
    list of bool
        Mask of length ``num_variables(levels)`` marking the owned variables.
    """
    name = EXPERT_RANK_TO_NAME.get(expert_rank)
    if name is None:
        return [True] * num_variables(levels)
    return variable_group_mask(EXPERT_VARIABLE_GROUPS[name], levels)
