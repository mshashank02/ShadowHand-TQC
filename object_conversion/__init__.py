"""Deterministic study-object conversion utilities."""

from .gmsh_to_rigid_surface import (
    CONVERTER_VERSION,
    ConversionResult,
    GmshFormatError,
    convert_gmsh_to_rigid_surface,
    extract_exterior_surface,
    parse_gmsh_v2,
    scaled_geometry_metrics,
)

__all__ = [
    "CONVERTER_VERSION",
    "ConversionResult",
    "GmshFormatError",
    "convert_gmsh_to_rigid_surface",
    "extract_exterior_surface",
    "parse_gmsh_v2",
    "scaled_geometry_metrics",
]
