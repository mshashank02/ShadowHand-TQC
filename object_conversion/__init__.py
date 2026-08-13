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
from .convex_decomposition import (
    CoACDParameters,
    ConvexPiece,
    DecompositionResult,
    decompose_surface_cached,
    decomposition_cache_key,
    load_decomposition_pieces,
    load_convex_piece_obj,
    validate_convex_piece,
)

__all__ = [
    "CONVERTER_VERSION",
    "ConversionResult",
    "GmshFormatError",
    "convert_gmsh_to_rigid_surface",
    "extract_exterior_surface",
    "parse_gmsh_v2",
    "scaled_geometry_metrics",
    "CoACDParameters",
    "ConvexPiece",
    "DecompositionResult",
    "decompose_surface_cached",
    "decomposition_cache_key",
    "load_decomposition_pieces",
    "load_convex_piece_obj",
    "validate_convex_piece",
]
