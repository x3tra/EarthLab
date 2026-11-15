from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LayerRule(BaseModel):
    """A requested variable and the terms used to find its actual API layer."""

    model_config = ConfigDict(extra="forbid")

    label: str
    match_any: list[str] = Field(min_length=1)


def validate_layers(
    layer_metadata: dict[str, dict[str, Any]],
    rules: list[LayerRule],
) -> list[dict[str, Any]]:
    """Match requested variable concepts to actual IDs in AppEEARS metadata."""

    results: list[dict[str, Any]] = []
    for rule in rules:
        terms = tuple(term.casefold() for term in rule.match_any)
        exact_matches = [
            {"layer_id": layer_id, **metadata}
            for layer_id, metadata in layer_metadata.items()
            if layer_id.casefold() in terms
        ]
        matches = exact_matches or [
            {"layer_id": layer_id, **metadata}
            for layer_id, metadata in layer_metadata.items()
            if any(
                term in layer_id.casefold()
                or term in str(metadata.get("Description", "")).casefold()
                for term in terms
            )
        ]
        results.append(
            {
                "requested_layer": rule.label,
                "status": "available" if matches else "unavailable",
                "matched_layers": matches,
                "reason": None if matches else "No matching layer identifier or description in the API metadata.",
            }
        )
    return results


def discover_layer_categories(
    layer_metadata: dict[str, dict[str, Any]],
    categories: dict[str, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    """Collect actual product layers relevant to exploratory semantic categories."""

    found: dict[str, list[dict[str, Any]]] = {}
    for category, terms in categories.items():
        normalized_terms = tuple(term.casefold() for term in terms)
        found[category] = [
            {"layer_id": layer_id, **metadata}
            for layer_id, metadata in layer_metadata.items()
            if any(
                term in layer_id.casefold()
                or term in str(metadata.get("Description", "")).casefold()
                for term in normalized_terms
            )
        ]
    return found
