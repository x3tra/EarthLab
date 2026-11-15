from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .appeears_client import AppEEARSClient
from .layer_validation import LayerRule, discover_layer_categories, validate_layers

logger = logging.getLogger(__name__)


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    metadata_file: str
    enabled: bool = False
    product_id: str | None = None
    requested_layers: list[str] = Field(default_factory=list)
    output_group: str = "OTHER"
    output_name: str | None = None
    output_format: Literal["geotiff", "netcdf4"] = "geotiff"
    requested_product_ids: list[str] = Field(default_factory=list)
    product_prefixes: list[str] = Field(default_factory=list)
    product_contains: list[str] = Field(default_factory=list)
    required_layers: list[LayerRule] = Field(default_factory=list)
    discover_categories: dict[str, list[str]] = Field(default_factory=dict)


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets: list[DatasetSpec] = Field(min_length=1)


class ProductDiscovery:
    """Discover live AppEEARS products, layers, and metadata."""

    def __init__(self, client: AppEEARSClient | None = None) -> None:
        self.client = client or AppEEARSClient()

    @staticmethod
    def load_config(path: str | Path) -> DatasetConfig:
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as file:
            raw_config = yaml.safe_load(file)
        return DatasetConfig.model_validate(raw_config)

    def discover(
        self,
        config: DatasetConfig,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        catalog = self.client.product_catalog()
        by_id = {
            entry["ProductAndVersion"]: entry
            for entry in catalog
            if isinstance(entry.get("ProductAndVersion"), str)
        }

        report: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "catalog_product_count": len(catalog),
            "catalog_endpoint": f"{self.client.base_url}/product",
            "datasets": [],
        }

        for spec in config.datasets:
            report_entry, metadata_file = self._discover_dataset(spec, catalog, by_id)
            self._write_json(output_path / spec.metadata_file, metadata_file)
            report["datasets"].append(report_entry)
            logger.info(
                "%s: %s (%d requested layer checks)",
                spec.name,
                report_entry["status"],
                sum(len(group["checks"]) for group in report_entry["layer_checks"]),
            )

        self._write_json(output_path / "availability_report.json", report)
        (output_path / "availability_report.md").write_text(
            self._format_markdown(report),
            encoding="utf-8",
        )
        return report

    def _discover_dataset(
        self,
        spec: DatasetSpec,
        catalog: list[dict[str, Any]],
        by_id: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        family_matches = [
            entry
            for entry in catalog
            if self._matches_family(entry, spec)
        ]
        exact_entries = [
            by_id[product_id]
            for product_id in spec.requested_product_ids
            if product_id in by_id
        ]

        if spec.requested_product_ids:
            checked_entries = exact_entries
            available_entries = [
                entry for entry in exact_entries if entry.get("Available") is True
            ]
            requested_available = len(available_entries) == len(spec.requested_product_ids)
        else:
            checked_entries = family_matches
            available_entries = [
                entry for entry in family_matches if entry.get("Available") is True
            ]
            requested_available = bool(available_entries)

        # Fetch requested products and potential newer family matches separately.
        metadata_ids = {
            entry["ProductAndVersion"]
            for entry in checked_entries + family_matches
            if entry.get("Available") is True
        }
        product_metadata: dict[str, dict[str, Any]] = {}
        for product_id in sorted(metadata_ids):
            product_metadata[product_id] = self.client.product_metadata(product_id)

        layer_checks: list[dict[str, Any]] = []
        category_matches: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for entry in checked_entries:
            product_id = entry.get("ProductAndVersion")
            layers = product_metadata.get(product_id, {})
            if spec.required_layers:
                layer_checks.append(
                    {
                        "product_id": product_id,
                        "checks": validate_layers(layers, spec.required_layers),
                    }
                )
            if spec.discover_categories:
                category_matches[product_id] = discover_layer_categories(
                    layers,
                    spec.discover_categories,
                )

        matched_ids = {entry.get("ProductAndVersion") for entry in checked_entries}
        alternatives = [
            {
                "product_id": entry.get("ProductAndVersion"),
                "available": entry.get("Available") is True,
                "description": entry.get("Description"),
                "selection_status": "not_substituted",
            }
            for entry in family_matches
            if entry.get("ProductAndVersion") not in matched_ids
        ]

        missing_ids = [
            product_id
            for product_id in spec.requested_product_ids
            if product_id not in by_id
        ]
        status = "available" if requested_available else "unavailable"
        if spec.requested_product_ids and missing_ids:
            reason = "Requested product/version is absent from the live AppEEARS catalog."
        elif spec.requested_product_ids and not requested_available:
            reason = "Requested product/version is listed but not marked available by AppEEARS."
        elif not requested_available:
            reason = "No available product matched the configured AppEEARS product family."
        else:
            reason = None

        layer_checks = self._mark_unevaluated_layers(
            layer_checks,
            spec.required_layers,
            checked_entries,
            product_metadata,
        )
        dataset_metadata = {
            "dataset": spec.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "requested_product_ids": spec.requested_product_ids,
            "reason": reason,
            "missing_requested_product_ids": missing_ids,
            "available_products": [
                {
                    "catalog": entry,
                    "layers": product_metadata.get(entry["ProductAndVersion"], {}),
                }
                for entry in checked_entries
            ],
            "alternative_products_not_substituted": [
                {
                    "catalog": alternative,
                    "layers": product_metadata.get(alternative["product_id"], {}),
                }
                for alternative in alternatives
            ],
            "layer_checks": layer_checks,
            "discovered_layer_categories": category_matches,
        }
        report_entry = {
            "dataset": spec.name,
            "status": status,
            "requested_product_ids": spec.requested_product_ids,
            "available_product_ids": [
                entry.get("ProductAndVersion") for entry in available_entries
            ],
            "alternative_products_not_substituted": alternatives,
            "reason": reason,
            "layer_checks": layer_checks,
            "discovered_layer_categories": category_matches,
        }
        return report_entry, dataset_metadata

    @staticmethod
    def _matches_family(entry: dict[str, Any], spec: DatasetSpec) -> bool:
        catalog_text = " ".join(
            str(entry.get(field, ""))
            for field in ("ProductAndVersion", "Product", "Description")
        ).casefold()
        product_id = str(entry.get("ProductAndVersion", "")).casefold()
        product_name = str(entry.get("Product", "")).casefold()
        prefix_match = any(
            product_id.startswith(prefix.casefold())
            or product_name == prefix.casefold().removesuffix(".")
            or (
                not prefix.endswith(".")
                and product_name.startswith(f"{prefix.casefold()}_")
            )
            for prefix in spec.product_prefixes
        )
        return prefix_match or any(
            term.casefold() in catalog_text
            for term in spec.product_contains
        )

    @staticmethod
    def _mark_unevaluated_layers(
        checks: list[dict[str, Any]],
        rules: list[LayerRule],
        checked_entries: list[dict[str, Any]],
        metadata: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        checked_ids = {item.get("product_id") for item in checks}
        if not rules:
            return checks
        if not checked_entries:
            return [
                {
                    "product_id": None,
                    "checks": [
                        {
                            "requested_layer": rule.label,
                            "status": "not_evaluated",
                            "matched_layers": [],
                            "reason": "No requested product is available in the catalog.",
                        }
                        for rule in rules
                    ],
                }
            ]
        for entry in checked_entries:
            product_id = entry.get("ProductAndVersion")
            if product_id in checked_ids:
                continue
            if entry.get("Available") is not True:
                checks.append(
                    {
                        "product_id": product_id,
                        "checks": [
                            {
                                "requested_layer": rule.label,
                                "status": "not_evaluated",
                                "matched_layers": [],
                                "reason": "AppEEARS marks this product unavailable.",
                            }
                            for rule in rules
                        ],
                    }
                )
                continue
            checks.append(
                {
                    "product_id": product_id,
                    "checks": validate_layers(metadata.get(product_id, {}), rules),
                }
            )
        return checks

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.write("\n")
        temporary_path.replace(path)

    @staticmethod
    def _format_markdown(report: dict[str, Any]) -> str:
        lines = [
            "# AppEEARS Product Availability",
            "",
            f"- Generated: {report['generated_at']}",
            f"- Catalog products retrieved: {report['catalog_product_count']}",
            f"- Catalog endpoint: `{report['catalog_endpoint']}`",
            "",
            "| Dataset | Status | Requested product IDs | Available product IDs |",
            "|---|---|---|---|",
        ]
        for dataset in report["datasets"]:
            requested = ", ".join(dataset["requested_product_ids"]) or "family discovery"
            available = ", ".join(
                product_id for product_id in dataset["available_product_ids"] if product_id
            ) or "—"
            lines.append(
                f"| {dataset['dataset']} | {dataset['status']} | {requested} | {available} |"
            )
        lines.extend(["", "## Dataset details", ""])
        for dataset in report["datasets"]:
            lines.append(f"### {dataset['dataset']} — {dataset['status']}")
            if dataset["reason"]:
                lines.append(f"\n{dataset['reason']}\n")
            for product_check in dataset["layer_checks"]:
                lines.append(
                    f"\n#### {product_check['product_id'] or 'Requested product layers'}"
                )
                lines.extend(
                    f"- {check['requested_layer']}: **{check['status']}**"
                    + (
                        f" — `{', '.join(layer['layer_id'] for layer in check['matched_layers'])}`"
                        if check["matched_layers"]
                        else f" — {check['reason']}"
                    )
                    for check in product_check["checks"]
                )
            for product_id, categories in dataset["discovered_layer_categories"].items():
                lines.append(f"\n#### Discovered layers — `{product_id}`")
                lines.extend(
                    f"- {category}: "
                    + (
                        ", ".join(f"`{layer['layer_id']}`" for layer in layers)
                        if layers
                        else "no matching layers found"
                    )
                    for category, layers in categories.items()
                )
            if dataset["alternative_products_not_substituted"]:
                lines.append("\nAlternative catalog versions found (not substituted):")
                lines.extend(
                    f"- `{item['product_id']}` — available: {item['available']}"
                    for item in dataset["alternative_products_not_substituted"]
                )
        lines.append("")
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover live AppEEARS products and layer metadata.")
    parser.add_argument("--config", type=Path, default=Path("config/datasets.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/metadata"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    discovery = ProductDiscovery()
    report = discovery.discover(discovery.load_config(args.config), args.output_dir)
    logger.info(
        "Saved AppEEARS availability report for %d dataset entries to %s",
        len(report["datasets"]),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
