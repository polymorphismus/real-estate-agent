"""Matching-rule constants for missing value checks."""

from typing import Final

MISSING_CHECK_COLUMNS: Final[tuple[str, ...]] = (
    "entity_name",
    "property_name",
    "tenant_name",
    "ledger_code",
    "ledger_type",
    "ledger_group",
    "ledger_category",
    "ledger_description",
)
