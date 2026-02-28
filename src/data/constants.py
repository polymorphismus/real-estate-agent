"""Data-layer constants for schema and filtering."""

from typing import Final

EXPECTED_COLUMNS: Final[list[str]] = [
    "entity_name",
    "property_name",
    "tenant_name",
    "ledger_type",
    "ledger_group",
    "ledger_category",
    "ledger_code",
    "ledger_description",
    "month",
    "quarter",
    "year",
    "profit",
]

OPTIONAL_NULLABLE_COLUMNS: Final[set[str]] = {"property_name", "tenant_name"}

PROFILE_VALUE_COLUMNS = [
    "entity_name",
    "property_name",
    "tenant_name",
    "ledger_type",
    "ledger_group",
    "ledger_category",
    "ledger_code",
    "ledger_description",
    "month",
    "quarter",
    "year",
]
