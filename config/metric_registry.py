"""Canonical metric registry for dataset-derivable calculations."""

from typing import Final

SUPPORTED_METRICS: Final[dict[str, dict[str, object]]] = {
    "pnl": {
        "description": "Net profit and loss = revenue_total + expenses_total (expenses are negative).",
        "required_columns": ["ledger_type", "profit"],
    },
    "revenue_total": {
        "description": "Total revenue where ledger_type == 'revenue'.",
        "required_columns": ["ledger_type", "profit"],
    },
    "expenses_total": {
        "description": "Total expenses where ledger_type == 'expenses'.",
        "required_columns": ["ledger_type", "profit"],
    },
    "net_pnl": {
        "description": "Net P&L computed as revenue_total + expenses_total.",
        "required_columns": ["ledger_type", "profit"],
    },
    "count": {
        "description": "Count rows or unique entities by grouping dimensions.",
        "required_columns": [],
    },
    "sum_profit": {
        "description": "Sum of profit across selected scope.",
        "required_columns": ["profit"],
    },
}
