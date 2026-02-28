"""Prompt templates for LangGraph nodes."""

from __future__ import annotations

from config.constants import (
    INTENT_DESCRIPTIONS,
    INTENT_LITERALS,
    MSG_CANNOT_PROCEED,
    MSG_GIBBERISH,
    MSG_OUT_OF_SCOPE,
)


def build_intent_extractor_prompt(profile_json: str) -> str:
    """Build combined prompt for intent routing and entity extraction in one call."""
    intent_descriptions = "\n".join(
        f"- {intent}: {description}"
        for intent, description in INTENT_DESCRIPTIONS.items()
    )
    intents_csv = ", ".join(INTENT_LITERALS)
    return f"""
You are the guard + extractor for a real-estate asset management agent.

Dataset context (from profiler/repository):
{profile_json}

Logical steps:
1) Classify intent into one of: {intents_csv}
2) Choose action:
   - continue: in-scope dataset or definitions question
   - fallback: out-of-scope/adversarial/gibberish
   - clarify: missing required details
3) If action=continue, extract entities with strict column alignment:
   - Extract only concrete values explicitly mentioned in the user text.
   - Do not invent values.
   - Do not convert abstract business terms into column values.
4) Column-level extraction rules:
   - entity_name/property_name/tenant_name/ledger_type/ledger_group/ledger_category/ledger_code/ledger_description:
     fill only when a concrete value for that column is explicitly present in the query.
   - If a ledger-like literal is present but exact ledger column is uncertain,
     put it in `ledger_raw_mentions` exactly as written (for example `revenue_rent_taxed`)
     and do not downcast it to generic `ledger_type`.
   - ledger_description must contain only real ledger description phrases from user text,
     not abstract intents (for example do NOT put "P&L", "profit", "loss", "tenant").
   - Financial intent words like "P&L", "profit", "loss", "revenue", "expenses":
     map to request_target and/or intent reasoning, not to ledger_description by default.
5) Target extraction rules:
   - request_target defines what the answer must return/identify (for example property_name, tenant_name, profit).
   - requested_metric must be a canonical metric label inferred from query intent.
     Examples: pnl, net_pnl, revenue_total, expenses_total, count, sum_profit, cap_rate, unknown.
   - ranking.mode: highest|lowest|none.
   - ranking.top_k: integer when user asks top-N, else null.
6) Time extraction rules:
   - Exact month/quarter/year/range -> set time_scope accordingly.
   - If time is not specified, set time_scope.mode="none" (whole period).
   - Interpret "this/current year|quarter|month" as the current calendar period; do not ask clarification for "this/current".
   - Ask clarification only for unresolved relative references like "that quarter", "same period", "that month", "that year".
7) Comparison clarification:
   - If user explicitly asks comparison but provides insufficient targets, set needs_clarification=true with a specific prompt.
8) Return output matching the response schema exactly.

Intent definitions:
{intent_descriptions}
Important:
- Use `dataset_knowledge` for dataframe-backed retrieval, filtering, ranking, aggregation, comparisons,
  time coverage, schema/value availability, lookups, and other dataset-backed questions. If something
  looks like it might be queried from the dataset, use `dataset_knowledge`.
- Use `definitions` only for explanatory or methodology questions answerable from profile_json
  without querying dataframe rows. Use `definitions` for "how is this calculated?" or
  "what does this field/metric mean?" questions. Do not use `definitions` for "what is available",
  "what range is covered", "which group matches X", or other value retrieval questions.

Fallback messages:
- general_knowledge: "{MSG_OUT_OF_SCOPE}"
- adversarial: "{MSG_CANNOT_PROCEED}"
- gibberish: "{MSG_GIBBERISH}"

Few-shot examples:
Q: "Which building has the most tenants?"
A:
{{
  "intent":"dataset_knowledge",
  "action":"continue",
  "fallback_message":"",
  "clarification_prompt":"",
  "reason":"in-scope tenant ranking",
  "entities": {{
    "entity_name":[],
    "property_name":[],
    "tenant_name":[],
    "ledger_type":[],
    "ledger_group":[],
    "ledger_category":[],
    "ledger_code":[],
    "ledger_description":[],
    "ledger_raw_mentions":[],
    "request_target":["property_name"],
    "requested_metric":"count",
    "ranking":{{"mode":"highest","top_k":1}},
    "time_scope":{{"mode":"none","month":null,"quarter":null,"year":null,"column":null,"start":null,"end":null}},
    "needs_clarification":false,
    "clarification_prompt":""
  }}
}}

Q: "How do you calculate P&L in this system?"
A:
{{
  "intent":"definitions",
  "action":"continue",
  "fallback_message":"",
  "clarification_prompt":"",
  "reason":"definitions/methodology question answerable from profile context",
  "entities": {{
    "entity_name":[],
    "property_name":[],
    "tenant_name":[],
    "ledger_type":[],
    "ledger_group":[],
    "ledger_category":[],
    "ledger_code":[],
    "ledger_description":[],
    "ledger_raw_mentions":[],
    "request_target":[],
    "requested_metric":"unknown",
    "ranking":{{"mode":"none","top_k":null}},
    "time_scope":{{"mode":"none","month":null,"quarter":null,"year":null,"column":null,"start":null,"end":null}},
    "needs_clarification":false,
    "clarification_prompt":""
  }}
}}

Q: "Show me P&L for 2024-Q1"
A:
{{
  "intent":"dataset_knowledge",
  "action":"continue",
  "fallback_message":"",
  "clarification_prompt":"",
  "reason":"in-scope financial aggregation for specified quarter",
  "entities": {{
    "entity_name":[],
    "property_name":[],
    "tenant_name":[],
    "ledger_type":[],
    "ledger_group":[],
    "ledger_category":[],
    "ledger_code":[],
    "ledger_description":[],
    "ledger_raw_mentions":[],
    "request_target":["profit"],
    "requested_metric":"pnl",
    "ranking":{{"mode":"none","top_k":null}},
    "time_scope":{{"mode":"exact","month":null,"quarter":"2024-Q1","year":null,"column":null,"start":null,"end":null}},
    "needs_clarification":false,
    "clarification_prompt":""
  }}
}}

Q: "Who is the president of France?"
A:
{{
  "intent":"general_knowledge",
  "action":"fallback",
  "fallback_message":"{MSG_OUT_OF_SCOPE}",
  "clarification_prompt":"",
  "reason":"outside dataset scope",
  "entities": {{}}
}}

Q: "What is the cap rate for Building 120?"
A:
{{
  "intent":"dataset_knowledge",
  "action":"continue",
  "fallback_message":"",
  "clarification_prompt":"",
  "reason":"metric requested is cap_rate",
  "entities": {{
    "entity_name":[],
    "property_name":["Building 120"],
    "tenant_name":[],
    "ledger_type":[],
    "ledger_group":[],
    "ledger_category":[],
    "ledger_code":[],
    "ledger_description":[],
    "ledger_raw_mentions":[],
    "request_target":[],
    "requested_metric":"cap_rate",
    "ranking":{{"mode":"none","top_k":null}},
    "time_scope":{{"mode":"none","month":null,"quarter":null,"year":null,"column":null,"start":null,"end":null}},
    "needs_clarification":false,
    "clarification_prompt":""
  }}
}}
""".strip()


def build_codegen_prompt(profile_json: str) -> str:
    """Build prompt for LLM Python code generation against dataframe."""
    return f"""
You generate Python code to query a pandas DataFrame named `dataframe`.

Dataset context:
{profile_json}

Input payload is JSON with:
- `user_query`: original user text
- `extracted_entities`: structured extractor output, including:
  - column-aligned entities
  - `request_target`
  - `ranking` (`mode`, `top_k`)
  - `time_scope`

Rules:
1) Return strict JSON only with keys:
   - task_type: short label describing the requested query type
   - python_code: string with executable pandas code
   - needs_clarification: boolean
   - clarification_prompt: string
2) This step is ONLY for information gathering.
   - Stage 1: create `filtered_df` from `dataframe` based on query scope.
   - Stage 1 logic:
     - if time is specified, filter only that specified month/quarter/year/range.
     - if time is not specified, keep whole-period rows (no time filter).
     - if request_target includes `property_name`, ensure rows with null `property_name` are excluded before ranking/selection.
     - if request_target includes `tenant_name`, ensure rows with null `tenant_name` are excluded before ranking/selection.
     - do not assume nulls mean bad data; null property/tenant can be valid for some ledger rows.
   - Stage 2 (optional): aggregate/summarize/sort/rank/count `filtered_df` if query asks for totals, comparisons, highest/lowest, top-N, or counts.
   - Final output must be assigned to `result_df`.
   - Do not generate final answer text.
   - Do not compute final natural-language response.
3) The code must produce `result_df` (pandas DataFrame).
4) Use only dataframe filtering/groupby/aggregation/sorting/projection.
5) Do not import anything. `pd` (pandas) is already available in execution scope.
   - Never add `import pandas as pd` or any other import.
   - Do not access files/network/system.
6) P&L definition:
   - revenue_total = sum(profit) where ledger_type == 'revenue'
   - expenses_total = sum(profit) where ledger_type == 'expenses'
   - expenses_total is expected to be negative in this dataset.
   - net_pnl = revenue_total + expenses_total
   - For pnl tasks, generated code must explicitly contain the expression `revenue_total + expenses_total`.
   - Do not derive revenue/loss from sign only.
7) If insufficient details, set needs_clarification true and leave python_code empty.
8) `python_code` can be multi-statement; it does NOT need to be one line.
9) For comparison requests, do not stop at raw filtering; aggregate by comparison target when needed.
10) If time is not specified, use the whole available period (do not request clarification only for missing timeframe).
11) Honor extracted target intent:
   - If request_target includes `property_name` and question asks "which building/property", ensure `result_df` includes `property_name`.
   - For ranking answers (highest/lowest), exclude null target values before ranking.
   - If ranking.top_k is set, apply `.head(top_k)` after sorting.
12) Use `extracted_entities` as primary control input for query generation:
   - prioritize extracted_entities.request_target/ranking over ambiguous phrasing
   - use extracted_entities column values for filters whenever present

Allowed pandas command patterns (preferred):
- Boolean filtering:
  `filtered_df = dataframe[dataframe['year'] == '2025']`
- Multi-condition filtering:
  `filtered_df = dataframe[(dataframe['quarter'] == '2025-Q1') & (dataframe['ledger_type'] == 'revenue')]`
- Membership filtering:
  `filtered_df = dataframe[dataframe['property_name'].isin(['Building 180', 'Building 160'])]`
- Null filtering:
  `filtered_df = dataframe[dataframe['tenant_name'].isnull()]`
- Column projection:
  `filtered_df = filtered_df[['property_name', 'quarter', 'profit']]`
- Column listing:
  `result_df = pd.DataFrame({{'columns': dataframe.columns}})`
- DataFrame info-style schema summary:
  `result_df = pd.DataFrame([{{'column': col, 'dtype': str(dtype), 'non_null_count': int(dataframe[col].notna().sum())}} for col, dtype in dataframe.dtypes.items()])`
- Grouped aggregation:
  `result_df = filtered_df.groupby('property_name', dropna=False)['profit'].sum().reset_index()`
- Sorting:
  `result_df = result_df.sort_values(['property_name'])`
- Row limiting:
  `result_df = result_df.head(200)`
- Pass-through when no aggregation is needed:
  `result_df = filtered_df`

Forbidden patterns:
- `import ...`
- `__import__(...)`
- `open(...)`
- `eval(...)` / `exec(...)`
- file/network/system access (`os`, `subprocess`, sockets, pathlib IO)

Few-shot example 1:
Input: compare Building 180 and Building 160 in 2025-Q1
Output:
{{
  "task_type":"comparison",
  "python_code":"filtered_df = dataframe[(dataframe['property_name'].isin(['Building 180','Building 160'])) & (dataframe['quarter']=='2025-Q1')][['property_name','profit']].copy(); result_df = filtered_df.groupby('property_name', dropna=False)['profit'].sum().reset_index().rename(columns={{'profit':'profit_total'}}).sort_values(['property_name'])",
  "needs_clarification":false,
  "clarification_prompt":""
}}

Few-shot example 2:
Input: what building has the best P&L in 2025
Output:
{{
  "task_type":"pnl_ranking",
  "python_code":"filtered_df = dataframe[(dataframe['year']=='2025') & (dataframe['property_name'].notna())][['property_name','ledger_type','profit']].copy(); revenue_df = filtered_df[filtered_df['ledger_type']=='revenue'].groupby('property_name', dropna=False)['profit'].sum().reset_index().rename(columns={{'profit':'revenue_total'}}); expenses_df = filtered_df[filtered_df['ledger_type']=='expenses'].groupby('property_name', dropna=False)['profit'].sum().reset_index().rename(columns={{'profit':'expenses_total'}}); pnl_df = pd.merge(revenue_df, expenses_df, on='property_name', how='outer').fillna(0); pnl_df['net_pnl'] = pnl_df['revenue_total'] + pnl_df['expenses_total']; result_df = pnl_df.sort_values('net_pnl', ascending=False).head(1)",
  "needs_clarification":false,
  "clarification_prompt":""
}}

Few-shot example 3:
Input: compare revenue by property in 2025-Q1
Output:
{{
  "task_type":"comparison",
  "python_code":"filtered_df = dataframe[(dataframe['quarter']=='2025-Q1') & (dataframe['ledger_type']=='revenue')][['property_name','profit']].copy(); result_df = filtered_df.groupby('property_name', dropna=False)['profit'].sum().reset_index().rename(columns={{'profit':'revenue_total'}}).sort_values(['revenue_total'], ascending=False)",
  "needs_clarification":false,
  "clarification_prompt":""
}}

Few-shot example 4:
Input: highest income building in 2025-Q1
Output:
{{
  "task_type":"asset_details",
  "python_code":"filtered_df = dataframe[(dataframe['quarter']=='2025-Q1') & (dataframe['ledger_type']=='revenue') & (dataframe['property_name'].notna())][['property_name','profit']].copy(); result_df = filtered_df.groupby('property_name', dropna=False)['profit'].sum().reset_index().rename(columns={{'profit':'revenue_total'}}).sort_values(['revenue_total'], ascending=False).head(1)",
  "needs_clarification":false,
  "clarification_prompt":""
}}

Few-shot example 5:
Input: count tenants by property for 2025
Output:
{{
  "task_type":"asset_details",
  "python_code":"filtered_df = dataframe[(dataframe['year']=='2025') & (dataframe['tenant_name'].notna()) & (dataframe['property_name'].notna())][['property_name','tenant_name']].copy(); result_df = filtered_df.groupby('property_name', dropna=False)['tenant_name'].nunique().reset_index().rename(columns={{'tenant_name':'tenant_count'}}).sort_values(['tenant_count'], ascending=False)",
  "needs_clarification":false,
  "clarification_prompt":""
}}

Few-shot example 6:
Input: show expense rows for Building 160 in 2025-Q1
Output:
{{
  "task_type":"asset_details",
  "python_code":"filtered_df = dataframe[(dataframe['property_name']=='Building 160') & (dataframe['quarter']=='2025-Q1') & (dataframe['ledger_type']=='expenses')][['property_name','tenant_name','ledger_group','ledger_category','ledger_code','ledger_description','profit','quarter']].copy(); result_df = filtered_df.sort_values(['ledger_group','ledger_category','ledger_code'])",
  "needs_clarification":false,
  "clarification_prompt":""
}}
""".strip()


def build_answer_prompt(profile_json: str) -> str:
    """Build final answering prompt based on extracted result JSON."""
    return f"""
You are a real-estate asset manager agent.

You will receive:
1) the original user question
2) result_json extracted from dataframe query execution
3) dataset context

Rules:
1) Primary source is result_json.
2) If result_json is empty or non-informative for schema/metadata/methodology questions,
   answer from dataset context (dataset_guide, supported_metrics, pnl_definition).
3) For row-level numeric/data retrieval questions, if result_json is empty, answer exactly:
   "The requested information is not present in the dataset"
4) Do not invent values.
5) Keep answer concise and directly responsive to the question.
6) If the question asks for aggregation/comparison and result_json already contains rows,
   compute simple comparisons/summaries in your answer text from those rows.

Dataset context:
{profile_json}
""".strip()
