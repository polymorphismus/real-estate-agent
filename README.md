# Real Estate Asset Manager Agent

## Overview
This project is a prototype multi-agent real estate asset management assistant built with LangGraph.

It accepts natural-language questions about a real-estate portfolio dataset to provide grounded answers on asset-management topics, such as calculating profit and loss (P&L), ranking properties, or retrieving asset details.

The system is delivered with a simple Streamlit chat UI for interactive use.

The implementation uses a local dataset stored at `data/cortex.parquet`.

## What The Agent Can Do
Examples of supported questions:
- "Show me P&L for 2024-Q1"
- "Which building has the most tenants?"
- "What ledger group includes insurance-related entries?"
- "Which buildings have revenue_rent_taxed in 2025?"
- "What is the profit of Building 140 for tenant 2 in 2025?"

## Technology Choices
- Python
- LangGraph for orchestration
- OpenAI API for routing, structured extraction, code generation, and final answer phrasing
- Pandas for dataset querying and aggregation
- Streamlit for the user interface

## Setup
Run all commands from the project root directory.

### 1. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set environment variables
Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Optional settings can also be provided:
- `OPENAI_TEMPERATURE`
- `OPENAI_MAX_OUTPUT_TOKENS`
- `OPENAI_MAX_OUTPUT_TOKENS_EXTRACTOR`
- `OPENAI_MAX_OUTPUT_TOKENS_CODEGEN`
- `OPENAI_MAX_OUTPUT_TOKENS_ANSWER`
- `OPENAI_TIMEOUT_SEC`

### 4. Run the app
```bash
streamlit run app/ui_streamlit.py
```

## User Interface
The Streamlit app is intentionally minimal:
- chat-style interaction
- persistent conversation state in session
- rotating wait messages while the graph runs

Main UI file:
- [ui_streamlit.py](app/ui_streamlit.py)

## Dataset
The agent uses a single local dataset:
- `data/cortex.parquet`

The assignment examples mention property addresses, but the provided dataset snippet does not contain a property address field. The prototype therefore uses `property_name` as the available property identifier and cannot return literal street-address details.

At startup, the system builds a compact dataset profile that is reused across requests. This profile contains:
- column names
- selected unique values
- null counts
- supported metric definitions
- a dataset guide used by prompts

This profile helps the LLM route requests more accurately and generate safer, better-grounded query plans.

## Solution Architecture
The application is structured as a LangGraph workflow with specialized nodes.

### Graph Flow
1. `guard_router_agent`
2. `entity_extractor_agent`
3. `query_agent`
4. `executor_response_agent`
5. `finalize_node`

If clarification is needed, the flow branches through `clarification_node` before finalizing.

Graph wiring:
- [flow.py](src/graph/flow.py)

### Multi-Agent Workflow
#### 1. Guard / Intent Routing
The first stage classifies the request and decides whether to:
- continue
- clarify
- fallback

It also performs early blocking for:
- multi-question inputs
- adversarial requests
- gibberish
- clearly out-of-scope questions

Intent categories currently used:
- `dataset_knowledge`
- `definitions`
- `general_knowledge`
- `ambiguous`
- `adversarial`
- `gibberish`

#### 2. Entity Extraction
If the query is in scope, the system extracts structured entities such as:
- `property_name`
- `tenant_name`
- `ledger_type`
- `ledger_group`
- `ledger_category`
- `ledger_code`
- `ledger_description`
- `time_scope`
- `request_target`
- `requested_metric`
- `ranking`

Extraction is column-aligned and validated against the dataset profile.

#### 3. Query Planning
For dataset-backed questions, the query agent asks the LLM to generate restricted pandas code.

The generated code must:
- query only the provided DataFrame
- assign the final output to `result_df`
- avoid imports, file access, network access, or unsafe operations

For pure explanatory questions, the `definitions` lane bypasses code generation and answers directly from profile context.

#### 4. Execution
Generated code runs in a restricted execution environment with:
- `pd` available
- a copied `dataframe`
- limited built-ins

If execution succeeds, the resulting `result_df` is serialized and passed to the final answer stage.

#### 5. Final Response
The final response is generated in concise natural language based on:
- `result_df` output for dataset-backed questions
- profile context for definition/methodology questions

## Current Intent Boundary
The system currently uses the following intent categories.

### In-Scope Intents
#### `definitions`
Intended primarily for profile-only explanatory questions, such as:
- "How do you calculate P&L in this system?"
- "Which ledger types are used in P&L calculation?"
- schema/field-definition style explanations

#### `dataset_knowledge`
Used for anything that requires actual dataset retrieval, filtering, matching, aggregation, or derived values, such as:
- min/max time coverage
- distinct ledger groups
- property and ledger lookups
- P&L calculations over a time period
- rankings and comparisons

This separation is enforced by:
- LLM intent classification
- a deterministic post-classification eligibility gate for `definitions`

That gate prevents “definition-like” wording from incorrectly bypassing the dataset query path.

### Fallback And Control Intents
#### `general_knowledge`
Used for questions that are outside the dataset domain and should return an out-of-scope response.

#### `ambiguous`
Used when the request is incomplete, unclear, or missing required context and the system should ask a clarification question.

#### `adversarial`
Used for unsafe, manipulative, or policy-violating requests that should be blocked.

#### `gibberish`
Used for unparseable or non-meaningful input that cannot be understood reliably.

## Key Design Decisions
### 1. Combined Intent + Extraction
Intent classification and entity extraction are done in one structured LLM call.

Why:
- reduces latency compared to separate calls
- keeps routing and extraction context aligned

### 2. Structured Contracts Everywhere
The LLM is not asked for free-form intermediate outputs.

Instead, structured schemas are used for:
- routing + extraction
- code generation

This reduces parsing ambiguity and makes failures easier to handle.

### 3. Code Generation Over Hardcoded Query Branches
For dataset-backed requests, the system uses generated pandas plans instead of a large set of handcrafted query handlers.

Why:
- supports broader natural-language coverage
- keeps the query surface flexible
- still allows safety constraints through execution policies

### 4. Narrow “Definitions” Lane
A separate profile-only lane exists for methodology/explanatory answers.

This is intentionally narrow so that:
- conceptual questions can be answered quickly
- lookup/range/distinct-value questions still use real data

### 5. Restricted Execution
Generated code runs in a guarded environment with forbidden code patterns and limited globals.

This keeps the system safer while still allowing useful pandas operations.

## Challenges And How They Were Solved
### 1. Balancing Intent Routing Precision
Natural-language asset-management questions often look similar on the surface, but require different handling paths. Some requests are best answered directly from system context, while others require dataset retrieval, aggregation, or filtering. A key challenge was designing routing logic that stays accurate across both straightforward and ambiguous user inputs.

Solution:
- combine intent classification and entity extraction in one structured step
- validate extracted structure inside the graph before continuing
- use a narrow `definitions` lane only for truly profile-only explanatory questions
- route dataset-backed requests through the full retrieval path

### 2. Managing Latency Across Multiple LLM Stages
The system uses several LLM-assisted steps: intent routing, structured extraction, query planning, and response generation. A practical challenge was keeping the interaction responsive while preserving enough reasoning quality for complex requests.

Solution:
- combine routing and extraction into one structured call
- reuse a compact startup dataset profile instead of sending raw dataset content to prompts
- use a direct-answer path for purely explanatory questions
- keep the more expensive dataset-query path only for requests that actually need it

### 3. Executing Flexible Queries Safely
Because the agent supports a wide range of natural-language requests, it needed a flexible way to build dataset queries without relying on a large set of handcrafted handlers. At the same time, generated execution had to remain constrained and predictable.

Solution:
- use generated pandas query plans with a strict structured contract
- validate generated code against forbidden patterns before execution
- run code in a restricted execution environment with limited globals
- validate the resulting output before producing the final answer

### 4. Robust Error Handling
The agent needed to remain stable when requests were incomplete, unsupported, or referred to unavailable data.

Solution:
- canonical fallback messages for unsupported or out-of-scope requests
- clarification branches for incomplete or ambiguous inputs
- explicit dataset validation before query execution
- graceful handling of empty query results and unsupported metrics

## Project Structure
The core submission-relevant files are:

- [ui_streamlit.py](app/ui_streamlit.py): Streamlit interface
- [constants.py](config/constants.py): intent labels, fallback messages, shared constants
- [prompts.py](config/prompts.py): LLM prompt templates
- [settings.py](config/settings.py): runtime configuration
- [flow.py](src/graph/flow.py): LangGraph wiring
- [nodes.py](src/graph/nodes.py): graph node logic
- [states.py](src/graph/states.py): state model and helpers
- [models.py](src/contracts/models.py): structured schemas
- [policies.py](src/contracts/policies.py): execution safety rules
- [repository.py](src/data/repository.py): cached dataset loading
- [profiler.py](src/data/profiler.py): startup dataset profile
- [intent_service.py](src/services/intent_service.py): structured intent + extraction call
- [codegen_service.py](src/services/codegen_service.py): code generation and execution
- [response_service.py](src/services/response_service.py): final answer generation
- [llm_client.py](src/services/llm_client.py): OpenAI client wrapper
