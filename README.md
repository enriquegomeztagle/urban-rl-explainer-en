# 🏙️ Explainable AI Framework for Urban RL Agent Decisions

------------------------------------------------------------------------

## 📋 System Overview

This web-based platform facilitates the interpretation of an urban RL
agent's decisions across **three distinct technical strata**:

-   **Level 1 --- Layperson**: Designed for the general public,
    utilizing accessible language devoid of technical jargon.
-   **Level 2 --- Professional**: Tailored for architects and urban
    planners, employing domain-specific urban design terminology.
-   **Level 3 --- Technical RL**: Engineered for data scientists and
    ML/RL researchers, utilizing advanced algorithmic and computational
    nomenclature.

------------------------------------------------------------------------

## ✨ Core Capabilities

### 🎚️ Dynamic Technical Stratification

-   Automated vocabulary modulation based on the target demographic.
-   Three distinct complexity tiers governed by specialized prompt
    engineering.
-   Comparative analysis mode allowing simultaneous evaluation of all
    three response levels.

### 💾 Intelligent Caching Mechanism

-   MD5 hash-based caching to eliminate redundant API queries.
-   Instantaneous data retrieval for previously processed queries.
-   Significant optimization of computational latency and API
    operational costs.

### 📊 Analytics and Telemetry

-   Generation latency tracking per query.
-   Token utilization metrics (input/output).
-   Cache hit-rate indicators.
-   Comprehensive conversational logging and audibility.

### 🔧 Flexible Configuration Protocol

-   User Interface (UI)-driven environment variable configuration.
-   Advanced customization capabilities for system prompts.
-   Predefined operational baselines (simplified and technical presets).
-   Automated diagnostic alerts for missing configuration parameters.

### 📜 Traceability and Auditing

-   Exhaustive registry of all system interactions.
-   Session-specific timestamps and performance metrics.
-   Administrator capabilities for cache and history purging.

------------------------------------------------------------------------

# 🚀 Deployment and Operational Workflow

## 🔹 Prerequisites

-   Python 3.11 or higher.
-   Access to an OpenAI-compatible API endpoint (e.g., OpenAI, Azure).

------------------------------------------------------------------------

## 💻 Local Environment Setup

### 1️⃣ Clone the repository

``` bash
git clone https://github.com/enriquegomeztagle/urban-rl-explainer-es
```

### 2️⃣ Initialize virtual environment

``` bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

``` bash
pip install -r requirements.txt
```

### 4️⃣ Configure environment variables

Create a `.env` file in the project's root directory:

``` env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com
OPENAI_MODEL=gpt-4
```

### 5️⃣ Execute the application

``` bash
streamlit run app.py
```

The application will deploy locally at:

http://localhost:8501

------------------------------------------------------------------------

## 🐳 Docker Containerization

### Build the image

``` bash
docker build -t rl-urbanism-explainer .
```

### Run the container

``` bash
docker run -p 8501:8501   -e OPENAI_API_KEY=your_api_key_here   -e OPENAI_BASE_URL=https://api.openai.com   -e OPENAI_MODEL=gpt-4   rl-urbanism-explainer
```

### Alternatively, using the `.env` file:

``` bash
docker run -p 8501:8501 --env-file .env rl-urbanism-explainer
```

Access the application:

http://localhost:8501

------------------------------------------------------------------------

# 🧭 User Protocol

## 🔹 Initial Configuration

Verify environment variables via the left sidebar and select the desired
technical tier using the UI slider.

## 🔹 Data Ingestion

Utilize the "Load Example" function for predefined presets, or manually
input:

-   **Agent Objective**: The optimization target.
-   **Policy Constraints**: Operational rules and restrictions.
-   **Computational Metrics**: Calculations and evaluations performed by
    the agent.
-   **Query**: The specific question regarding the agent's decision.

## 🔹 Response Generation

-   **Isolated Mode**: Generates a singular response at the selected
    technical tier.
-   **Comparative Mode**: Synthesizes all three technical responses
    simultaneously.

## 🔹 Output Analysis

Review generation telemetry (latency, tokens, cache status), analyze the
adapted explanation, and consult the interaction history.

------------------------------------------------------------------------

# 🏗️ System Architecture

## Core Components

``` text
app.py
├── Configuration Module
│   ├── Environment Variables (OPENAI_API_KEY, BASE_URL, MODEL)
│   ├── Session State Management (history, cache, metrics)
│   └── Presets (technical and simplified baselines)
│
├── Prompt Engineering Framework
│   ├── BASE_CRITICAL_RULES (shared operational constraints)
│   ├── SYSTEM_PROMPT_LEVEL_CONFIG (tier-specific configurations)
│   └── build_system_prompt() (dynamic prompt synthesis)
│
├── Generation Pipeline
│   ├── generate_response_from_inputs() (MD5 caching integration)
│   ├── LangChain + ChatOpenAI integration
│   └── Exception and telemetry management
│
└── Streamlit Interface
    ├── Sidebar (environment configuration)
    ├── Technical tier selector
    ├── Data ingestion form
    ├── Tabs (isolated vs. comparative views)
    └── Expanders (history and cache auditing)
```

------------------------------------------------------------------------

## 🔐 Caching Architecture

-   **Cryptographic Key**:
    `MD5(objective + constraints + metrics + query + technical_tier)`
-   **Storage Protocol**: `st.session_state` (In-memory execution)
-   **Systematic Benefits**: Near-instantaneous response times and
    substantial cost reduction.

------------------------------------------------------------------------

## 🧠 Prompt Synthesis Architecture

-   **Shared Foundation**: Critical constraints universally applied
    across all tiers.
-   **Tier-Specific Configuration**: Defined roles, tasks, supplementary
    rules, formatting, and few-shot examples.
-   **Dynamic Assembly**: `build_system_prompt(level)` function
    aggregates the final contextual payload.

------------------------------------------------------------------------

# 🔍 Advanced Mechanisms

## Hallucination Mitigation Framework

-   Explicit critical constraints embedded within the foundational
    prompt.
-   Rigorous validation of provided contextual data.
-   Strict directives mandating an "unknown" response state when data is
    insufficient.
-   Absolute delineation between structural formatting examples and
    empirical data inputs.

## Processing Telemetry

-   Multi-stage execution progress indicators.
-   Real-time processing state tracking.
-   Visual feedback delineating cached retrieval versus novel
    generation.

## Exception Management

-   Robust connection error handling protocols.
-   Timeout resolution handling.
-   Descriptive error logging coupled with actionable resolution
    pathways.

------------------------------------------------------------------------

# 📊 Available Metrics

-   **Generation Latency**: Total duration of the query execution.
-   **Technical Tier**: The specific stratum utilized for the output.
-   **Cache Status**: Binary indicator of whether the output was
    retrieved from memory.
-   **Token Utilization**: Aggregation of input/output tokens (where
    supported by the API).
-   **Timestamp**: Exact chronological marker of each interaction.

------------------------------------------------------------------------

# 🛠️ Technology Stack

-   **Streamlit**: Interactive web application framework.
-   **LangChain**: LLM orchestration and integration.
-   **OpenAI API**: Natural language generation engine.
-   **Loguru**: Advanced logging framework.
-   **Python-dotenv**: Environment variable management.

------------------------------------------------------------------------

# 📦 Dependencies

Refer to `requirements.txt` for the exhaustive dependency manifest.

Primary libraries include:

-   streamlit
-   langchain-openai
-   loguru

------------------------------------------------------------------------

# 🔒 Security Protocols

-   API keys are strictly managed via environment variables.
-   Sensitive UI input fields utilize `type="password"` masking.
-   Zero credential hardcoding within the source code.
-   Strong recommendation for localized `.env` file utilization.

------------------------------------------------------------------------

# 🐛 Troubleshooting

### Exception: Missing Environment Variables

Resolution: Ensure the `.env` file contains all mandatory variables or
configure them directly via the application sidebar.

### Exception: Connection Timeout

Resolution: Verify network connectivity and validate the integrity of
the `OPENAI_BASE_URL`.

### Exception: Invalid API Key

Resolution: Confirm the `OPENAI_API_KEY` is active and possesses the
requisite endpoint permissions.

### Issue: Output Inconsistencies

Resolution: Purge the session cache via the "Cache Statistics" expander
module.

------------------------------------------------------------------------

# 🚧 Known Limitations

-   Cache memory is volatile and flushes upon session termination.
-   Maximum output token constraint is capped at 1024 (programmatically
    configurable).
-   Requires an active internet connection for LLM API routing.

------------------------------------------------------------------------

# 📄 Licensing and Usage Terms

This codebase was developed exclusively for research purposes in urban
planning utilizing Reinforcement Learning agents and is strictly
intended for academic evaluation.

## Copyright Notice

All rights reserved.

## Terms of Use

-   Research and Evaluation Only.
-   No Commercial Usage without explicit written authorization.
-   No Redistribution without prior consent.
-   Attribution Required.

## Intellectual Property

This project constitutes original, independently developed work focusing
on urban planning research and Explainable AI within Reinforcement
Learning.

------------------------------------------------------------------------

# 📫 Contact Information

Domain: RL Research in Urban Planning

## Authors

-   Enrique Ulises Baez Gomez Tagle
-   Daniel Adrián Contreras Olivas
-   Francisco Javier Tallabs Utrilla

GitHub: @enriquegomeztagle
