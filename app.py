import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
import requests
from dotenv import load_dotenv
import re
import hashlib
import time
from datetime import datetime

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

print(f"Loaded environment variables")
print(f"MODEL: {OPENAI_MODEL}")

if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []
if "response_cache" not in st.session_state:
    st.session_state["response_cache"] = {}
if "metrics_history" not in st.session_state:
    st.session_state["metrics_history"] = []

PLACEHOLDER_OBJECTIVE = "E.g.: Maximize that all housing has access to health, education, green areas and supermarkets within 15 minutes."
PLACEHOLDER_RULES = "E.g.: Do not build on rivers; prioritize compatibility between uses; maintain connectivity with existing streets; avoid saturating a service in a single area, etc."
PLACEHOLDER_CALCULATIONS = "E.g.: Count nearby services per housing unit; measure walking distances; apply a simple compatibility matrix; avoid duplicating the same service if there is already sufficient coverage."
PLACEHOLDER_QUESTION = "E.g.: Why did you build a hospital here?"

PLACEHOLDER_TECH_OBJECTIVE = r"Maximize cumulative return \sum_t \gamma^t r_t under optimal policy \pi^\*, reducing mean distance to essential services with threshold N=15 (Manhattan)."
PLACEHOLDER_TECH_RULES = (
    "• ε-greedy policy with ε0=1.0 and Gompertz decay: ε(x)=exp(exp(-c·x + b)), b=-e, c=-0.03883259.\n"
    "• Learning rate α=0.5, discount γ=0.95.\n"
    "• Spatial compatibility according to matrix C∈[1,5]; evaluate neighbors at Manhattan distance 2.\n"
    "• Do not build on obstacles (rivers/non-buildable zones); respect road connectivity."
)
PLACEHOLDER_TECH_CALCULATIONS = (
    "• Q-update (Bellman): Q(s_t,a_t) ← Q(s_t,a_t) + α [ r_{t+1} + γ max_a Q(s_{t+1},a) − Q(s_t,a_t) ].\n"
    "• Residential reward: sum compatibilities of nearby services weighted by maxAmount=2 per type; decrement when exceeded; total per city R=∑_i R_i.\n"
    "• Coverage: count distinct services per residence within N=15.\n"
    "• (DQN Alternative) MLP [128,64,128], dropout 0.22; identical ε-greedy exploration."
)
PLACEHOLDER_TECH_QUESTION = r"Why did policy \pi choose to place hospital in cell (i,j) given current Q(s,a) and maxAmount per service?"

PROMPT_FORMATTING_CONFIG = {
    "rules_prefix": "- ",
    "rules_no_data_text": "No specific agent rules were provided.",
    "calculations_prefix": "- ",
    "calculations_no_data_text": "No specific calculations performed by the agent were provided.",
    "objective_no_data_text": "No specific objective was specified",
    "decision_default_text": "Take this action at this site",
    "calculations_phrase": "The calculations were performed",
    "decision_phrase": "That is why it was decided:",
}

PRESET_SIMPLE = {
    "objective": PLACEHOLDER_OBJECTIVE,
    "rules": PLACEHOLDER_RULES,
    "calculations": PLACEHOLDER_CALCULATIONS,
    "question": PLACEHOLDER_QUESTION,
}
PRESET_TECHNICAL = {
    "objective": PLACEHOLDER_TECH_OBJECTIVE,
    "rules": PLACEHOLDER_TECH_RULES,
    "calculations": PLACEHOLDER_TECH_CALCULATIONS,
    "question": PLACEHOLDER_TECH_QUESTION,
}

BASE_CRITICAL_RULES = [
    "- NEVER invent information that is not explicitly in the provided context.",
    "- Only respond 'I don't know' if the fields are literally empty or contain only placeholder text.",
    "- The EXAMPLES are only to show the format, DO NOT use their data. Use ONLY the data from the current context.",
    "- Do not repeat or literally quote the person's message. Do not include their text in the response.",
    "- Do not invent data, numbers, metrics, calculations, or decisions that are not in the context.",
    '- Do not use metatext like "Understood", "Next" or similar.',
    "- Keep the output EXACTLY in the format indicated below.",
]

SYSTEM_PROMPT_LEVEL_CONFIG = {
    1: {
        "rol": "an URBAN EXPLAINER for general non-technical audience",
        "tarea": "Your task: explain in simple, everyday language why the agent made an urban decision.",
        "rules_extra": [
            "- Forbidden to use technical jargon of any kind (neither specialized urbanism nor RL).",
            '- Use everyday words: "neighborhood" instead of "zone", "walk" instead of "pedestrian mobility".',
            "- Maximum 200 words. Close, friendly and conversational tone.",
        ],
        "format_section": (
            "OUTPUT FORMAT (EXACT):\n\n"
            "Given the urban agent objective, which is {objective},\n"
            "and the established rules:\n"
            "{rules_in_simple}\n\n"
            "The calculations were performed:\n"
            "{calculations_in_simple}\n\n"
            "That is why it was decided: {clear_decision}"
        ),
        "style_guides": [
            '- Explain with very simple words: "neighborhoods", "proximity", "variety of places", "paths", "not saturate".',
            "- Avoid any technicalities. Speak as if explaining to a neighbor.",
            "- Mental structure: objective → practical rules → what was reviewed → final decision.",
        ],
        "principles_section": (
            "PRINCIPLES (EXPLAIN SIMPLY IN 1–2 SENTENCES):\n"
            "- That people can walk to the services they need.\n"
            "- That there is variety of services without them piling up.\n"
            "- That paths and streets connect everything well."
        ),
        "example_section": (
            "FORMAT EXAMPLE (DO NOT use this data, it is only to show the structure):\n"
            "If you had the objective 'bring services closer to homes', rules about 'favoring proximity', "
            "and calculations of 'benefited homes', the response would follow this pattern:\n\n"
            "Given the urban agent objective, which is [real objective from context],\n"
            "and the established rules:\n"
            "- [rule 1 from context]\n"
            "- [rule 2 from context]\n"
            "The calculations were performed:\n"
            "- [calculation 1 from context]\n"
            "- [calculation 2 from context]\n\n"
            "That is why it was decided: [decision based on real context]\n\n"
            "IMPORTANT: Replace EVERYTHING between [ ] with information from the current context. "
            "If something says 'I don't know', respond that this information is missing."
        ),
    },
    2: {
        "rol": "an URBAN EXPLAINER for urban design and architecture professionals",
        "tarea": "Your task: explain from an urban planning perspective why the agent made a decision.",
        "rules_extra": [
            "- Use professional urbanism and urban design terminology.",
            "- Avoid specific RL/ML jargon (do not mention Q-learning, DQN, policies, Bellman, etc.).",
            "- Allowed terms: zoning, urban morphology, accessibility, density, mixed use, road network, connectivity, equipment.",
            "- Maximum 250 words. Professional but accessible tone.",
            "- IMPORTANT: Use EXACTLY the information provided in the context. Do not say 'I don't know' if information is available.",
            "- MANDATORY: If there is text in the calculations and decision fields, use it directly without questioning its completeness.",
        ],
        "format_section": (
            "OUTPUT FORMAT (EXACT):\n\n"
            "Given the urban agent objective, which is {objective},\n"
            "and the established rules:\n"
            "{rules_in_simple}\n\n"
            "The calculations were performed:\n"
            "{calculations_in_simple}\n\n"
            "That is why it was decided: {clear_decision}"
        ),
        "style_guides": [
            "- Use urban design vocabulary: pedestrian accessibility, coverage radius, use compatibility, road structure, service density.",
            "- Connect with sustainable urbanism principles: proximity, functional diversity, permeability.",
            "- Structure: planning objective → design criteria → spatial analysis → grounded decision.",
        ],
        "principles_section": (
            "URBAN DESIGN PRINCIPLES (INCLUDE IN CONCLUSION IN 1–2 SENTENCES):\n"
            "- Proximity/walkability: optimize pedestrian influence radii towards essential equipment.\n"
            "- Diversity/compatibility: promote mixed use avoiding functional conflicts and saturation.\n"
            "- Connectivity: integrate the intervention into the road structure and mobility system."
        ),
        "example_section": "",
    },
    3: {
        "rol": "a TECHNICAL EXPLAINER of Reinforcement Learning systems applied to urban planning",
        "tarea": "Your task: explain from the RL/DQN perspective why the agent made a decision.",
        "rules_extra": [
            "- Use technical RL terminology: Q-learning, DQN, policy, value function, reward, state, action, exploration/exploitation.",
            "- Allowed technical terms: Q(s,a), policy π, reward function R, state space, action space, Bellman equation, epsilon-greedy, experience replay.",
            "- If information about technical parameters is missing, request it specifically.",
            "- Maximum 300 words. Technical-academic tone.",
            "- You can reference network architectures, hyperparameters, reward functions.",
        ],
        "format_section": (
            "OUTPUT FORMAT (EXACT):\n\n"
            "Given the RL agent objective, which is {objective},\n"
            "and the implemented policy:\n"
            "{rules_in_simple}\n\n"
            "States and actions were evaluated:\n"
            "{calculations_in_simple}\n\n"
            "That is why the action was selected: {clear_decision}"
        ),
        "style_guides": [
            "- Explain in RL terms: Q value function, expected reward maximization, environment state.",
            "- Use technical notation when appropriate: Q(s,a), R(s,a,s'), γ (discount factor), ε (epsilon).",
            "- Structure: objective/reward function → policy and decision rules → Q-values evaluation → optimal action selection.",
        ],
        "principles_section": (
            "RL SYSTEM PRINCIPLES (INCLUDE IN CONCLUSION IN 1–2 SENTENCES):\n"
            "- Optimization: maximize cumulative reward considering pedestrian proximity, service diversity and road connectivity.\n"
            "- Trade-offs: balance between exploration (new configurations) and exploitation (proven strategies).\n"
            "- Convergence: how this action contributes to the optimal policy π* according to estimated Q-values."
        ),
        "example_section": (
            "FORMAT EXAMPLE (DO NOT use this invented data, it is only to show the structure):\n"
            "If you had a defined reward function, a specific policy and calculated Q-values, "
            "the response would follow this pattern:\n\n"
            "Given the RL agent objective, which is [real objective from context],\n"
            "and the implemented policy:\n"
            "- [policy 1 from context]\n"
            "- [policy 2 from context]\n"
            "States and actions were evaluated:\n"
            "- [evaluation 1 from context]\n"
            "- [evaluation 2 from context]\n\n"
            "That is why the action was selected: [action based on real context]\n\n"
            "CRITICAL: Replace EVERYTHING between [ ] with data from the provided context. "
            "DO NOT invent Q-values, weights, epsilon, or any parameter. If they are not in the context, say 'I don't know'."
        ),
    },
}


def build_system_prompt(level: int) -> str:
    config = SYSTEM_PROMPT_LEVEL_CONFIG.get(level, SYSTEM_PROMPT_LEVEL_CONFIG[1])

    rules = BASE_CRITICAL_RULES + config.get("rules_extra", [])
    rules_block = "CRITICAL RULES (MANDATORY):\n" + "\n".join(rules)

    style_guides = config.get("style_guides", [])
    style_block = (
        "STYLE GUIDELINES:\n" + "\n".join(style_guides) if style_guides else ""
    )

    sections = [
        f"You are {config['rol']}",
        config["tarea"],
        rules_block,
        config.get("format_section", "").strip(),
        style_block,
    ]

    principles_section = config.get("principles_section", "").strip()
    if principles_section:
        sections.append(principles_section)

    example_section = config.get("example_section", "").strip()
    if example_section:
        sections.append(example_section)

    return "\n\n".join(section for section in sections if section).strip()


def get_system_prompt_by_level(level: int) -> str:
    return build_system_prompt(level)


st.set_page_config(page_title="Urban Agent Explainer", page_icon="🏙️", layout="centered")
st.title("Urban Agent Decision Explainer")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("Environment variables")

    sidebar_api_key = st.text_input(
        "OPENAI_API_KEY",
        value=OPENAI_API_KEY or "",
        type="password",
        help="OpenAI API Key",
    )
    sidebar_base_url = st.text_input(
        "OPENAI_BASE_URL",
        value=OPENAI_BASE_URL or "",
        help="OpenAI service base URL",
    )
    sidebar_model = st.text_input(
        "OPENAI_MODEL", value=OPENAI_MODEL or "", help="Model to use"
    )

    if sidebar_api_key:
        OPENAI_API_KEY = sidebar_api_key
    if sidebar_base_url:
        OPENAI_BASE_URL = sidebar_base_url
    if sidebar_model:
        OPENAI_MODEL = sidebar_model

    st.divider()

    st.subheader("Configuration status")
    if OPENAI_API_KEY:
        st.success("✓ API Key configured")
    else:
        st.error("✗ API Key missing")

    if OPENAI_BASE_URL:
        st.success("✓ Base URL configured")
    else:
        st.error("✗ Base URL missing")

    if OPENAI_MODEL:
        st.success("✓ Model configured")
    else:
        st.error("✗ Model missing")

missing_vars = []
if not OPENAI_API_KEY:
    missing_vars.append("OPENAI_API_KEY")
if not OPENAI_BASE_URL:
    missing_vars.append("OPENAI_BASE_URL")
if not OPENAI_MODEL:
    missing_vars.append("OPENAI_MODEL")

if missing_vars:
    st.error(
        f"⚠️ **Missing environment variables:** {', '.join(missing_vars)}. "
        f"Please configure them in the .env file or in the sidebar."
    )


main_tab, second_tab = st.tabs(
    ["🏙️ Practical Syllogism Agent Explainer", "🤖 General RL Explanation"]
)

with main_tab:
    st.subheader("🎚️ Technical Explanation Level")
    technical_level = st.radio(
        "Select the level of technicality in the response:",
        options=[1, 2, 3],
        index=st.session_state.get("technical_level", 1) - 1,
        format_func=lambda x: {
            1: "1️⃣ Common Language (General Audiende)",
            2: "2️⃣ Professional Language (Architect/Urban Planner)",
            3: "3️⃣ Technical Language (Deep Q-Learning / RL)",
        }[x],
        horizontal=True,
        help="""💡 Adjust the vocabulary and complexity of the explanation:
    
        • Level 1: Everyday language without technical terms (ideal for citizens)
        • Level 2: Professional urban planning terminology (for architects/urban planners)  
        • Level 3: Technical RL/ML vocabulary (for data scientists)
    
        Responses adapt completely to the selected level.""",
    )
    st.session_state["technical_level"] = technical_level

    level_descriptions = {
        1: "💬 **Simple and everyday language** - Perfect for explaining to neighbors or general audience without technical knowledge.",
        2: "🏗️ **Professional urban planning terminology** - Uses urban design concepts, zoning, and planning for architects and designers.",
        3: "🤖 **Reinforcement Learning vocabulary** - Technical explanation with Q-learning, policies, reward functions and network architectures.",
    }
    st.info(level_descriptions[technical_level])

    with st.expander("🔧 Customize System Prompt (Advanced)", expanded=False):
        st.caption("Modify the system prompt to change the agent behavior.")
        default_prompt = get_system_prompt_by_level(technical_level)
        prompt_value = st.session_state.get("system_prompt_override", default_prompt)

        if "custom_prompt_level" not in st.session_state:
            st.session_state["custom_prompt_level"] = technical_level
        if "custom_system_prompt" not in st.session_state:
            st.session_state["custom_system_prompt"] = prompt_value

        has_override = "system_prompt_override" in st.session_state
        if (
            has_override
            and st.session_state["custom_system_prompt"]
            != st.session_state["system_prompt_override"]
        ):
            st.session_state["custom_system_prompt"] = st.session_state[
                "system_prompt_override"
            ]
        if (
            not has_override
            and st.session_state["custom_prompt_level"] != technical_level
        ):
            st.session_state["custom_system_prompt"] = default_prompt

        st.session_state["custom_prompt_level"] = technical_level

        custom_system_prompt = st.text_area(
            "System Prompt",
            height=300,
            help="This is the prompt that guides the LLM behavior",
            key="custom_system_prompt",
        )
        if st.button("Apply custom prompt"):
            st.session_state["system_prompt_override"] = custom_system_prompt
            st.success("✓ Custom prompt applied")
        if st.button("Restore default prompt"):
            if "system_prompt_override" in st.session_state:
                del st.session_state["system_prompt_override"]
            if "custom_system_prompt" in st.session_state:
                del st.session_state["custom_system_prompt"]
            st.success("✓ Prompt restored to default value")
            st.rerun()

    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        preset_choice = st.selectbox(
            "Example preset",
            options=["Simple (non-technical)", "Technical (RL)"],
            index=0,
            help="Choose an example and press 'Load example' to fill the fields.",
        )
    with col_p2:
        if st.button("Load example"):
            p = (
                PRESET_SIMPLE
                if preset_choice.startswith("Simple")
                else PRESET_TECHNICAL
            )
            st.session_state["objective"] = p["objective"]
            st.session_state["rules"] = p["rules"]
            st.session_state["calculations"] = p["calculations"]
            st.session_state["question"] = p["question"]

    if preset_choice.startswith("Simple"):
        current_placeholder_objective = PLACEHOLDER_OBJECTIVE
        current_placeholder_rules = PLACEHOLDER_RULES
        current_placeholder_calculations = PLACEHOLDER_CALCULATIONS
        current_placeholder_question = PLACEHOLDER_QUESTION
    else:
        current_placeholder_objective = PLACEHOLDER_TECH_OBJECTIVE
        current_placeholder_rules = PLACEHOLDER_TECH_RULES
        current_placeholder_calculations = PLACEHOLDER_TECH_CALCULATIONS
        current_placeholder_question = PLACEHOLDER_TECH_QUESTION

    st.session_state["placeholder_objective"] = current_placeholder_objective
    st.session_state["placeholder_rules"] = current_placeholder_rules
    st.session_state["placeholder_calculations"] = current_placeholder_calculations
    st.session_state["placeholder_question"] = current_placeholder_question

    objective = st.text_area(
        "1) Agent objective",
        placeholder=current_placeholder_objective,
        height=100,
        key="objective",
        help="🎯 Describe what the agent seeks to optimize. Example: maximize accessibility to services, minimize walking distances.",
    )
    rules = st.text_area(
        "2) Rules the agent follows",
        placeholder=current_placeholder_rules,
        height=140,
        key="rules",
        help="📋 Define the agent constraints and policies. Example: do not build in protected areas, maintain service diversity, respect maximum capacity.",
    )
    calculations = st.text_area(
        "3) Calculations performed",
        placeholder=current_placeholder_calculations,
        height=140,
        key="calculations",
        help="🧮 Specify the metrics and evaluations performed. Example: Manhattan distances, compatibility matrix, nearby services count.",
    )
    question = st.text_area(
        "4) Person question",
        placeholder=current_placeholder_question,
        height=80,
        key="question",
        help="❓ Formulate the question about the agent decision. Example: Why did it place the hospital here? Why didn't it choose this other location?",
    )

    SYSTEM_PROMPT = """
    You are an URBAN EXPLAINER for non-technical audience.
    Your task: explain in clear language why the agent made an urban decision.

    CRITICAL RULES (MANDATORY):
    - Do not repeat or literally quote the person's message. Do not include their text in the response.
    - Forbidden to use RL jargon (do not say Q-learning, DQN, policy, Bellman, etc.).
    - If information is missing, respond "I don't know" and suggest 1–2 concrete data points that would be needed.
    - Maximum 200 words. Close and respectful tone.
    - Do not invent data or metrics.
    - Do not use metatext like "Understood", "Next" or similar.
    - Keep the output EXACTLY in the format indicated below.

    OUTPUT FORMAT (EXACT):

    Given the urban agent objective, which is {objective},
    and the established rules:
    {rules_in_simple}

    The calculations were performed:
    {calculations_in_simple}

    That is why it was decided: {clear_decision}

    STYLE GUIDELINES:
    - Explain rules and calculations with simple words (neighborhoods, proximity, variety of services, connections, avoid saturation).
    - Avoid technicalities, formulas or symbols.
    - Mental structure like practical syllogism: end (objective) → norms (rules) → perception/calculation (computations) → action (decision).

    PROXIMITY PRINCIPLES (INCLUDE IN CONCLUSION IN 1–2 SENTENCES):
    - Proximity/walkability: improve real walking distances to essential services.
    - Diversity/compatibility: distribute different services without use conflicts.
    - Connectivity: integrate the decision with streets and transport for effective access.
    (Explicitly summarize how the decision favors proximity + diversity/compatibility + connectivity.)

    EXAMPLE (MINI few-shot; imitate the tone and structure, DO NOT COPY the user's content):
    Agent response:
    Given the RL agent objective, which is to bring education and green areas closer to housing,
    and the established rules:
    - Favor people walking short distances to reach key services.
    - Maintain variety without saturating a single zone.
    - Locate uses that are compatible with each other.
    The calculations were performed:
    - We counted how many houses would gain walking access.
    - We verified that the zone would not be overloaded and that connected paths existed.
    - Nearby alternatives with less benefit were compared.

    That is why it was decided: Locate a school next to the park
    """

    def test_llm_connection() -> bool:
        try:
            if not OPENAI_BASE_URL:
                return False
            base = OPENAI_BASE_URL.rstrip("/")
            headers = {}
            if OPENAI_API_KEY:
                headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
            url = base
            resp = requests.get(url, headers=headers, timeout=8)
            logger.info(f"[LLM] Connection probe {url} → {resp.status_code}")
            return resp.status_code < 400 or resp.status_code in (401, 403, 404)
        except Exception as e:
            logger.warning(f"[LLM] Connection test failed (non-fatal): {e}")
            return False

    def value_or_default(val: str | None, default: str) -> str:
        if val is None:
            return default
        v = val.strip()
        return v if v else default

    def _clean(s: str) -> str:
        if not s:
            return s
        s = s.strip()
        lowers = s.lower()
        if lowers.startswith(("ej.:", "ej:", "ejemplo:", "e.g.")):
            s = s.split(":", 1)[-1].strip()
        return s.strip(' "')

    if not OPENAI_API_KEY or not OPENAI_BASE_URL or not OPENAI_MODEL:
        logger.error(
            "Missing OpenAI environment variables. The LLM will not be initialized."
        )
        llm = None
    else:
        try:
            try:
                from langchain.schema import BaseCache

                ChatOpenAI.model_rebuild()
            except (ImportError, AttributeError):
                pass

            base_url = OPENAI_BASE_URL
            llm = ChatOpenAI(
                model=OPENAI_MODEL,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=base_url,
                timeout=70,
                max_retries=3,
                model_kwargs={},
            )
            logger.info(f"[LLM] Initialized with model: {OPENAI_MODEL}")
            if test_llm_connection():
                logger.info("[LLM] Connection test passed")
            else:
                logger.warning(
                    "[LLM] Connection test failed (may still work if provider doesn't expose /)"
                )
        except Exception as e:
            logger.error(f"[LLM] Failed to initialize: {e}")
            llm = None

    def build_user_prompt(
        objective: str, rules: str, calculations: str, question: str
    ) -> str:
        objective_clean = (objective or "").strip()
        rules_clean = (rules or "").strip()
        calculations_clean = (calculations or "").strip()
        question_clean = (question or "").strip()

        rules_in_simple = (
            PROMPT_FORMATTING_CONFIG["rules_prefix"] + rules_clean
            if rules_clean
            else PROMPT_FORMATTING_CONFIG["rules_prefix"]
            + PROMPT_FORMATTING_CONFIG["rules_no_data_text"]
        )
        calculations_phrase = PROMPT_FORMATTING_CONFIG["calculations_phrase"]
        if calculations_phrase in rules_in_simple and not rules_in_simple.endswith(
            "\n"
        ):
            rules_in_simple = rules_in_simple.replace(
                calculations_phrase, "\n" + calculations_phrase
            )
        if not rules_in_simple.endswith("\n"):
            rules_in_simple += "\n"
        rules_in_simple = re.sub(
            rf"(?<!\n)\s*{re.escape(calculations_phrase)}",
            f"\n\n{calculations_phrase}",
            rules_in_simple,
        )

        calculations_in_simple = (
            PROMPT_FORMATTING_CONFIG["calculations_prefix"] + calculations_clean
            if calculations_clean
            else PROMPT_FORMATTING_CONFIG["calculations_prefix"]
            + PROMPT_FORMATTING_CONFIG["calculations_no_data_text"]
        )
        if (
            calculations_phrase in calculations_in_simple
            and not calculations_in_simple.endswith("\n")
        ):
            calculations_in_simple = calculations_in_simple.replace(
                calculations_phrase, "\n" + calculations_phrase
            )
        if not calculations_in_simple.endswith("\n"):
            calculations_in_simple += "\n"
        calculations_in_simple = re.sub(
            rf"(?<!\n)\s*{re.escape(calculations_phrase)}",
            f"\n\n{calculations_phrase}",
            calculations_in_simple,
        )

        # Build user context (always the same structure)
        prompt_text = f"""**Agent's Objective:**
{objective_clean if objective_clean else PROMPT_FORMATTING_CONFIG["objective_no_data_text"]}

**Rules to Follow:**
{rules_in_simple}

**Calculations Performed:**
{calculations_in_simple}

**User's Question:**
{question_clean}"""
        
        return prompt_text

    def generate_response_from_inputs(
        objective_in: str, rules_in: str, calculations_in: str, question_in: str
    ) -> tuple[str | None, dict]:
        metrics = {
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "cached": False,
            "technical_level": st.session_state.get("technical_level", 1),
            "timestamp": datetime.now().isoformat(),
        }

        if not llm:
            logger.error("LLM is not initialized. Cannot generate response.")
            metrics["end_time"] = time.time()
            metrics["duration"] = metrics["end_time"] - metrics["start_time"]
            return None, metrics

        objective_effective = _clean(
            value_or_default(
                objective_in,
                st.session_state.get("placeholder_objective", PLACEHOLDER_OBJECTIVE),
            )
        )
        rules_effective = _clean(
            value_or_default(
                rules_in, st.session_state.get("placeholder_rules", PLACEHOLDER_RULES)
            )
        )
        calculations_effective = _clean(
            value_or_default(
                calculations_in,
                st.session_state.get(
                    "placeholder_calculations", PLACEHOLDER_CALCULATIONS
                ),
            )
        )
        question_effective = _clean(
            value_or_default(
                question_in,
                st.session_state.get("placeholder_question", PLACEHOLDER_QUESTION),
            )
        )

        cache_key = hashlib.md5(
            f"{objective_effective}|{rules_effective}|{calculations_effective}|{question_effective}|{metrics['technical_level']}".encode()
        ).hexdigest()

        if cache_key in st.session_state["response_cache"]:
            logger.info(f"[CACHE] Using cached response for key: {cache_key[:8]}...")
            cached_data = st.session_state["response_cache"][cache_key]
            metrics["cached"] = True
            metrics["end_time"] = time.time()
            metrics["duration"] = metrics["end_time"] - metrics["start_time"]
            return cached_data["response"], metrics

        prompt = build_user_prompt(
            objective_effective,
            rules_effective,
            calculations_effective,
            question_effective,
        )

        try:
            logger.info(
                f"[LLM] Generating response for pregunta: {question_effective[:80]}..."
            )
            is_custom = "system_prompt_override" in st.session_state
            logger.info(
                f"[LLM] Using {'custom' if is_custom else 'default'} system prompt"
            )

            # Get the appropriate system prompt
            tech_level = st.session_state.get("technical_level", 1)
            default_system_prompt = get_system_prompt_by_level(tech_level)
            active_system_prompt = st.session_state.get(
                "system_prompt_override", default_system_prompt
            )

            system_message_content = (
                f"{active_system_prompt}\n\n"
                "CRITICAL: Respond ONLY with information from the provided context. "
                "DO NOT invent data, numbers, metrics or decisions. "
                "If the context says 'I don't know', you must respond that this information is missing. "
                "The examples in the prompt are ONLY for format, DO NOT use their data. "
                "Respond in the exact format indicated. "
                "Do not include prefaces or metatext like 'Understood', 'I'm ready', 'Next', etc."
            )
            
            logger.info("=" * 80)
            logger.info("[PROMPT] COMPLETE SYSTEM MESSAGE:")
            for line in system_message_content.split('\n'):
                logger.info(line)
            logger.info("=" * 80)
            logger.info("[PROMPT] COMPLETE USER CONTEXT:")
            for line in prompt.split('\n'):
                logger.info(line)
            logger.info("=" * 80)
            logger.info(f"[PROMPT] USER QUESTION: {question_effective}")
            logger.info("=" * 80)

            result = llm.invoke(
                [
                    SystemMessage(content=system_message_content),
                    HumanMessage(content=prompt),
                ],
                config={"configurable": {"model_kwargs": {}}},
            )
            response = (result.content or "").strip()

            if hasattr(result, "response_metadata"):
                metrics["tokens"] = result.response_metadata.get("token_usage", {})

            metrics["end_time"] = time.time()
            metrics["duration"] = metrics["end_time"] - metrics["start_time"]

            st.session_state["response_cache"][cache_key] = {
                "response": response,
                "timestamp": metrics["timestamp"],
                "metrics": metrics.copy(),
            }

            logger.info(
                f"[LLM] Response generated in {metrics['duration']:.2f}s: {response[:80]}..."
            )
            return response, metrics
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[ERROR] Connection error to LLM endpoint: {e}")
            metrics["error"] = str(e)
            metrics["end_time"] = time.time()
            metrics["duration"] = metrics["end_time"] - metrics["start_time"]
            return None, metrics
        except requests.exceptions.Timeout as e:
            logger.error(f"[ERROR] Timeout error with LLM endpoint: {e}")
            metrics["error"] = str(e)
            metrics["end_time"] = time.time()
            metrics["duration"] = metrics["end_time"] - metrics["start_time"]
            return None, metrics
        except Exception as e:
            logger.error(f"[ERROR] Response generation failed: {e}")
            metrics["error"] = str(e)
            metrics["end_time"] = time.time()
            metrics["duration"] = metrics["end_time"] - metrics["start_time"]
            return None, metrics

    st.divider()

    tab1, tab2 = st.tabs(["💬 Single Response", "Comparison Mode (3 Levels)"])

    with tab1:
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            generate_btn = st.button(
                "🚀 Generate response",
                type="primary",
                disabled=(llm is None),
                key="generate_single",
                help="💡 Generate an explanation using the selected technical level. Responses are cached for repeated queries.",
                use_container_width=True,
            )
        with col_btn2:
            if st.session_state.get("conversation_history"):
                total_conversations = len(st.session_state["conversation_history"])
                st.metric(
                    "💬 Total",
                    total_conversations,
                    help="Total number of conversations generated",
                )

        if generate_btn:
            progress_bar = st.progress(0, text="🔄 Initializing...")
            status_text = st.empty()

            progress_bar.progress(20, text="📝 Building prompt...")
            status_text.info("⚙️ Preparing context for the model...")
            time.sleep(0.3)

            progress_bar.progress(40, text="🤖 Querying the model...")
            status_text.info(
                f"🎚️ Using Level {st.session_state.get('technical_level', 1)} - {['Common Language', 'Professional', 'Technical RL'][st.session_state.get('technical_level', 1) - 1]}"
            )

            answer, metrics = generate_response_from_inputs(
                objective, rules, calculations, question
            )

            progress_bar.progress(80, text="✅ Processing response...")
            status_text.success(
                f"{'💾 Response retrieved from cache' if metrics.get('cached') else '🆕 Response generated'} in {metrics['duration']:.2f}s"
            )
            time.sleep(0.5)

            progress_bar.progress(100, text="✨ Completed!")
            time.sleep(0.3)
            progress_bar.empty()
            status_text.empty()

            if answer is None:
                st.error("⚠️ An error occurred calling the LLM.")
                if "error" in metrics:
                    with st.expander("🔍 Error details", expanded=True):
                        st.error(f"**Error:** {metrics['error']}")
                        st.info(
                            """💡 **Possible solutions:**
                        - Verify that environment variables are configured correctly
                        - Check your internet connection
                        - Confirm that the model is available
                        - Try again in a few moments"""
                        )
            else:
                question_text = value_or_default(
                    question,
                    st.session_state.get("placeholder_question", PLACEHOLDER_QUESTION),
                )
                st.session_state["conversation_history"].append(
                    {
                        "timestamp": metrics["timestamp"],
                        "question": question_text,
                        "answer": answer,
                        "metrics": metrics,
                        "technical_level": metrics["technical_level"],
                    }
                )
                st.session_state["metrics_history"].append(metrics)

                st.markdown("### 💬 Response")

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric(
                        "⏱️ Time",
                        f"{metrics['duration']:.2f}s",
                        help="⏱️ Total generation time (includes model call and processing)",
                    )
                with col_m2:
                    level_names = {1: "Common", 2: "Professional", 3: "Technical"}
                    st.metric(
                        "🎚️ Level",
                        level_names.get(metrics["technical_level"], "N/A"),
                        help=f"🎚️ Technical level used: {metrics['technical_level']} - Determines the vocabulary and complexity of the response",
                    )
                with col_m3:
                    cache_icon = "💾" if metrics["cached"] else "🆕"
                    cache_status = "Yes" if metrics["cached"] else "No"
                    cache_delta = "Instant" if metrics["cached"] else None
                    st.metric(
                        f"{cache_icon} Cache",
                        cache_status,
                        delta=cache_delta,
                        help="💾 Indicates if the response was retrieved from cache (faster) or generated anew",
                    )
                with col_m4:
                    if "tokens" in metrics:
                        total_tokens = metrics["tokens"].get("total_tokens", "N/A")
                        st.metric(
                            "🔤 Tokens",
                            total_tokens,
                            help="🔤 Total number of tokens processed (input + output). Affects API cost.",
                        )
                    else:
                        st.metric(
                            "🔤 Tokens",
                            "N/A",
                            help="🔤 Token information not available for this model",
                        )

                st.divider()

                with st.chat_message("user"):
                    st.markdown(question_text)
                with st.chat_message("assistant"):
                    st.markdown(answer)

    with tab2:
        st.info(
            """🔄 **Advanced Comparison Mode**
    
        This mode generates responses simultaneously at all 3 technical levels:
        - 🗣️ **Level 1**: Common language for general audience
        - 🏗️ **Level 2**: Professional urban planning terminology
        - 🤖 **Level 3**: Technical RL/ML vocabulary
    
        Useful to see how the explanation changes according to the audience."""
        )

        comparison_btn = st.button(
            "🔄 Generate comparison (3 levels)",
            type="primary",
            disabled=(llm is None),
            key="generate_comparison",
            help="💡 Generate 3 simultaneous responses (one per technical level) to compare vocabularies and approaches.",
            use_container_width=True,
        )

        if comparison_btn:
            progress_bar = st.progress(0, text="🔄 Initializing comparison...")
            status_container = st.empty()

            responses = {}
            all_metrics = {}

            level_names = {
                1: "🗣️ Level 1: Common Language",
                2: "🏗️ Level 2: Professional",
                3: "🤖 Level 3: Technical RL",
            }

            for idx, level in enumerate([1, 2, 3], 1):
                progress = int((idx - 1) / 3 * 100)
                progress_bar.progress(
                    progress, text=f"⚙️ Generating {level_names[level]}..."
                )

                with status_container:
                    st.info(f"🔄 Processing level {idx}/3: {level_names[level]}")

                original_level = st.session_state.get("technical_level", 1)
                st.session_state["technical_level"] = level

                answer, metrics = generate_response_from_inputs(
                    objective, rules, calculations, question
                )

                responses[level] = answer
                all_metrics[level] = metrics

                cache_status = "💾 (caché)" if metrics.get("cached") else "🆕 (new)"
                with status_container:
                    st.success(
                        f"✅ {level_names[level]} completed {cache_status} - {metrics['duration']:.2f}s"
                    )
                time.sleep(0.3)

                st.session_state["technical_level"] = original_level

            progress_bar.progress(100, text="✨ ¡Comparison completed!")
            time.sleep(0.5)
            progress_bar.empty()
            status_container.empty()

            st.markdown("### 🔄 Response Comparison")

            st.markdown("#### 📊 Metrics Summary")
            col_sum1, col_sum2, col_sum3 = st.columns(3)

            level_names = {
                1: "Level 1: Common Language",
                2: "Level 2: Professional",
                3: "Level 3: Technical RL",
            }

            for idx, level in enumerate([1, 2, 3]):
                with [col_sum1, col_sum2, col_sum3][idx]:
                    st.markdown(f"**{level_names[level]}**")
                    m = all_metrics[level]
                    st.metric("⏱️ Time", f"{m['duration']:.2f}s")
                    cache_text = "💾 Cache" if m["cached"] else "🆕 Nueva"
                    st.caption(cache_text)
                    if "tokens" in m:
                        st.caption(
                            f"🔤 {m['tokens'].get('total_tokens', 'N/A')} tokens"
                        )

            st.divider()

            col_r1, col_r2, col_r3 = st.columns(3)

            for idx, level in enumerate([1, 2, 3]):
                with [col_r1, col_r2, col_r3][idx]:
                    st.markdown(f"#### {level_names[level]}")
                    if responses[level]:
                        with st.container(border=True):
                            st.markdown(responses[level])
                    else:
                        st.error("Error generating response")
                        if "error" in all_metrics[level]:
                            st.caption(f"Error: {all_metrics[level]['error']}")

    st.divider()
    with st.expander("📜 Conversation History", expanded=False):
        st.caption(
            "💡 **Tip:** All your previous queries with their metrics are saved here. Useful for reviewing past responses or analyzing patterns."
        )
        if st.session_state["conversation_history"]:
            col_clear1, col_clear2 = st.columns([3, 1])
            with col_clear2:
                if st.button("🗑️ Clear history"):
                    st.session_state["conversation_history"] = []
                    st.session_state["metrics_history"] = []
                    st.rerun()

            st.markdown(
                f"**Total conversations:** {len(st.session_state['conversation_history'])}"
            )
            st.divider()

            for idx, entry in enumerate(
                reversed(st.session_state["conversation_history"])
            ):
                with st.container(border=True):
                    col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
                    with col_h1:
                        st.caption(f"🕐 {entry['timestamp']}")
                    with col_h2:
                        st.caption(f"🎚️ Level {entry['technical_level']}")
                    with col_h3:
                        cached_text = (
                            "💾 Cache" if entry["metrics"]["cached"] else "🆕 Nueva"
                        )
                        st.caption(cached_text)

                    st.markdown("**Question:**")
                    st.info(entry["question"])
                    st.markdown("**Response:**")
                    st.success(entry["answer"])

                    m = entry["metrics"]
                    st.caption(
                        f"⏱️ Time: {m['duration']:.2f}s | 🔤 Tokens: {m.get('tokens', {}).get('total_tokens', 'N/A')}"
                    )
        else:
            st.info(
                "No conversations in history yet. Generate a response to get started."
            )

    with st.expander("💾 Cache Statistics", expanded=False):
        st.caption(
            "💡 **What is cache?** The system saves generated responses to avoid duplicate LLM queries, saving time and costs. Identical queries return instantly."
        )
        cache_size = len(st.session_state["response_cache"])
        st.metric("Responses in cache", cache_size)

        if cache_size > 0:
            if st.button("🗑️ Clear cache"):
                st.session_state["response_cache"] = {}
                st.success("✓ Cache limpiado")
                st.rerun()

            st.markdown("**Cache entries:**")
            for key, value in st.session_state["response_cache"].items():
                with st.container(border=True):
                    st.caption(f"🔑 Key: `{key[:16]}...`")
                    st.caption(f"🕐 Timestamp: {value['timestamp']}")
                    st.caption(f"⏱️ Original time: {value['metrics']['duration']:.2f}s")
        else:
            st.info("No responses in cache yet.")

with second_tab:
    st.subheader("🤖 General Reinforcement Learning Explanation")
    st.info(
        "💡 Ask about general RL concepts: Q-Learning, DQN, policies, value functions, etc."
    )

    if "rl_history" not in st.session_state:
        st.session_state["rl_history"] = []
    if "rl_cache" not in st.session_state:
        st.session_state["rl_cache"] = {}

    rl_question = st.text_area(
        "Your question about Reinforcement Learning",
        placeholder="Ej: What is Q-Learning? How does DQN work? What is an epsilon-greedy policy?",
        height=120,
        key="rl_question",
    )

    if st.button("🚀 Get explanation", type="primary", key="rl_submit"):
        if not rl_question.strip():
            st.warning("⚠️ Please enter a question")
        elif not llm:
            st.error("⚠️ LLM is not initialized")
        else:
            cache_key = hashlib.md5(rl_question.strip().encode()).hexdigest()

            if cache_key in st.session_state["rl_cache"]:
                cached_data = st.session_state["rl_cache"][cache_key]
                st.success("💾 Response retrieved from cache")

                with st.chat_message("user"):
                    st.markdown(rl_question)
                with st.chat_message("assistant"):
                    st.markdown(cached_data["response"])

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("⏱️ Time", f"{cached_data['metrics']['duration']:.2f}s")
                with col2:
                    st.metric("💾 Cache", "Sí", delta="Instant")
            else:
                progress_bar = st.progress(0, text="🔄 Generating explicación...")

                rl_prompt = f"""You are an expert Reinforcement Learning (RL) professor with extensive research and practical application experience.

OBJECTIVE: Explain ONLY Reinforcement Learning concepts. If the question is NOT related to RL, respond: "This question is not related to Reinforcement Learning. Please ask about RL topics like Q-Learning, DQN, policies, value functions, etc."

TOPICS YOU MASTER (ONLY THESE):
• Fundamentals: MDP, states, actions, rewards, policies, value functions
• Classic algorithms: Q-Learning, SARSA, TD-Learning, Monte Carlo
• Deep RL: DQN, Double DQN, Dueling DQN, Rainbow
• Policy-based: REINFORCE, Actor-Critic, A3C, PPO, TRPO
• Exploration: ε-greedy, UCB, Thompson Sampling, curiosity-driven
• Mathematics: Bellman equation, convergence, optimality
• Architectures: MLPs, CNNs for RL, experience replay, target networks

PROHIBITED TOPICS (REJECT THESE QUESTIONS):
• Supervised Learning, classification, regression
• NLP, transformers, LLMs (unless used IN RL)
• General Computer Vision (unless for states in RL)
• Topics unrelated to ML/RL

RESPONSE STYLE (ONLY IF RL TOPIC):
1. Concise definition of the concept
2. Practical intuition (what is it for?)
3. Mathematical formalization (when applicable)
4. Concrete example or pseudocode
5. Advantages/disadvantages or use cases

RULES:
- FIRST verify if the question is about RL. If NOT, politely reject it
- Use standard mathematical notation: π (policy), Q(s,a), V(s), γ (discount), α (learning rate)
- Be precise but accessible
- Maximum 400 words per response

USER QUESTION:
{rl_question}

RESPONSE:"""

                try:
                    start_time = time.time()
                    progress_bar.progress(50, text="🤖 Querying the model...")

                    result = llm.invoke([HumanMessage(content=rl_prompt)])
                    response = (result.content or "").strip()

                    end_time = time.time()
                    duration = end_time - start_time

                    metrics = {
                        "duration": duration,
                        "cached": False,
                        "timestamp": datetime.now().isoformat(),
                    }

                    if hasattr(result, "response_metadata"):
                        metrics["tokens"] = result.response_metadata.get(
                            "token_usage", {}
                        )

                    st.session_state["rl_cache"][cache_key] = {
                        "response": response,
                        "metrics": metrics,
                    }
                    st.session_state["rl_history"].append(
                        {
                            "question": rl_question,
                            "response": response,
                            "metrics": metrics,
                        }
                    )

                    progress_bar.progress(100, text="✨ Completed!")
                    time.sleep(0.3)
                    progress_bar.empty()

                    with st.chat_message("user"):
                        st.markdown(rl_question)
                    with st.chat_message("assistant"):
                        st.markdown(response)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Time", f"{duration:.2f}s")
                    with col2:
                        st.metric("💾 Cache", "No")
                    with col3:
                        if "tokens" in metrics:
                            st.metric(
                                "🔤 Tokens",
                                metrics["tokens"].get("total_tokens", "N/A"),
                            )

                except Exception as e:
                    progress_bar.empty()
                    st.error(f"⚠️ Error: {e}")

    st.divider()

    with st.expander("📜 RL Questions History", expanded=False):
        if st.session_state["rl_history"]:
            if st.button("🗑️ Clear history RL"):
                st.session_state["rl_history"] = []
                st.rerun()

            for idx, entry in enumerate(reversed(st.session_state["rl_history"])):
                with st.container(border=True):
                    st.caption(f"🕐 {entry['metrics']['timestamp']}")
                    st.markdown("**Question:**")
                    st.info(entry["question"])
                    st.markdown("**Response:**")
                    st.success(entry["response"])
                    st.caption(f"⏱️ {entry['metrics']['duration']:.2f}s")
        else:
            st.info("No questions in history yet")
