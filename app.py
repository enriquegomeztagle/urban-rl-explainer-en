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

load_dotenv()

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
    "- EXAMPLES are only to show the format, DO NOT use their data. Use ONLY the data from the current context.",
    "- Do not repeat or quote the person's message literally. Do not include their text in the response.",
    "- Do not invent data, numbers, metrics, calculations, or decisions that are not in the context.",
    '- Do not use metatext like "Understood", "Next" or similar.',
    "- Keep the output EXACTLY in the format indicated below.",
]

SYSTEM_PROMPT_LEVEL_CONFIG = {
    1: {
        "role": "an URBAN EXPLAINER for general non-technical public",
        "task": "Your task: explain in simple and everyday language why the agent made an urban decision.",
        "rules_extra": [
            "- Forbidden to use technical jargon of any kind (neither specialized urbanism, nor RL).",
            '- Use everyday words: "neighborhood" instead of "zone", "walking" instead of "pedestrian mobility".',
            "- Maximum 200 words. Close, friendly and conversational tone.",
        ],
        "format_section": (
            "OUTPUT FORMAT (EXACT):\n\n"
            "Given the urban agent's objective, which is {objective},\n"
            "and the established rules:\n"
            "{rules_in_simple}\n\n"
            "The following calculations were performed:\n"
            "{calculations_in_simple}\n\n"
            "{question_context}\n\n"
            "[Generate the appropriate decision based on the context provided]"
        ),
        "style_guides": [
            '- Explain with very simple words: "neighborhoods", "proximity", "variety of places", "paths", "not saturating".',
            "- Avoid any technical terms. Speak as if explaining to a neighbor.",
            "- Mental structure: objective → practical rules → what was reviewed → final decision.",
            "- Generate the decision based on the question context and provided information, not from hardcoded rules.",
        ],
        "principles_section": (
            "PRINCIPLES (EXPLAIN SIMPLE IN 1–2 SENTENCES):\n"
            "- That people can walk to the services they need.\n"
            "- That there is variety of services without them piling up.\n"
            "- That paths and streets connect everything well."
        ),
        "example_section": (
            "FORMAT EXAMPLE (DO NOT use this data, it's only to show the structure):\n"
            "If you had the objective 'bring services closer to housing', rules about 'favor proximity', "
            "and calculations of 'benefited houses', the response would follow this pattern:\n\n"
            "Given the urban agent's objective, which is [real objective from context],\n"
            "and the established rules:\n"
            "- [rule 1 from context]\n"
            "- [rule 2 from context]\n"
            "The following calculations were performed:\n"
            "- [calculation 1 from context]\n"
            "- [calculation 2 from context]\n\n"
            "[decision based on real context]\n\n"
            "IMPORTANT: Replace EVERYTHING between [ ] with information from the current context. "
            "If something says 'I don't know', respond that this information is missing."
        ),
    },
    2: {
        "role": "an URBAN EXPLAINER for urban design and architecture professionals",
        "task": "Your task: explain from an urban planning perspective why the agent made a decision.",
        "rules_extra": [
            "- Use professional urbanism and urban design terminology.",
            "- Avoid specific RL/ML jargon (do not mention Q-learning, DQN, policies, Bellman, etc.).",
            "- Allowed terms: zoning, urban morphology, accessibility, density, mixed use, road network, connectivity, facilities.",
            "- Maximum 250 words. Professional but accessible tone.",
            "- IMPORTANT: Use EXACTLY the information provided in the context. Do not say 'I don't know' if information is available.",
            "- MANDATORY: If there is text in the calculation and decision fields, use it directly without questioning its completeness.",
        ],
        "format_section": (
            "OUTPUT FORMAT (EXACT):\n\n"
            "Given the urban agent's objective, which is {objective},\n"
            "and the established rules:\n"
            "{rules_in_simple}\n\n"
            "The following calculations were performed:\n"
            "{calculations_in_simple}\n\n"
            "{question_context}\n\n"
            "[Generate the appropriate decision based on the context provided]"
        ),
        "style_guides": [
            "- Use urban design vocabulary: pedestrian accessibility, coverage radius, use compatibility, road structure, service density.",
            "- Connect with sustainable urbanism principles: proximity, functional diversity, permeability.",
            "- Structure: planning objective → design criteria → spatial analysis → justified decision.",
            "- Generate the decision based on the question context and provided information, not from hardcoded rules.",
        ],
        "principles_section": (
            "URBAN DESIGN PRINCIPLES (INCLUDE IN CONCLUSION IN 1–2 SENTENCES):\n"
            "- Proximity/walkability: optimize pedestrian influence radii towards essential facilities.\n"
            "- Diversity/compatibility: promote mixed use avoiding functional conflicts and saturation.\n"
            "- Connectivity: integrate the intervention into the road structure and mobility system."
        ),
        "example_section": "",
    },
    3: {
        "role": "a TECHNICAL EXPLAINER of Reinforcement Learning systems applied to urban planning",
        "task": "Your task: explain from the RL/DQN perspective why the agent made a decision.",
        "rules_extra": [
            "- Use technical RL terminology: Q-learning, DQN, policy, value function, reward, state, action, exploration/exploitation.",
            "- Allowed technical terms: Q(s,a), policy π, reward function R, state space, action space, Bellman equation, epsilon-greedy, experience replay.",
            "- If information about technical parameters is missing, request it specifically.",
            "- Maximum 300 words. Technical-academic tone.",
            "- You can reference network architectures, hyperparameters, reward functions.",
        ],
        "format_section": (
            "OUTPUT FORMAT (EXACT):\n\n"
            "Given the RL agent's objective, which is {objective},\n"
            "and the implemented policy:\n"
            "{rules_in_simple}\n\n"
            "States and actions were evaluated:\n"
            "{calculations_in_simple}\n\n"
            "{question_context}\n\n"
            "[Generate the appropriate decision based on the context provided]"
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
            "FORMAT EXAMPLE (DO NOT use this invented data, it's only to show the structure):\n"
            "If you had a defined reward function, a specific policy and calculated Q-values, "
            "the response would follow this pattern:\n\n"
            "Given the RL agent's objective, which is [real objective from context],\n"
            "and the implemented policy:\n"
            "- [policy 1 from context]\n"
            "- [policy 2 from context]\n"
            "States and actions were evaluated:\n"
            "- [evaluation 1 from context]\n"
            "- [evaluation 2 from context]\n\n"
            "[action based on real context]\n\n"
            "CRITICAL: Replace EVERYTHING between [ ] with data from the provided context. "
            "DO NOT invent Q-values, weights, epsilon, or any parameters. If they are not in the context, say 'I don't know'."
        ),
    },
}


def build_system_prompt(level: int) -> str:
    config = SYSTEM_PROMPT_LEVEL_CONFIG.get(level, SYSTEM_PROMPT_LEVEL_CONFIG[1])

    rules = BASE_CRITICAL_RULES + config.get("rules_extra", [])
    rules_block = "CRITICAL RULES (MANDATORY):\n" + "\n".join(rules)

    style_guides = config.get("style_guides", [])
    style_block = "STYLE GUIDES:\n" + "\n".join(style_guides) if style_guides else ""

    sections = [
        f"You are {config['role']}",
        config["task"],
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
    st.subheader("Environment Variables")

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

    st.subheader("Configuration Status")
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

st.subheader("🎚️ Technical Explanation Level")
technical_level = st.select_slider(
    "Select the level of technicality in the response:",
    options=[1, 2, 3],
    value=st.session_state.get("technical_level", 1),
    format_func=lambda x: {
        1: "1️⃣ Common Language (General Public)",
        2: "2️⃣ Professional Language (Architect/Urban Planner)",
        3: "3️⃣ Technical Language (Deep Q-Learning / RL)",
    }[x],
    help="""💡 Adjust the vocabulary and complexity of the explanation:
    
    • Level 1: Everyday language without technical terms (ideal for citizens)
    • Level 2: Professional urban planning terminology (for architects/urban planners)  
    • Level 3: Technical RL/ML vocabulary (for data scientists)
    
    Responses adapt completely to the selected level.""",
)
st.session_state["technical_level"] = technical_level

level_descriptions = {
    1: "💬 **Simple and everyday language** - Perfect for explaining to neighbors or general public without technical knowledge.",
    2: "🏗️ **Professional urban planning terminology** - Uses urban design concepts, zoning, and planning for architects and designers.",
    3: "🤖 **Reinforcement Learning vocabulary** - Technical explanation with Q-learning, policies, reward functions and network architectures.",
}
st.info(level_descriptions[technical_level])

with st.expander("🔧 Customize System Prompt (Advanced)", expanded=False):
    st.caption("Modify the system prompt to change the agent's behavior.")
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
    if not has_override and st.session_state["custom_prompt_level"] != technical_level:
        st.session_state["custom_system_prompt"] = default_prompt

    st.session_state["custom_prompt_level"] = technical_level

    custom_system_prompt = st.text_area(
        "System Prompt",
        height=300,
        help="This is the prompt that guides the LLM's behavior",
        key="custom_system_prompt",
    )
    if st.button("Apply custom prompt"):
        st.session_state["system_prompt_override"] = custom_system_prompt
        st.success("✓ Custom prompt applied")
    if st.button("Restore default prompt"):
        if "system_prompt_override" in st.session_state:
            del st.session_state["system_prompt_override"]
        st.session_state["custom_system_prompt"] = default_prompt
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
        p = PRESET_SIMPLE if preset_choice.startswith("Simple") else PRESET_TECHNICAL
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
    help="📋 Define the agent's constraints and policies. Example: do not build in protected areas, maintain service diversity, respect maximum capacity.",
)
calculations = st.text_area(
    "3) Calculations performed",
    placeholder=current_placeholder_calculations,
    height=140,
    key="calculations",
    help="🧮 Specify the metrics and evaluations performed. Example: Manhattan distances, compatibility matrix, count of nearby services.",
)
question = st.text_area(
    "4) Person's question",
    placeholder=current_placeholder_question,
    height=80,
    key="question",
    help="❓ Formulate the question about the agent's decision. Example: Why did it place the hospital here? Why didn't it choose this other location?",
)

SYSTEM_PROMPT = """
You are an URBAN EXPLAINER for non-technical public.
Your task: explain in clear language why the agent made an urban decision.

CRITICAL RULES (MANDATORY):
- Do not repeat or quote the person's message literally. Do not include their text in the response.
- Forbidden to use RL jargon (do not say Q-learning, DQN, policy, Bellman, etc.).
- If information is missing, respond "I don't know" and suggest 1–2 concrete data points that would need to be requested.
- Maximum 200 words. Close and respectful tone.
- Do not invent data or metrics.
- Do not use metatext like "Understood", "Next" or similar.
- Keep the output EXACTLY in the format indicated below.

OUTPUT FORMAT (EXACT):

Given the urban agent's objective, which is {objective},
and the established rules:
{rules_in_simple}

The following calculations were performed:
{calculations_in_simple}

{question_context}

[Generate the appropriate decision based on the context provided]

STYLE GUIDES:
- Explain rules and calculations with simple words (neighborhoods, proximity, variety of services, connections, avoiding saturation).
- Avoid technical terms, formulas or symbols.
- Mental structure like practical syllogism: end (objective) → norms (rules) → perception/calculation (computations) → action (decision).

PROXIMITY PRINCIPLES (INCLUDE IN CONCLUSION IN 1–2 SENTENCES):
- Proximity/walkability: improve real walking distances to essential services.
- Diversity/compatibility: distribute different services without use conflicts.
- Connectivity: integrate the decision with streets and transportation for effective access.
(Explicitly summarize how the decision favors proximity + diversity/compatibility + connectivity.)

EXAMPLE (few-shot; imitate the tone and structure, DO NOT COPY the user's content):
Agent response:
Given the RL agent's objective, which is to bring education and green areas closer to housing,
and the established rules:
- Favor that people walk little to reach key services.
- Maintain variety without saturating a single area.
- Locate uses that get along well with each other.
The following calculations were performed:
- Counted how many houses would gain walking access.
- Verified that the area would not be overloaded and that connected paths exist.
- Compared nearby alternatives with less benefit.
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
    logger.error("Missing OpenAI environment variables. LLM will not be initialized.")
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
            temperature=0.2,
            max_tokens=1024,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=base_url,
            timeout=70,
            max_retries=3,
        )
        logger.info(f"[LLM] Initialized with model: {OPENAI_MODEL}")
        if test_llm_connection():
            logger.info("[LLM] Connection test passed")
        else:
            logger.warning(
                "[LLM] Connection test failed (puede seguir funcionando si el proveedor no expone /)"
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
        "- " + rules_clean
        if rules_clean
        else "- No specific agent rules were provided."
    )
    if (
        "The following calculations were performed" in rules_in_simple
        and not rules_in_simple.endswith("\n")
    ):
        rules_in_simple = rules_in_simple.replace(
            "The following calculations were performed",
            "\nThe following calculations were performed",
        )
    if not rules_in_simple.endswith("\n"):
        rules_in_simple += "\n"
    rules_in_simple = re.sub(
        r"(?<!\n)\s*The following calculations were performed",
        "\n\nThe following calculations were performed",
        rules_in_simple,
    )

    calculations_in_simple = (
        "- " + calculations_clean
        if calculations_clean
        else "- No specific calculations performed by the agent were provided."
    )
    if (
        "The following calculations were performed" in calculations_in_simple
        and not calculations_in_simple.endswith("\n")
    ):
        calculations_in_simple = calculations_in_simple.replace(
            "The following calculations were performed",
            "\nThe following calculations were performed",
        )
    if not calculations_in_simple.endswith("\n"):
        calculations_in_simple += "\n"
    calculations_in_simple = re.sub(
        r"(?<!\n)\s*The following calculations were performed",
        "\n\nThe following calculations were performed",
        calculations_in_simple,
    )

    tech_level = st.session_state.get("technical_level", 1)
    default_prompt = get_system_prompt_by_level(tech_level)
    active_system_prompt = st.session_state.get(
        "system_prompt_override", default_prompt
    )

    question_context = (
        f"\n\nQuestion to address: {question_clean}" if question_clean else ""
    )

    format_params = {
        "objective": (
            objective_clean
            if objective_clean
            else "No specific objective was specified"
        ),
        "rules_in_simple": rules_in_simple,
        "calculations_in_simple": calculations_in_simple,
        "question_context": question_context,
    }

    try:
        prompt_text = active_system_prompt.format(**format_params)
    except KeyError as e:
        logger.warning(f"Missing format parameter: {e}. Using fallback prompt.")
        prompt_text = active_system_prompt

    prompt_text = re.sub(
        r"(?<!\n)\s*The following calculations were performed",
        "\n\nThe following calculations were performed",
        prompt_text,
    )
    prompt_text = re.sub(
        r"(?<!\n)\s*That is why it was decided:",
        "\n\nThat is why it was decided:",
        prompt_text,
    )
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
            st.session_state.get("placeholder_calculations", PLACEHOLDER_CALCULATIONS),
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
        objective_effective, rules_effective, calculations_effective, question_effective
    )

    try:
        logger.info(
            f"[LLM] Generating response for question: {question_effective[:80]}..."
        )
        is_custom = "system_prompt_override" in st.session_state
        logger.info(f"[LLM] Using {'custom' if is_custom else 'default'} system prompt")

        result = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "CRITICAL: Respond ONLY with information from the provided context. "
                        "DO NOT invent data, numbers, metrics or decisions. "
                        "If the context says 'I don't know', you must respond that this information is missing. "
                        "The examples in the prompt are ONLY for format, DO NOT use their data. "
                        "Respond in the exact format indicated. "
                        "Do not include prefaces or metatext like 'Understood', 'I'm ready', 'Next', etc."
                    )
                ),
                HumanMessage(content=prompt),
            ]
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

tab1, tab2 = st.tabs(["💬 Individual Response", "🔄 Comparison Mode (3 Levels)"])

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

        progress_bar.progress(40, text="🤖 Consulting the model...")
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
            st.error("⚠️ An error occurred when calling the LLM.")
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
                    help="💾 Indicates if the response was retrieved from cache (faster) or generated again",
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
    
    This mode generates responses simultaneously in the 3 technical levels:
    - 🗣️ **Level 1**: Common language for general public
    - 🏗️ **Level 2**: Professional urban planning terminology
    - 🤖 **Level 3**: Technical RL/ML vocabulary
    
    Useful to see how the explanation changes according to the audience."""
    )

    comparison_btn = st.button(
        "🔄 Generate comparison (3 levels)",
        type="primary",
        disabled=(llm is None),
        key="generate_comparison",
        help="💡 Generate 3 simultaneous responses (one for each technical level) to compare vocabularies and approaches.",
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

            cache_status = "💾 (cached)" if metrics.get("cached") else "🆕 (new)"
            with status_container:
                st.success(
                    f"✅ {level_names[level]} completed {cache_status} - {metrics['duration']:.2f}s"
                )
            time.sleep(0.3)

            st.session_state["technical_level"] = original_level

        progress_bar.progress(100, text="✨ Comparison completed!")
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
                cache_text = "💾 Cache" if m["cached"] else "🆕 New"
                st.caption(cache_text)
                if "tokens" in m:
                    st.caption(f"🔤 {m['tokens'].get('total_tokens', 'N/A')} tokens")

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

        for idx, entry in enumerate(reversed(st.session_state["conversation_history"])):
            with st.container(border=True):
                col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
                with col_h1:
                    st.caption(f"🕐 {entry['timestamp']}")
                with col_h2:
                    st.caption(f"🎚️ Level {entry['technical_level']}")
                with col_h3:
                    cached_text = "💾 Cache" if entry["metrics"]["cached"] else "🆕 New"
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
        st.info("No conversations in history yet. Generate a response to get started.")

with st.expander("💾 Cache Statistics", expanded=False):
    st.caption(
        "💡 **What is cache?** The system saves generated responses to avoid duplicate queries to the LLM, saving time and costs. Identical queries return instantly."
    )
    cache_size = len(st.session_state["response_cache"])
    st.metric("Cached responses", cache_size)

    if cache_size > 0:
        if st.button("🗑️ Clear cache"):
            st.session_state["response_cache"] = {}
            st.success("✓ Cache cleared")
            st.rerun()

        st.markdown("**Cache entries:**")
        for key, value in st.session_state["response_cache"].items():
            with st.container(border=True):
                st.caption(f"🔑 Key: `{key[:16]}...`")
                st.caption(f"🕐 Timestamp: {value['timestamp']}")
                st.caption(f"⏱️ Original time: {value['metrics']['duration']:.2f}s")
    else:
        st.info("No responses in cache yet.")
