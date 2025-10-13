# 🏙️ Urban RL Agent Decision Explainer

Interactive system for explaining Reinforcement Learning agent decisions applied to urban planning, with automatic technical vocabulary adaptation based on the audience.

## 📋 Description

This web portal allows explaining the decisions of an urban RL agent at **three different technical levels**:

- **Level 1 - Common Language**: For general public without technical knowledge
- **Level 2 - Professional**: For architects and urban planners with specialized terminology
- **Level 3 - RL Technical**: For data scientists and RL/ML researchers

## ✨ Main Features

### 🎚️ Technical Levels System

- Automatic vocabulary adaptation based on audience
- Three complexity levels with specialized prompts
- Comparison mode to view all three responses simultaneously

### 💾 Intelligent Caching System

- MD5-based cache to avoid duplicate queries
- Instant responses for repeated queries
- Cost reduction and response time improvement

### 📊 Metrics and Analysis

- Generation time per response
- Token count (input/output)
- Cache usage indicator
- Complete conversation history

### 🔧 Flexible Configuration

- Environment variables configurable from the interface
- Advanced system prompt customization
- Predefined presets (simple and technical)
- Automatic alerts for missing configuration

### 📜 History and Traceability

- Complete log of all queries
- Timestamps and metrics per conversation
- Ability to clear history and cache

## 🚀 Installation and Usage

### Prerequisites

- Python 3.11 or higher
- Access to an OpenAI-compatible API (OpenAI, Azure, etc.)

### Local Installation

1. **Clone the repository**

```bash
git clone https://github.com/enriquegomeztagle/urban-rl-explainer-en.git
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com
OPENAI_MODEL=gpt-4
```

5. **Run the application**

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### 🐳 Docker Installation

1. **Build the image**

```bash
docker build -t rl-urbanism-explainer .
```

2. **Run the container**

```bash
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_api_key_here \
  -e OPENAI_BASE_URL=https://api.openai.com \
  -e OPENAI_MODEL=gpt-4 \
  rl-urbanism-explainer
```

Or using an `.env` file:

```bash
docker run -p 8501:8501 --env-file .env rl-urbanism-explainer
```

3. **Access the application**

Open browser at `http://localhost:8501`

### 1. Initial Configuration

- Verify that environment variables are configured (left sidebar)
- Select the desired technical level with the slider

### 2. Load Data

- Use the "Load example" button for predefined presets
- Or enter manually:
  - **Agent objective**: What it seeks to optimize
  - **Rules**: Constraints and policies
  - **Calculations**: Metrics and evaluations performed
  - **Question**: The specific query about the decision

### 3. Generate Response

- **Individual Mode**: Generates response at the selected technical level
- **Comparison Mode**: Generates all 3 responses simultaneously

### 4. Review Results

- View generation metrics (time, tokens, cache)
- Review the explanation adapted to the selected level
- Consult previous conversation history

## 🏗️ Technical Architecture

### Main Components

```
app.py
├── Configuration
│   ├── Environment variables (OPENAI_API_KEY, BASE_URL, MODEL)
│   ├── Session State (history, cache, metrics)
│   └── Presets (technical and simple examples)
│
├── Prompt System
│   ├── BASE_CRITICAL_RULES (shared rules)
│   ├── SYSTEM_PROMPT_LEVEL_CONFIG (level configurations)
│   └── build_system_prompt() (dynamic composition)
│
├── Response Generation
│   ├── generate_response_from_inputs() (with MD5 cache)
│   ├── LangChain + ChatOpenAI
│   └── Error handling and metrics
│
└── Streamlit Interface
    ├── Sidebar (env vars configuration)
    ├── Technical level selector
    ├── Input form
    ├── Tabs (individual vs comparison)
    └── Expanders (history and cache)
```

### Caching System

- **Key**: MD5(objective + rules + calculations + question + technical_level)
- **Storage**: st.session_state (in memory)
- **Benefits**: Instant responses, cost reduction

### Prompt Architecture

1. **Shared base**: Critical rules common to all levels
2. **Level configuration**: Role, task, extra rules, format, examples
3. **Dynamic composition**: `build_system_prompt(level)` assembles the final prompt

## 🔍 Advanced Features

### Anti-Hallucination System

The system includes multiple safeguards to prevent the LLM from inventing information:

- Explicit critical rules in the prompt
- Context validation
- Instructions to respond "I don't know" when information is missing
- Clear separation between format examples and real data

### Progress Tracking

- Multi-stage progress bars
- Real-time processing states
- Visual feedback of cache vs new generation

### Error Handling

- Connection error handling
- Timeout handling
- Descriptive error messages with suggested solutions

## 📊 Available Metrics

- **Generation time**: Total duration of the query
- **Technical level**: Level used for the response
- **Cache status**: Whether the response comes from cache
- **Tokens**: Input/output token count (when available)
- **Timestamp**: Timestamp of each conversation

## 🛠️ Technologies Used

- **Streamlit**: Interactive web framework
- **LangChain**: LLM integration
- **OpenAI API**: Natural language generation
- **Loguru**: Logging system
- **Python-dotenv**: Environment variable management

## 📝 Dependencies

See `requirements.txt` for the complete list of dependencies.

Main ones:

- `streamlit`
- `langchain-openai`
- `loguru`

## 🔐 Security

- API keys are stored in environment variables
- Input type="password" for sensitive fields in the UI
- No credentials stored in source code
- Recommended to use `.env` file

## 🐛 Troubleshooting

### Error: Missing environment variables

**Solution**: Verify that `.env` contains all required variables or configure them from the sidebar.

### Error: Connection timeout

**Solution**: Verify internet connectivity and validity of OPENAI_BASE_URL.

### Error: Invalid API key

**Solution**: Check that OPENAI_API_KEY is valid and has necessary permissions.

### Inconsistent responses

**Solution**: Clear cache from the "💾 Cache Statistics" expander.

## 🚧 Known Limitations

- Cache is volatile (lost when closing the session)
- Maximum tokens per response: 1024 (configurable in code)
- Requires internet connection for LLM queries

## 📄 License

This project was developed for research purposes in urban planning using Reinforcement Learning agents and is intended solely for academic evaluation.

### Copyright Notice

© 2025 Enrique Ulises Baez Gomez Tagle. All rights reserved.

### Terms of Use

**Research and Evaluation Only**: This codebase is created specifically for academic research in urban planning using Reinforcement Learning techniques.

**No Commercial Use**: This project cannot be used for commercial purposes without explicit written permission from the author.

**No Redistribution**: The code cannot be redistributed, copied, or modified without authorization from the author.

**Attribution Required**: Any reference to this work must include proper attribution to the author.

### Intellectual Property

This project represents original work developed independently for research in urban planning and Reinforcement Learning. The architecture, implementation, and design decisions are intellectual property of the author.

### Contact

For questions about this project or licenses, please contact:

- **Author**: Enrique Ulises Baez Gomez Tagle
- **GitHub**: [@enriquegomeztagle](https://github.com/enriquegomeztagle)
- **Purpose**: RL Research Project in Urban Planning

---

## 👨‍💻 Author

**Enrique Ulises Baez Gomez Tagle**

GitHub: [@enriquegomeztagle](https://github.com/enriquegomeztagle)

---

**Made with ❤️ for urban planning research and AI explainability**
