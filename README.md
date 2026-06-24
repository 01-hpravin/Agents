# Agents

A collection of AI agents built with CrewAI.

## Projects

| Agent | Description |
|-------|-------------|
| Nanobook | Generates a structured technical NanoBook from documentation on any given topic |

## Setup

```bash
pip install crewai crewai-tools python-dotenv
```

Create a `.env` file in the project root:

```
MODEL=your_model_name_here
OPENAI_API_KEY=your_key_here
SERPER_API_KEY=your_key_here
```

## Usage

```bash
cd nanobook
crewai install
crewai run
```

## Tech Stack

- [CrewAI](https://crewai.com) — agent orchestration
- [SerperDev](https://serper.dev) — web search
- OpenAI — LLM
