# LangGraph AI Research Agent

A Python project exploring **LangGraph**, conditional workflows, and LLM tool calling with LangChain and NVIDIA AI Endpoints.

The agent receives a topic, decides whether to call a research tool, retrieves information from a predefined knowledge base, and generates a natural-language response.

## Features

- **LLM integration:** Uses NVIDIA AI Endpoints through LangChain.
- **Tool calling:** Defines a custom `get_topic_info` tool.
- **Conditional routing:** Uses LangGraph to route execution based on whether the LLM requests a tool.
- **Research workflow:** Retrieves topic information from predefined responses.
- **Answer generation:** Uses the retrieved information to generate a final answer.
- **Python implementation:** Uses typed state definitions and modular graph nodes.

## How It Works

1. `choose_topic` selects the topic.
2. `llm_node` asks the language model to decide whether to call the research tool.
3. `route_after_llm` checks whether the model requested a tool call.
4. `research_node` executes the requested tool and stores the result.
5. `final_answer_node` sends the tool result back to the model to generate a final response.
6. If no tool is requested, `no_tool_final_answer_node` generates an answer without using the research tool.

## Tech Stack

- Python
- LangGraph
- LangChain
- NVIDIA AI Endpoints
- python-dotenv

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd langgraph-ai-research-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_nvidia_api_key
```

Get an API key from [NVIDIA Build](https://build.nvidia.com/).

**Never commit your `.env` file or expose your API key.**

### 5. Run the agent

```bash
python exer.py
```

## Current Limitations

- The research tool uses predefined topic information rather than live web search.
- Topics outside the predefined knowledge base return a fallback message.
- The workflow currently handles a single tool call per request.

## Learning Goals

This project was built to practice:

- Defining graph state with `TypedDict`
- Creating LangGraph nodes and edges
- Using conditional edges for workflow control
- Binding tools to a language model
- Executing tool calls and passing results back to an LLM
- Managing environment variables securely

## Future Improvements

- Add live web search for real research.
- Support multiple tool calls.
- Add error handling and retry logic.
- Allow users to supply their own topics.
- Add automated tests.

## License

No license has been specified yet. Add a license if you want to define how others may use, modify, and distribute this project.
