# Synthetic Data Pipeline

This folder contains scripts for generating synthetic training data through a multi-stage process using various LLMs.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a '.env' file with required API keys:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
BRAVE_API_KEY=... # For web search
SCRAPFLY_API_KEY=... # For web browsing
```

## Pipeline Stages

The pipeline processes system prompts (sample data `sample_prompts.jsonl` provided in this repo, or our [full dataset](https://huggingface.co/datasets/normster/RealGuardrails/blob/main/prompts.jsonl.gz)) through multiple stages:

1. **[Optional] Clause Labeling** (Claude 3.5 Sonnet): Labels guardrail clauses in system prompts
2. **Message Generation** (Claude 3.5 Sonnet): Generates both aligned and conflicting user messages
3. **Response Generation** (GPT-4o): Generates responses to user messages with tool use

## Usage

Basic usage with sample data and small models:

```bash
./run.sh
```

To collect responses to user messages from a local model, e.g. Mistral 7B Instruct v0.3, launch a vLLM OpenAI-compatible server and switch out the `base_url` in the OpenAI client initialization. Be careful with chat templates: vLLM handles the formatting automaticaly but you should always verify what's actually being passed in and out of the underlying LLM by instrumenting the vLLM code.

## Formatting

The pipeline produces JSONL files containing:
- Original system prompts
- Generated user messages (aligned/conflicting)
- Model responses with tool interactions.

The `format_tools.ipynb` notebook is used to flatten all the conversations and reformat the tool calls for use with our training scripts.