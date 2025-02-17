# Evaluation Scripts

This directory contains scripts for evaluating language models on various benchmarks and metrics. AgentDojo evaluations are handled separately in a fork: [https://github.com/normster/agentdojo](https://github.com/normster/agentdojo).

## Setup

1. **torchllms**: Core library for model inference using vLLM/OpenAI providers
2. **lm-eval-harness**: Custom fork of [lm-eval-harness](https://github.com/EleutherAI/lm-eval-harness) for MMLU evaluation with torchllms models
3. **llm_rules**: External dependency for rule following evaluation: [https://github.com/normster/llm_rules](https://github.com/normster/llm_rules)

## Usage

Usage for running/scoring benchmarks varies but is mostly documented in the different `run*.sh` and `read*.sh` shell scripts in each benchmark directory.

Results in output directories can be read using `read.sh` scripts that calculate metrics with bootstrap confidence intervals.

RealGuardrails, Monkey Island stress test, S-IFEval, and TensorTrust use a simple inference wrapper implemented in `torchllms` which dispatches to either vLLM for local models or the OpenAI API for hosted models. Other OpenAI-compatible providers can be used by setting the `base_url` parameter in [torchllms/inference/providers.py](https://github.com/normster/torchllms/blob/main/torchllms/inference/providers.py#L182), which `torchllms` currently handles automatically for DeepSeek (using together.ai) and Gemini.
