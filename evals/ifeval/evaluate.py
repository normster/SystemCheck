# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary of evaluating instruction following. See README.md."""

import dataclasses
import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Union

import utils.instructions_registry as instructions_registry
from absl import app, flags

_INPUT_DATA = flags.DEFINE_string(
    "input_data", None, "path to input data", required=True
)

_INPUT_RESPONSE_DATA = flags.DEFINE_string(
    "input_response_data", None, "path to input response data", required=True
)

_INSTRUCTION_IDS_TO_SKIP = flags.DEFINE_list(
    "instruction_ids_to_skip",
    None,
    "Instruction ids to skip",
)

_BOOTSTRAP = flags.DEFINE_integer(
    "bootstrap",
    0,
    "Number of bootstrap samples",
)


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    messages: list[dict[str, str]]
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    key: int
    instruction_id_list: list[str]
    messages: list[dict[str, str]]
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def read_prompt_list(input_jsonl_filename):
    """Read inputs from jsonl."""
    inputs = []
    with open(input_jsonl_filename, "r") as f:
        for line in f:
            example = json.loads(line)
            if _INSTRUCTION_IDS_TO_SKIP.value and len(set(example["instruction_id_list"]).intersection(set(_INSTRUCTION_IDS_TO_SKIP.value))):
                continue
            inputs.append(
                InputExample(
                    key=example["key"],
                    instruction_id_list=example["instruction_id_list"],
                    messages=example.get("messages", [{"role": "user", "content": example['prompt']}]),
                    kwargs=example["kwargs"],
                )
            )
    return inputs


def test_instruction_following_strict(
    inp,
    prompt_to_response,
):
    """Tests response to see if instrutions are followed."""
    response = prompt_to_response[inp.key]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**inp.kwargs[index])

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        key=inp.key,
        instruction_id_list=inp.instruction_id_list,
        messages=inp.messages,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp,
    prompt_to_response,
):
    """Tests response for an upper bound for following instructions."""
    response = prompt_to_response[inp.key]
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**inp.kwargs[index])

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        key=inp.key,
        instruction_id_list=inp.instruction_id_list,
        messages=inp.messages,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def read_prompt_to_response_dict(input_jsonl_filename):
    """Creates dictionary matching prompt and response."""
    return_dict = {}
    with open(input_jsonl_filename, "r") as f:
        for line in f:
            example = json.loads(line)
            return_dict[example["key"]] = example["messages"][-1]["content"]
    return return_dict


def bootstrap_confidence_interval(statistics, width=0.95):
    statistics = sorted(statistics)
    lower_idx = int((1 - width) / 2 * len(statistics))
    upper_idx = int((1 + width) / 2 * len(statistics))
    return statistics[lower_idx], statistics[upper_idx]


def compute_scores(results: Dict[str, List]) -> Dict[str, float]:
    scores = {}
    for version in ["strict", "loose"]:
        prompt_correct = sum(results[f"prompt_{version}"])
        prompt_total = len(results[f"prompt_{version}"])
        prompt_acc = prompt_correct / prompt_total
        scores[f"prompt_{version}"] = prompt_acc
        
        inst_correct = sum([sum(x) for x in results[f"instruction_{version}"]])
        inst_total = sum([len(x) for x in results[f"instruction_{version}"]])
        inst_acc = inst_correct / inst_total
        scores[f"instruction_{version}"] = inst_acc

    return scores


def main(argv):
    if len(argv) > 2:
        raise app.UsageError("Too many command-line arguments.")

    prompt_to_response = read_prompt_to_response_dict(_INPUT_RESPONSE_DATA.value)
    inputs = read_prompt_list(_INPUT_DATA.value)

    results = defaultdict(list)
    for func, version in [
        (test_instruction_following_strict, "strict"),
        (test_instruction_following_loose, "loose"),
    ]:
        outputs = []
        for inp in inputs:
            output = func(inp, prompt_to_response)
            results[f"prompt_{version}"].append(output.follow_all_instructions)
            results[f"instruction_{version}"].append(output.follow_instruction_list)
            outputs.append(output)

    scores = compute_scores(results)

    if _BOOTSTRAP.value > 0:
        scores_list = defaultdict(list)
        for _ in range(_BOOTSTRAP.value):
            sample = {
                k: random.choices(results[k], k=len(results[k]))
                for k in results
            }
            scores_ = compute_scores(sample)
            for k in scores:
                scores_list[k].append(scores_[k])

        outputs = []
        for k in scores:
            lower, upper = bootstrap_confidence_interval(scores_list[k])
            print(f"{k}: {scores[k]:.5f} ({lower:.5f}-{upper:.5f})")
            outputs.append(f"{scores[k]:.5f}")
            outputs.append(f"({lower:.5f}-{upper:.5f})")

        print("\n\ncopypaste:")
        print(",".join(outputs))
    else:
        outputs = []
        for k in scores:
            print(f"{k}: {scores[k]:.5f}")
            outputs.append(f"{scores[k]:.5f}")

        print("\n\ncopypaste:")
        print(",".join(outputs))


if __name__ == "__main__":
    app.run(main)
