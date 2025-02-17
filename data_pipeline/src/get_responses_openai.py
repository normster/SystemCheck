import argparse
import concurrent.futures
import json
import random
import time
import traceback
from collections import deque
from pathlib import Path

import openai
import jsonlines
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_random_exponential,
)
from tqdm import tqdm

from data_types import Example, AssistantResponse
from tools import TOOL_FUNCTIONS, OPENAI_SCHEMAS
# uncomment to use gpt-4o-mini simulated tools, useful for debugging
# from fake_tools import TOOL_FUNCTIONS, OPENAI_SCHEMAS

load_dotenv()


OPENAI_CLIENT = openai.OpenAI()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_assistants", type=int, default=None)
    parser.add_argument("--max_concurrency", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--input_file", type=Path, default="prompts_and_queries.jsonl")
    parser.add_argument("--output_file", type=Path, default="prompts_and_queries_and_responses.jsonl")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


TOOL_PROMPT = """Today's date: {date}

You are an AI assistant with access to several tools: you can search the web, browse webpages, generate images from text prompts, and run Python code. Remember to search the web and visit appropriate web pages for all relevant information needed to fulfill the user's request.

Your name is '{title}'. Your full instructions are as follows:

<instructions>{system}</instructions>

Keep the following guardrail clauses in mind:

<clauses>{clauses}</clauses>"""


@retry(
    retry=retry_if_exception_type(
        (
            openai.RateLimitError,
            openai.InternalServerError,
        )
    ),
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_delay(60),
)
def create_with_retry(*args, **kwargs):
    try:
        return OPENAI_CLIENT.chat.completions.create(*args, **kwargs)
    except openai.ContentFilterFinishReasonError:
        return None

def _print_clauses(elem: Example):
    all_clauses = [f"{i + 1}. {clause}" for i, clause in enumerate(elem.clauses)]
    all_clauses = "\n".join(all_clauses)
    return all_clauses


def get_completion_tools(
    system,
    user,
    model="gpt-4o-mini",
    temperature=0.0,
    top_p=1.0,
    max_tokens=None,
    input_tracker=None,
    output_tracker=None,
    verbose=False,
    max_tool_rounds=5,
    parallel_tool_calls=False
):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    response = create_with_retry(
        model=model,
        messages=messages,
        tools=list(OPENAI_SCHEMAS.values()),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        parallel_tool_calls=parallel_tool_calls
    )

    if response is None:
        return None

    if input_tracker is not None:
        input_tracker.append(response.usage.prompt_tokens)
        output_tracker.append(response.usage.completion_tokens)

    message = response.choices[0].message
    messages.append(message.to_dict())
    
    if verbose and message.content:
        print("Model said:", message.content)
    
    num_tool_calls = 0
    while message.tool_calls and num_tool_calls < max_tool_rounds:
        tool_responses = []
        for call in message.tool_calls:
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                print("ERROR IN JSON DECODE:", call.function.arguments)
                arguments = {}

            if verbose:
                print("Model wants to call tool:", call.function.name, arguments)

            try:
                output = TOOL_FUNCTIONS[call.function.name](**arguments)
            except Exception:
                print("ERROR WHEN CALLING TOOL:", call.function.name, arguments)
                print(traceback.format_exc())
                output = {"error": "Unexpected error"}

            if verbose:
                print("Tool output:", output)

            tool_responses.append({"role": "tool", "content": json.dumps(output), "tool_call_id": call.id})

        if len(tool_responses) == 0:
            return messages

        messages.extend(tool_responses)

        response = create_with_retry(
            model=model,
            messages=messages,
            tools=list(OPENAI_SCHEMAS.values()),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls
        )

        if response is None:
            return messages

        message = response.choices[0].message
        messages.append(message.to_dict())
        
        if verbose and message.content:
            print("Model said:", message.content)

        if input_tracker is not None:
            input_tracker.append(response.usage.prompt_tokens)
            output_tracker.append(response.usage.completion_tokens)

        num_tool_calls += 1

    return messages


def generate_responses(
    args: argparse.Namespace,
    elem: Example,
    input_tracker: deque,
    output_tracker: deque,
):
    for user_message in elem.user_messages:
        system = TOOL_PROMPT.format(
            date=time.strftime("%Y-%m-%d"),
            title=elem.title,
            system=elem.instructions,
            clauses=_print_clauses(elem),
        )

        try:
            messages = get_completion_tools(
                system=system,
                user=user_message.content,
                model=args.model,
                temperature=args.temperature,
                input_tracker=input_tracker,
                output_tracker=output_tracker,
                verbose=args.debug,
            )
        except Exception:
            print("ERROR IN get_completion_tools:", elem.id, user_message.content)
            print(traceback.format_exc())
            continue

        if messages is None:
            continue

        asst_response = AssistantResponse(
            messages=messages,
            score=1,
        )
        user_message.responses.append(asst_response)


def main():
    args = parse_args()

    with jsonlines.open(args.input_file) as reader:
        data = list(reader)

    if args.seed is not None:
        random.seed(args.seed)
        random.shuffle(data)

    if args.num_assistants is not None:
        data = data[: args.num_assistants]

    data_models = [Example(**elem) for elem in data]

    INPUT_TRACKER = deque()
    OUTPUT_TRACKER = deque()

    if args.debug:
        for elem in tqdm(data_models):
            generate_responses(args, elem, INPUT_TRACKER, OUTPUT_TRACKER)

    else:
        with concurrent.futures.ThreadPoolExecutor(args.max_concurrency) as executor:
            threads = []
            for elem in data_models:
                t = executor.submit(
                    generate_responses, args, elem, INPUT_TRACKER, OUTPUT_TRACKER
                )
                threads.append(t)

            try:
                for t in tqdm(
                    concurrent.futures.as_completed(threads), total=len(threads)
                ):
                    t.result()
            except KeyboardInterrupt:
                executor._threads.clear()
                concurrent.futures.thread._threads_queues.clear()
                print(
                    "Keyboard interrupt received. Shutting down and saving results..."
                )

    print("Input tokens:", sum(INPUT_TRACKER))
    print("Output tokens:", sum(OUTPUT_TRACKER))

    for e, m in zip(data, data_models):
        e |= m.model_dump()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(args.output_file, "w") as writer:
        writer.write_all(data)


if __name__ == "__main__":
    main()
