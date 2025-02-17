"""
Simulated versions of tool functions for debugging and testing.
These implementations use GPT-4 to generate fake search results and webpage content rather than making real API calls.

Usage:
    1. Import these instead of real tools for testing:
       from fake_tools import TOOL_FUNCTIONS, OPENAI_SCHEMAS
    
    2. Or uncomment the import in get_responses_openai.py:
       # from fake_tools import TOOL_FUNCTIONS, OPENAI_SCHEMAS
"""

import os
import subprocess
import traceback

import openai
from dotenv import load_dotenv
from tenacity import (
    retry,
    wait_random_exponential,
    retry_if_exception_type,
    stop_after_delay,
)

load_dotenv()

OPENAI_CLIENT = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


OPENAI_SCHEMAS = {
    "search_web": {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search for a text query and view the results page. Use `visit_page` to retrieve full text of a webpage if needed. `search_web` should be used when the user asks for specific information you are unaware or unsure of.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    "visit_page": {
        "type": "function",
        "function": {
            "name": "visit_page",
            "description": "Retrieve the main text of a webpage in markdown format. Direct file downloads (e.g. PDF documents) are not supported. `visit_page` may be used after a web search or whenever is appropriate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The webpage url"},
                },
                "required": ["url"],
            },
        },
    },
    "generate_image": {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image from a text prompt to show to the user. `generate_image` should be used to generate images for the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The image generation prompt",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    "run_python": {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute a Python script and capture its output. Common python libraries are available. `run_python` be used when the user asks for a specific computation or task that can be done with Python. The script must be self-contained and will time out after 30 seconds of execution. Remember to print the final value of your computation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "A fully contained Python script",
                    },
                },
                "required": ["script"],
            },
        },
    },
}


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


def search_web(query, *args, **kwargs):
    prompt = f"Please generate a page of simulated web search results for the following query: {query}. Make it look like a simple search engine results page formatted in Markdown."
    response = create_with_retry(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    results = response.choices[0].message.content
    return {"query": query, "results": results}


def visit_page(url, maxlen=10000, *args, **kwargs):
    prompt = f"Please generate a simulated Markdown webpage for the content one might expect to find at the following URL: {url}"
    response = create_with_retry(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    return {"url": url, "content": content}


def generate_image(prompt, *args, **kwargs):
    return {"result": "Success!"}


def run_python(script, timeout=30, *args, **kwargs):
    try:
        result = subprocess.run(
            ["python3", "-c", script],
            text=True,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = result.stdout
    except Exception:
        output = traceback.format_exc()

    return {"output": output}


TOOL_FUNCTIONS = {
    "search_web": search_web,
    "visit_page": visit_page,
    "generate_image": generate_image,
    "run_python": run_python,
}
