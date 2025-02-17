import os
import requests
import subprocess
import traceback
from datetime import datetime
from threading import Semaphore

from dotenv import load_dotenv
from scrapfly import ScrapflyClient
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random,
)

load_dotenv()

SCRAPFLY_CLIENT = ScrapflyClient(key=os.getenv("SCRAPFLY_API_KEY"))
SCRAPFLY_URL = "https://api.scrapfly.io/scrape"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

SCRAPFLY_SEMAPHORE = Semaphore(5)
BRAVE_SEMAPHORE = Semaphore(20)

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
        }
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
        }
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
        }
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
            }
        },
    },
}

@retry(
    wait=wait_random(0, 1),
    stop=stop_after_attempt(3),
)
def search_web(query, *args, **kwargs):
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": os.getenv("BRAVE_API_KEY"),
    }

    with BRAVE_SEMAPHORE:
        data = requests.get(BRAVE_URL, headers=headers, params={"q": query}).json()

    output_lines = []
    if "news" in data:
        output_lines.append("News Results:")
        output_lines.append("==========")
        for i, result in enumerate(data["news"]["results"]):
            line = f"{i+1}. {result['title']}\n"
            if "page_age" in result:
                date = datetime.fromisoformat(result["page_age"]).strftime("%B %d, %Y")
                line += f"{date}\n"
            line += f"URL: {result['url']}\n\n{result['description']}"
            output_lines.append(line)
            output_lines.append("==========")
        output_lines.append("")

    if "web" in data:
        output_lines.append("Web Results:")
        output_lines.append("==========")
        for i, result in enumerate(data["web"]["results"]):
            line = f"{i+1}. {result['title']}\n"
            if "page_age" in result:
                date = datetime.fromisoformat(result["page_age"]).strftime("%B %d, %Y")
                line += f"{date}\n"
            line += f"URL: {result['url']}\n\n{result['description']}"
            output_lines.append(line)
            output_lines.append("==========")

    if len(output_lines) == 0:
        output_lines = ["No search results found."]

    output = "\n".join(output_lines)
    return {"query": query, "results": output}


@retry(
    wait=wait_random(0, 1),
    stop=stop_after_attempt(3),
)
def visit_page(url, maxlen=10000, *args, **kwargs):
    params = {
        "key": os.getenv("SCRAPFLY_API_KEY"),
        "lang": "en",
        "country": "us",
        "proxy_pool": "public_residential_pool",
        "format": "markdown",
        "asp": True,
        "cache": True,
        "url": url,
    }
    with SCRAPFLY_SEMAPHORE:
        response = requests.get(SCRAPFLY_URL, params=params).json()

    if response.get("error", None):
        message = "API error: " + response.get("message", "unknown error")
        print("ERROR WHEN VISITING URL:", url)
        print(message)
        return {"error": message}

    if result := response.get("result", None):
        if error := result.get("error", None):
            message = "Website error: " + error.get("message", "unknown error")
            print("ERROR WHEN VISITING URL:", url)
            print(message)
            return {"error": message}

        if content := result.get("content", None):
            if len(content) > maxlen:
                content = (
                    "**Webpage content has been truncated**\n\n"
                    + content[:maxlen]
                    + "[..]"
                )

            return {"url": url, "content": content}

    print("UNEXPECTED ERROR WHEN VISITING URL:", url)
    return {"error": "Unexpected error"}


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
