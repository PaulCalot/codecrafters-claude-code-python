import argparse
import json
import os
import sys
from typing import Callable

from openai import OpenAI
from openai.types.chat import ChatCompletion

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


class ResponseHandler:
    @classmethod
    def handle(cls, response: ChatCompletion):
        first_msg = response.choices[0].message
        if (first_msg.tool_calls is not None) and (len(first_msg.tool_calls) > 0):
            print("Tool called")
            first_tool = first_msg.tool_calls.pop(0)
            name = first_tool.function.name
            args = json.loads(first_tool.function.arguments)
            tool_res = cls.tools_switcher(name)(**args)
            print(tool_res)
        else:
            # msg by defualt
            print("Message")
            print(first_msg.content)

    @classmethod
    def tools_switcher(cls, name: str) -> Callable:
        if name == "Read":
            return cls.read_tool
        else:
            raise RuntimeError(f"Could not find tool {name}")

    @classmethod
    def read_tool(cls, file_path: str) -> str:
        with open(file_path, "r") as f:
            lines = f.readlines()
        return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    # client = OpenAI()

    chat = client.chat.completions.create(
        # model="gpt-4.1-nano",
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": args.p}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read and return the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to read",
                            }
                        },
                        "required": ["file_path"],
                    },
                },
            }
        ],
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)
    ResponseHandler.handle(chat)


if __name__ == "__main__":
    main()
