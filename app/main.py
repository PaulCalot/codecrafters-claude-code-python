import argparse
import io
import json
import logging
import os
import subprocess
import sys
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCallUnion,
)
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="latin-1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


class ResponseType(Enum):
    MESSAGE_ONLY = "message"
    TOOL_USE = "tool"


class ToolResult(BaseModel):
    role: str = "tool"  # Make it frozen
    tool_call_id: str
    content: str  # answer from the tool


class ToolHandler:
    @classmethod
    def config(cls) -> List[Dict[str, Any]]:
        return [
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
            },
            {
                "type": "function",
                "function": {
                    "name": "Write",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "required": ["file_path", "content"],
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path of the file to write to",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content to write to the file",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Execute a shell command",
                    "parameters": {
                        "type": "object",
                        "required": ["command"],
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to execute",
                            }
                        },
                    },
                },
            },
        ]

    @classmethod
    def apply_tool(cls, name: str, args: Dict[str, Any]) -> str:
        logging.info(f"Calling tool {name} with args {args}.")
        tool_fn = cls.tools_switcher(name)
        res = tool_fn(**args)
        return res

    @classmethod
    def tools_switcher(cls, name: str) -> Callable:
        if name == "Read":
            return cls.read_tool
        elif name == "Write":
            return cls.write_tool
        elif name == "Bash":
            return cls.bash_tool
        else:
            raise RuntimeError(f"Could not find tool {name}")

    @classmethod
    def read_tool(cls, file_path: str) -> str:
        with open(file_path, "r") as f:
            lines = f.readlines()
        return "\n".join(lines)

    @classmethod
    def write_tool(cls, file_path: str, content: str) -> str:
        with open(file_path, "w") as f:
            f.write(content)
        return content

    @classmethod
    def bash_tool(cls, command: str) -> str:
        completed = subprocess.run(command, capture_output=True, text=True, shell=True)
        if completed.returncode == 0:  # success
            return completed.stdout
        else:
            return completed.stderr


class ChoiceHandler:
    @classmethod
    def handle(cls, choice: Choice) -> Tuple[ResponseType, List[ToolResult] | str]:
        first_msg = choice.message
        if (first_msg.tool_calls is not None) and (len(first_msg.tool_calls) > 0):
            tool_results = list(map(cls.call_tool, first_msg.tool_calls))
            return ResponseType.TOOL_USE, tool_results
        else:
            assert first_msg.content is not None
            return ResponseType.MESSAGE_ONLY, first_msg.content

    @classmethod
    def call_tool(cls, tool: ChatCompletionMessageToolCallUnion) -> ToolResult:
        assert isinstance(tool, ChatCompletionMessageFunctionToolCall)
        id = tool.id
        args = json.loads(tool.function.arguments)
        res = ToolHandler.apply_tool(tool.function.name, args)
        return ToolResult(tool_call_id=id, content=res)


def main():
    # p = argparse.ArgumentParser()
    # p.add_argument("-p", required=True)
    # args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    # client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    client = OpenAI()

    first_msg = input("> user: ")
    messages = [{"role": "user", "content": first_msg}]

    while True:
        chat = client.chat.completions.create(
            model="gpt-4.1-nano",
            # model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=ToolHandler.config(),
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        first_choice = chat.choices[0]
        messages.append(first_choice.message.to_dict())

        # You can use print statements as follows for debugging, they'll be visible when running tests.
        response_type, result = ChoiceHandler.handle(first_choice)

        if response_type.value == ResponseType.MESSAGE_ONLY.value:
            print(f"> agent: {result}")
            user_answer = input("> user: ")
            messages.append({"role": "user", "content": user_answer})
        elif response_type.value == ResponseType.TOOL_USE.value:
            for tool_res in result:
                assert isinstance(tool_res, ToolResult)
                messages.append(tool_res.model_dump())

        # if first_choice.finish_reason == "stop":
        #     exit(0)


if __name__ == "__main__":
    main()
