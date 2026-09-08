import asyncio
import json
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Tools that change something real - require human approval before running
WRITE_TOOLS = {"star_email", "archive_email", "create_draft"}

server_params = StdioServerParameters(
    command="uv",
    args=["run", "gmail_mcp_server.py"],
)


def mcp_tool_to_openai_format(tool):
    """Convert an MCP tool definition into the format Groq/OpenAI expects."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


async def run_agent(task: str):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            tools = [mcp_tool_to_openai_format(t) for t in mcp_tools.tools]

            messages = [{"role": "user", "content": task}]

            max_turns = 8
            for turn in range(max_turns):
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    max_tokens=500,
                    messages=messages,
                    tools=tools,
                )
                msg = response.choices[0].message

                if not msg.tool_calls:
                    print(f"\nFinal answer:\n{msg.content}")
                    return

                messages.append(msg)

                for tool_call in msg.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    print(f"\n[Agent wants to call: {name}({args})]")

                    if name in WRITE_TOOLS:
                        approve = input("This is a write action. Approve? (y/n): ")
                        if approve.lower() != "y":
                            result_text = "User did not approve this action."
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result_text,
                            })
                            continue

                    result = await session.call_tool(name, args)
                    result_text = result.content[0].text if result.content else "No result"
                    print(f"[Result: {result_text[:200]}]")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    })

            print("\nStopped: reached max turns without a final answer.")


if __name__ == "__main__":
    task = input("What should the agent do? ")
    asyncio.run(run_agent(task))