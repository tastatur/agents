from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, function_tool
from ddgs import DDGS
from dotenv import load_dotenv
import asyncio
import os

from openai import AsyncOpenAI

load_dotenv(override=True)

INSTRUCTIONS = """
You are a research assistant. Given a search term, you use the web_search tool to search the web for that term and
produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 words.
Capture the main points and be succinct. Reply only with the summary.
"""


@function_tool
async def web_search(query: str) -> str:
    """
    Search the web and return the top results (title, URL and snippet for each).

    Args:
        query: The search term
    """
    def _search():
        return DDGS().text(query, max_results=8)
    try:
        # ddgs is synchronous - run it in a thread so parallel searches don't block each other
        results = await asyncio.to_thread(_search)
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)


settings = ModelSettings(tool_choice="required")
tools = [web_search]

ollama_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # api_key is required by the client, ignored by Ollama
ollama_model = OpenAIChatCompletionsModel(model="qwen3:8b", openai_client=ollama_client)

search_agent = Agent(name="Search Agent", instructions=INSTRUCTIONS, tools=tools, model=ollama_model, model_settings=settings)
