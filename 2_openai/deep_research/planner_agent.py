from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from agents import Agent, OpenAIChatCompletionsModel
import os
from dotenv import load_dotenv
load_dotenv(override=True)

HOW_MANY_SEARCHES = int(os.getenv("HOW_MANY_SEARCHES", 5))


INSTRUCTIONS = f"""
You are a research assistant. Given a user query, come up with a set of web searches
to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms to query for.
"""

class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")

ollama_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # api_key is required by the client, ignored by Ollama
ollama_model = OpenAIChatCompletionsModel(model="mimo-v2.6:9b", openai_client=ollama_client)

planner_agent = Agent(name="Planner Agent", instructions=INSTRUCTIONS, model=ollama_model, output_type=WebSearchPlan)