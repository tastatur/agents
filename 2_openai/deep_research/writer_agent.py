from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from agents import Agent, OpenAIChatCompletionsModel
from dotenv import load_dotenv
import os

load_dotenv(override=True)

INSTRUCTIONS = """
You are a senior researcher tasked with writing a cohesive report for a research query.
You will be provided with the original query, and some research.
Generate a comprehensive report based on the research and the query.
The final output should be in markdown format, and it should be lengthy and detailed. Aim 
for 1-2 pages of content, around 300-500 words, not more than 500 words.
"""


class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final report")
    follow_up_questions: list[str] = Field(description="Suggested topics to research further")


ollama_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # api_key is required by the client, ignored by Ollama
ollama_model = OpenAIChatCompletionsModel(model="mimo-v2.6:9b", openai_client=ollama_client)

writer_agent = Agent(name="Writer Agent", instructions=INSTRUCTIONS, model=ollama_model, output_type=ReportData)
