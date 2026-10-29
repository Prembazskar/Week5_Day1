
import asyncio
import os

from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load API key from .env
load_dotenv()


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from .env")

    # Create OpenAI model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini"
    )

    try:
        # Create HR Assistant Agent
        hr_agent = AssistantAgent(
            name="hr_assistant",
            model_client=model_client,
            system_message="You are an HR assistant."
        )

        # Define the HR task
        task = (
            "Write a job description for a websphere developer. "
            "Return exactly five numbered lines, one sentence per line. "
            "Include responsibilities,required skills, teamwork, "
            "testing, and qualifications. "
            "Do not add a heading or introduction."
        )

        # Execute agent and display response
        await Console(hr_agent.run_stream(task=task))

    finally:
        # Close model client
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
