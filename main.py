from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import (
    input_guardrail,
    RunContextWrapper,
    TResponseInputItem,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered
)
import chainlit as cl

# Load environment variables
load_dotenv()
set_tracing_disabled(disabled=True)

# Load Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Setup OpenAI-compatible Gemini client
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/"  # 👈 correct base URL
)

# Initialize model with Gemini
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",  # Or your preferred model
    openai_client=client
)

# Output schema for guardrail
class OutputPython(BaseModel):
    is_python_related: bool
    reasoning: str

# Guardrail agent to check if input is python-related
input_guardrails_agent = Agent(
    name="Input Guardrail Checker",
    instructions="Check if the user's question is related to Python programming. Return True if yes, otherwise False.",
    model=model,
    output_type=OutputPython
)

# Guardrail function
@input_guardrail
async def input_guardrail_func(
    ctx: RunContextWrapper,
    agent: Agent,
    input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(
        input_guardrails_agent,
        input
    )
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_python_related
    )

# Main Python expert agent
main_agent = Agent(
    name="Python Expert Agent",
    instructions="You are a Python expert. Only respond to Python-related programming questions.",
    model=model,
    input_guardrails=[input_guardrail_func]
)

# When chat starts
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="👋 I'm ready to help with Python-related questions!").send()

# When a message is received
@cl.on_message
async def on_message(message: cl.Message):
    try:
        result = await Runner.run(
            main_agent,
            input=message.content
        )

        # Send the result
        await cl.Message(content=str(result.final_output)).send()

    except InputGuardrailTripwireTriggered:
        await cl.Message(content="🚫 Please try Python-related questions only.").send()
