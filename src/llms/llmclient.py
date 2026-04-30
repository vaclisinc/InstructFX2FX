# Setup LLM client
import json
import os
import openai
import getpass
from dotenv import load_dotenv
from prompts.prompt import Prompt
from utilities.text_processing import extract_json
# from anthropic import Anthropic
from dataclasses import dataclass

@dataclass
class LLMClient:
    model="gpt-4o"
    parameter_max_tokens=(1024, 2048, 4096)

    """LLM client for interacting with OpenRouter or Anthropic."""
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            self.api_key = getpass.getpass("Enter your OpenRouter API key: ")

        self.llm = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        # self.llm = Anthropic(api_key=self.api_key)
        print("✓ LLM client ready")


    def generate_parameters(self, prompt: Prompt, model: str | None = None) -> dict:
        """Generate parameters from the LLM based on the given prompt."""
        last_error = None
        for attempt, max_tokens in enumerate(self.parameter_max_tokens, start=1):
            response = self.llm.chat.completions.create(
                model=model or self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": prompt.sys_prompt},
                    {"role": "user", "content": prompt.instruction}]
            )

            print(
                "LLM generated parameters based on the following instruction: ",
                prompt.instruction,
                f"(attempt {attempt}, max_tokens={max_tokens})",
            )

            text = response.choices[0].message.content
            if not text:
                last_error = ValueError("Empty response from LLM")
                continue

            try:
                return extract_json(text)
            except ValueError as e:
                last_error = e
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                if finish_reason != "length" and attempt == len(self.parameter_max_tokens):
                    break

        raise ValueError(f"Invalid JSON from LLM after retries: {last_error}")
