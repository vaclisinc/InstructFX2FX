# Setup LLM client
import json
import os
import openai
import getpass
from dotenv import load_dotenv
from prompts.prompt import Prompt
from utilities.parsing import extract_json
# from anthropic import Anthropic

class LLMClient:
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


    def generate_parameters(self, prompt: Prompt) -> dict:
        """Generate parameters from the LLM based on the given prompt."""

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": prompt.sys_prompt},
                {"role": "user", "content": prompt.instruction}]
        )

        print("LLM generated parameters based on the following instruction: ", prompt.instruction)

        text = response.choices[0].message.content
        if text:
            try:
                return extract_json(text)
            except ValueError as e:
                raise ValueError(f"Invalid JSON from LLM: {e}")
        else:
            return {}
