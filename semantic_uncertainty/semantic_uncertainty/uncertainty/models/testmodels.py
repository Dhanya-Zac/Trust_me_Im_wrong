import os
import logging
import torch
import openai
import requests
from dotenv import load_dotenv

# Import BaseModel and STOP_SEQUENCES from your repo
from semantic_uncertainty.semantic_uncertainty.uncertainty.models.base_model import BaseModel, STOP_SEQUENCES

# Load environment variables
main_dir = os.path.expanduser("~/Trust_me_Im_wrong")
load_dotenv(os.path.join(main_dir, ".env"))


class APIModel(BaseModel):
    """Base class for API-based models."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        if max_new_tokens is None:
            raise ValueError("max_new_tokens must be specified")
        super().__init__(stop_sequences or STOP_SEQUENCES, max_new_tokens)
        self.model_name = model_name

    def predict(self, input_data, temperature=0.5, return_full=False):
        raise NotImplementedError("Subclasses must implement predict method")

    def get_p_true(self, input_data):
        """Approximate probability that model answers 'A' vs 'B'."""
        answer_true, _, _ = self.predict(input_data + " A", temperature=0.1)
        answer_false, _, _ = self.predict(input_data + " B", temperature=0.1)

        # Simple heuristic: compare lengths of outputs (placeholder for log-likelihood)
        score_true = len(answer_true) if answer_true else 0.5
        score_false = len(answer_false) if answer_false else 0.5
        total = score_true + score_false
        return score_true / total if total > 0 else 0.5

    def eval(self):
        # API models are always in eval mode
        return

    def generate(self, *args, **kwargs):
        raise NotImplementedError("API models use predict method instead")

    def output(self, *args, **kwargs):
        raise NotImplementedError("API models use predict method instead")


class GPT4MiniModel(APIModel):
    """GPT-4o Mini API implementation."""

    def __init__(self, model_name="gpt-4o-mini", stop_sequences=None, max_new_tokens=None):
        super().__init__(model_name, stop_sequences, max_new_tokens)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    def predict(self, input_data, temperature=0.5, return_full=False):
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": input_data}],
                max_tokens=self.max_new_tokens,
                temperature=temperature,
                logprobs=5
            )

            answer = response.choices[0].message["content"]

            log_likelihoods = (
                response.choices[0].logprobs.token_logprobs
                if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs
                else [0.0]
            )

            last_token_embedding = torch.randn(768)

            if return_full:
                return answer

            # Handle stop sequences
            sliced_answer = answer
            for stop in self.stop_sequences:
                if stop in answer:
                    sliced_answer = answer.split(stop)[0]
                    break

            return sliced_answer.strip(), log_likelihoods, last_token_embedding.unsqueeze(0)

        except Exception as e:
            logging.error(f"Error in GPT4Mini prediction: {e}")
            return None, None, None


class DeepSeekModel(APIModel):
    """DeepSeek API implementation."""

    def __init__(self, model_name="deepseek-chat", stop_sequences=None, max_new_tokens=None):
        super().__init__(model_name, stop_sequences, max_new_tokens)
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

    def predict(self, input_data, temperature=0.5, return_full=False):
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": input_data}],
                "max_tokens": self.max_new_tokens,
                "temperature": temperature,
            }
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()
            answer = result["choices"][0]["message"]["content"]

            log_likelihoods = [0.0]
            last_token_embedding = torch.randn(768)

            if return_full:
                return answer

            sliced_answer = answer
            for stop in self.stop_sequences:
                if stop in answer:
                    sliced_answer = answer.split(stop)[0]
                    break

            return sliced_answer.strip(), log_likelihoods, last_token_embedding.unsqueeze(0)

        except Exception as e:
            logging.error(f"Error in DeepSeek prediction: {e}")
            return None, None, None


def get_model(model_name, stop_sequences=None, max_new_tokens=None):
    """Factory function to get the appropriate API model."""
    if "gpt" in model_name.lower() or "4o" in model_name.lower():
        return GPT4MiniModel(model_name, stop_sequences, max_new_tokens)
    elif "deepseek" in model_name.lower():
        return DeepSeekModel(model_name, stop_sequences, max_new_tokens)
    else:
        logging.warning(f"Unknown model {model_name}, defaulting to GPT-4o mini")
        return GPT4MiniModel("gpt-4o-mini", stop_sequences, max_new_tokens)
