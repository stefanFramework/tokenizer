import torch
from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer,BitsAndBytesConfig

from config import TokenizerConfiguration


class PipelineStep:
    def run(self, data) -> dict[str, Any]:
        raise NotImplementedError("Each step must implement a run method")


class LoadTranscript(PipelineStep):
    def run(self, data):
        file_url = TokenizerConfiguration.TRANSCRIPT_FILE_NAME
        with open(file_url, "r", encoding="utf-8") as file:
            transcription = file.read()
        data["transcription"] = transcription
        return data


class PreparePrompt(PipelineStep):
    SYSTEM_MESSAGE = (
        "You are an assistant that produces structured meeting minutes from transcripts. "
        "Format the response using markdown with headings and bullet points. "
        "Include the following sections:\n"
        "- **Attendees** (list names and roles)\n"
        "- **Location and Date** (where and when the meeting took place)\n"
        "- **Discussion Points** (key topics discussed)\n"
        "- **Takeaways** (important conclusions from the meeting)\n"
        "- **Action Items** (list tasks with owners)\n"
    )

    def run(self, data):
        user_prompt = (
            f"Below is a transcript of a council meeting. "
            f"Make a summary and include attendees, location and dates, "
            f"discussion points, takeaways, and action items with owners.\n### Transcript:\n"
            f"{data['transcription']}\n\n####"
        )
        data["messages"] = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": user_prompt}
        ]
        return data


class TokenizeInput(PipelineStep):
    def run(self, data):
        tokenizer = AutoTokenizer.from_pretrained(TokenizerConfiguration.TEXT_MODEL)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer.apply_chat_template(
            data["messages"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        data["tokenizer"] = tokenizer
        data["inputs"] = inputs
        data["prompt_length"] = inputs.shape[1]

        return data


class LoadModel(PipelineStep):
    def run(self, data):
        model = AutoModelForCausalLM.from_pretrained(
            TokenizerConfiguration.TEXT_MODEL,
            trust_remote_code=True,
            device_map="cpu"
        )

        if hasattr(model.config, "sliding_window"):
            model.config.sliding_window = None

        data["model"] = model
        return data


class GenerateResponse(PipelineStep):
    def run(self, data):
        attention_mask = data["inputs"].ne(data["tokenizer"].pad_token_id)

        outputs = data["model"].generate(
            data["inputs"],
            max_new_tokens=TokenizerConfiguration.MAX_AMOUNT_OF_TOKENS,
            attention_mask=attention_mask,
        )

        generated_tokens = outputs[0][data["prompt_length"]:]
        response = data["tokenizer"].decode(generated_tokens, skip_special_tokens=True)

        data["response"] = response
        return data
