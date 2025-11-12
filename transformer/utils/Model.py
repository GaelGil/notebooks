from transformers import FlaxT5ForConditionalGeneration, T5Tokenizer

model_name = "google/mt5-small"  # multilingual T5
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = FlaxT5ForConditionalGeneration.from_pretrained(model_name)


class Model:
    def __init__(
        self,
        model_name: str,
        tokenizer: T5Tokenizer,
        model: FlaxT5ForConditionalGeneration,
    ):
        self.model_name = "google/mt5-small"  # multilingual T5
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = FlaxT5ForConditionalGeneration.from_pretrained(model_name)

    def get_model(self) -> FlaxT5ForConditionalGeneration:
        return self.model
