import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

LOG = logging.getLogger(__name__)

class Estimator:
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name', 'distilgpt2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None

    def load(self, model_path=""):
        if not model_path:
            model_path = os.environ.get("MODEL_URL")
        LOG.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        LOG.info("Model loaded successfully.")

    def predict(self, data, **kwargs):
        text = data[0] if isinstance(data, (list, tuple)) else data
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=50)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result 