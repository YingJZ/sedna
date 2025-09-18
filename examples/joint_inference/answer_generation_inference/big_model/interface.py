import logging
import os
from abc import ABC, abstractmethod
from typing import List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests
import json

LOG = logging.getLogger(__name__)


class BaseLLM(ABC):
    
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name', os.environ.get('MODEL_NAME', 'gpt2-xl'))

    @abstractmethod
    def load(self, model_url: str = "") -> None:
        pass
    
    @abstractmethod
    def predict(self, data: Any, **kwargs) -> List[str]:
        pass


class HuggingFaceLLM(BaseLLM):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
    
    def load(self, model_url: str = "") -> None:
        model_path = model_url if model_url else os.environ.get("MODEL_URL", self.model_name)
        
        LOG.info(f"Loading HuggingFace model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        LOG.info("HuggingFace model loaded successfully.")
    
    def predict(self, data: Any, **kwargs) -> List[str]:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model and tokenizer must be loaded before prediction")
        
        text = data[0] if isinstance(data, (list, tuple)) else data
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **kwargs)
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [result]


class APIBasedLLM(BaseLLM):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_url = None
        self.api_key = kwargs.get('api_key', os.environ.get('API_KEY', ''))
    
    def load(self, model_url: str = "") -> None:
        model_path = model_url if model_url else os.environ.get("MODEL_URL", self.model_name)
        
        self.api_url = model_path if model_path.startswith(('http://', 'https://')) else f"http://{model_path}"
        LOG.info(f"API mode: Using endpoint {self.api_url}")
    
    def predict(self, data: Any, **kwargs) -> List[str]:
        if self.api_url is None:
            raise RuntimeError("API URL must be set before prediction")
        
        text = data[0] if isinstance(data, (list, tuple)) else data
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": text}
            ],
            "stream": False
        }
        
        generation_params = ['max_tokens', 'temperature', 'top_p', 'frequency_penalty', 'presence_penalty']
        for param in generation_params:
            if param in kwargs:
                payload[param] = kwargs[param]
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                generated_text = result['choices'][0]['message']['content']
                return [generated_text]
            else:
                LOG.error(f"Unexpected API response format: {result}")
                return ["Error: Invalid API response format"]
                
        except requests.exceptions.RequestException as e:
            LOG.error(f"API request failed: {e}")
            return [f"Error: API request failed - {str(e)}"]
        except json.JSONDecodeError as e:
            LOG.error(f"Failed to parse API response: {e}")
            return ["Error: Failed to parse API response"]
        except Exception as e:
            LOG.error(f"Unexpected error during API call: {e}")
            return [f"Error: {str(e)}"]


class Estimator:
    
    def __init__(self, **kwargs):
        self.load_mode = os.environ.get("MODEL_LOAD_MODE", "file")
        
        if self.load_mode == "api":
            self._llm = APIBasedLLM(**kwargs)
            LOG.info("Using API-based LLM implementation")
        else:
            self._llm = HuggingFaceLLM(**kwargs)
            LOG.info("Using HuggingFace LLM implementation")
    
    def load(self, model_url: str = "") -> None:
        return self._llm.load(model_url)
    
    def predict(self, data: Any, **kwargs) -> List[str]:
        return self._llm.predict(data, **kwargs) 