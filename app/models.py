from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from typing import Dict, Any, AsyncGenerator, List
import asyncio

class BaseModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        raise NotImplementedError
    
    async def generate_stream(self, image: torch.Tensor, question: str, prompt: str) -> AsyncGenerator[str, None]:
        raise NotImplementedError

class Blip2Model(BaseModel):
    def _load_model(self):
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
    
    async def generate_stream(self, image: torch.Tensor, question: str, prompt: str) -> AsyncGenerator[str, None]:
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        
        # Initialize generation
        generated_ids = torch.tensor([[self.processor.tokenizer.bos_token_id]], device=self.device)
        
        # Generate tokens one by one
        for _ in range(100):  # max_length
            outputs = self.model.generate(
                **inputs,
                decoder_input_ids=generated_ids,
                max_length=generated_ids.shape[1] + 1,
                num_beams=5,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            next_token = outputs.sequences[0, -1]
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            token = self.processor.decode(next_token, skip_special_tokens=True)
            if token:  # Only yield non-empty tokens
                yield token
            
            # Check for end of sequence
            if next_token == self.processor.tokenizer.eos_token_id:
                break
            
            await asyncio.sleep(0.01)  # Small delay between tokens

class OFAModel(BaseModel):
    def _load_model(self):
        self.processor = AutoProcessor.from_pretrained("OFA-Sys/OFA-base")
        self.model = AutoModelForCausalLM.from_pretrained(
            "OFA-Sys/OFA-base",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
    
    async def generate_stream(self, image: torch.Tensor, question: str, prompt: str) -> AsyncGenerator[str, None]:
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        
        # Initialize generation
        generated_ids = torch.tensor([[self.processor.tokenizer.bos_token_id]], device=self.device)
        
        # Generate tokens one by one
        for _ in range(100):  # max_length
            outputs = self.model.generate(
                **inputs,
                decoder_input_ids=generated_ids,
                max_length=generated_ids.shape[1] + 1,
                num_beams=5,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            next_token = outputs.sequences[0, -1]
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            token = self.processor.decode(next_token, skip_special_tokens=True)
            if token:  # Only yield non-empty tokens
                yield token
            
            # Check for end of sequence
            if next_token == self.processor.tokenizer.eos_token_id:
                break
            
            await asyncio.sleep(0.01)  # Small delay between tokens

class LLaVAModel(BaseModel):
    def _load_model(self):
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b")
        self.model = AutoModelForCausalLM.from_pretrained(
            "llava-hf/llava-1.5-7b",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
    
    async def generate_stream(self, image: torch.Tensor, question: str, prompt: str) -> AsyncGenerator[str, None]:
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        
        # Initialize generation
        generated_ids = torch.tensor([[self.processor.tokenizer.bos_token_id]], device=self.device)
        
        # Generate tokens one by one
        for _ in range(100):  # max_length
            outputs = self.model.generate(
                **inputs,
                decoder_input_ids=generated_ids,
                max_length=generated_ids.shape[1] + 1,
                num_beams=5,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            next_token = outputs.sequences[0, -1]
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            token = self.processor.decode(next_token, skip_special_tokens=True)
            if token:  # Only yield non-empty tokens
                yield token
            
            # Check for end of sequence
            if next_token == self.processor.tokenizer.eos_token_id:
                break
            
            await asyncio.sleep(0.01)  # Small delay between tokens

class ModelManager:
    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self.available_models = {
            "blip2": Blip2Model,
            "ofa": OFAModel,
            "llava": LLaVAModel
        }
    
    def get_model(self, model_name: str) -> BaseModel:
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not supported")
        
        if model_name not in self.models:
            self.models[model_name] = self.available_models[model_name](model_name)
        
        return self.models[model_name]
    
    def list_available_models(self) -> List[str]:
        return list(self.available_models.keys()) 