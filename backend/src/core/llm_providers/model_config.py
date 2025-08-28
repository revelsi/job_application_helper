"""
Copyright 2024 Job Application Helper Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Model Configuration System for LLM Providers.

This module defines comprehensive model configurations including capabilities,
token limits, reasoning support, and implementation details for each model.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class ModelType(Enum):
    """Types of models based on their capabilities."""
    STANDARD = "standard"
    REASONING = "reasoning"
    FUNCTION_CALLING = "function_calling"
    MULTIMODAL = "multimodal"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    
    # Basic info
    name: str
    display_name: str
    provider: str
    
    # Capabilities
    model_type: ModelType
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_reasoning: bool = False
    
    # Token limits
    max_input_tokens: int = 4096
    max_output_tokens: int = 2048
    max_total_tokens: int = 8192
    
    # Performance & Cost
    cost_per_1k_input_tokens: Optional[float] = None
    cost_per_1k_output_tokens: Optional[float] = None
    rate_limit_rpm: int = 60  # Requests per minute
    
    # Implementation details
    reasoning_format: Optional[str] = None  # "structured", "thinking_tags", etc.
    api_endpoint_override: Optional[str] = None
    special_parameters: Optional[Dict] = None
    
    # Display information
    description: Optional[str] = None  # Human-readable description for UI


# Model configurations for each provider
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    
    # OpenAI Models
    "gpt-5-mini": ModelConfig(
        name="gpt-5-mini",
        display_name="GPT-5 Mini",
        provider="openai",
        model_type=ModelType.REASONING,
        supports_streaming=True,
        supports_function_calling=True,
        supports_reasoning=True,
        max_input_tokens=128000,
        max_output_tokens=16384,
        max_total_tokens=144384,
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0006,
        rate_limit_rpm=1000,
        reasoning_format="structured",
        description="GPT-5 Mini with reasoning capabilities and 128K context",
    ),
    
    # Mistral Models
    "mistral-small-latest": ModelConfig(
        name="mistral-small-latest",
        display_name="Mistral Small Latest",
        provider="mistral",
        model_type=ModelType.STANDARD,
        supports_streaming=True,
        supports_function_calling=True,
        supports_reasoning=False,
        max_input_tokens=128000,
        max_output_tokens=8192,
        max_total_tokens=136192,
        cost_per_1k_input_tokens=0.0002,
        cost_per_1k_output_tokens=0.0006,
        rate_limit_rpm=60,
        description="Mistral Small - Advanced language model with function calling",
    ),

    "mistral-medium-latest": ModelConfig(
        name="mistral-medium-latest",
        display_name="Mistral Medium Latest",
        provider="mistral",
        model_type=ModelType.STANDARD,
        supports_streaming=True,
        supports_function_calling=True,
        supports_reasoning=False,
        max_input_tokens=128000,
        max_output_tokens=8192,
        max_total_tokens=136192,
        cost_per_1k_input_tokens=0.0005,
        cost_per_1k_output_tokens=0.0015,
        rate_limit_rpm=60,
        description="Mistral Medium - High-performance language model with function calling",
    ),
    
    # Novita Models
    "openai/gpt-oss-20b": ModelConfig(
        name="openai/gpt-oss-20b",
        display_name="GPT-OSS-20B",
        provider="novita",
        model_type=ModelType.STANDARD,
        supports_streaming=True,
        supports_function_calling=False,
        supports_reasoning=False,
        max_input_tokens=32000,
        max_output_tokens=8192,
        max_total_tokens=40192,
        cost_per_1k_input_tokens=0.0001,
        cost_per_1k_output_tokens=0.0002,
        rate_limit_rpm=100,
        description="GPT-OSS-20B - Open-source model via Novita API",
    ),
    
    "qwen/qwen3-32b-fp8": ModelConfig(
        name="qwen/qwen3-32b-fp8",
        display_name="Qwen3-32B-FP8",
        provider="novita",
        model_type=ModelType.STANDARD,
        supports_streaming=True,
        supports_function_calling=True,
        supports_reasoning=False,
        max_input_tokens=128000,
        max_output_tokens=8192,
        max_total_tokens=136192,
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0003,
        rate_limit_rpm=100,
        description="Qwen3-32B-FP8 - High-performance open-source model via Novita",
    ),
    
    "zai-org/glm-4.5": ModelConfig(
        name="zai-org/glm-4.5",
        display_name="GLM-4.5",
        provider="novita",
        model_type=ModelType.REASONING,
        supports_streaming=True,
        supports_function_calling=True,
        supports_reasoning=True,
        max_input_tokens=128000,
        max_output_tokens=8192,
        max_total_tokens=136192,
        cost_per_1k_input_tokens=0.0002,
        cost_per_1k_output_tokens=0.0004,
        rate_limit_rpm=100,
        reasoning_format="structured",
        description="GLM-4.5 - Reasoning model with structured output via Novita",
    ),
    
    # Ollama Models (Local - no cost, variable limits based on hardware)
    "gemma3:1b": ModelConfig(
        name="gemma3:1b",
        display_name="Gemma 3 (1B)",
        provider="ollama",
        model_type=ModelType.STANDARD,
        supports_streaming=True,
        supports_function_calling=False,
        supports_reasoning=False,
        max_input_tokens=32768,  # 32K context window per official docs
        max_output_tokens=8192,
        max_total_tokens=40960,
        cost_per_1k_input_tokens=0.0,  # Local model
        cost_per_1k_output_tokens=0.0,
        rate_limit_rpm=1000,  # Local limit
        description="Gemma 3 (1B) - Local lightweight open-source language model with 32K context",
    ),
    
    "llama3.2:1b": ModelConfig(
        name="llama3.2:1b",
        display_name="Llama 3.2 (1B)",
        provider="ollama",
        model_type=ModelType.STANDARD,
        supports_streaming=True,
        supports_function_calling=False,
        supports_reasoning=False,
        max_input_tokens=131072,  # 128K context window per official docs
        max_output_tokens=8192,
        max_total_tokens=139264,
        cost_per_1k_input_tokens=0.0,  # Local model
        cost_per_1k_output_tokens=0.0,
        rate_limit_rpm=1000,  # Local limit
        description="Llama 3.2 (1B) - Local lightweight open-source language model with 128K context",
    ),
}


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a specific model."""
    return MODEL_CONFIGS.get(model_name)


def get_models_for_provider(provider: str) -> List[ModelConfig]:
    """Get all models for a specific provider."""
    return [config for config in MODEL_CONFIGS.values() if config.provider == provider]


def get_provider_models_dict() -> Dict[str, List[str]]:
    """Get a dictionary mapping providers to their model names."""
    provider_models = {}
    for config in MODEL_CONFIGS.values():
        if config.provider not in provider_models:
            provider_models[config.provider] = []
        provider_models[config.provider].append(config.name)
    return provider_models


def get_model_display_info(model_name: str) -> Dict[str, any]:
    """Get display information for a model."""
    config = get_model_config(model_name)
    if not config:
        return {"name": model_name, "display_name": model_name, "info": "Unknown model"}
    
    info_parts = []
    
    # Add model type
    if config.model_type == ModelType.REASONING:
        info_parts.append("🧠 Reasoning")
    elif config.model_type == ModelType.FUNCTION_CALLING:
        info_parts.append("🔧 Function Calling")
    elif config.model_type == ModelType.MULTIMODAL:
        info_parts.append("🎨 Multimodal")
    
    # Add token info
    input_tokens = f"{config.max_input_tokens//1000}K" if config.max_input_tokens >= 1000 else str(config.max_input_tokens)
    output_tokens = f"{config.max_output_tokens//1000}K" if config.max_output_tokens >= 1000 else str(config.max_output_tokens)
    info_parts.append(f"📝 {input_tokens}→{output_tokens}")
    
    # Add cost info (if not free)
    if config.cost_per_1k_input_tokens and config.cost_per_1k_input_tokens > 0:
        cost = f"${config.cost_per_1k_input_tokens:.4f}/1K"
        info_parts.append(f"💰 {cost}")
    elif config.provider == "ollama":
        info_parts.append("🆓 Local")
    
    return {
        "name": config.name,
        "display_name": config.display_name,
        "info": " • ".join(info_parts),
        "reasoning": config.supports_reasoning,
        "function_calling": config.supports_function_calling,
        "max_input_tokens": config.max_input_tokens,
        "max_output_tokens": config.max_output_tokens,
    }


def validate_model_for_provider(provider: str, model: str) -> bool:
    """Validate that a model belongs to a provider."""
    config = get_model_config(model)
    return config is not None and config.provider == provider


def get_safe_token_limits(model_name: str, request_tokens: Optional[int] = None) -> Dict[str, int]:
    """Get safe token limits for a model, considering request size."""
    config = get_model_config(model_name)
    if not config:
        # Fallback for unknown models
        return {
            "max_input": 4096,
            "max_output": 2048,
            "recommended_output": 1024,
        }
    
    # Leave buffer for system prompts and formatting
    safe_input = int(config.max_input_tokens * 0.9)
    safe_output = min(config.max_output_tokens, int(config.max_total_tokens * 0.3))
    
    # If we know the request size, adjust output accordingly
    if request_tokens:
        remaining_tokens = config.max_total_tokens - request_tokens
        safe_output = min(safe_output, max(512, int(remaining_tokens * 0.8)))
    
    return {
        "max_input": safe_input,
        "max_output": safe_output,
        "recommended_output": min(safe_output, 4096),  # Reasonable default
    }


def get_provider_description(provider: str) -> str:
    """Get a description for a provider listing its available models."""
    models = get_models_for_provider(provider)
    if not models:
        return f"{provider.title()} - No models available"
    
    # Get model display names
    model_names = [model.display_name for model in models]
    
    # Format the description
    if len(model_names) == 1:
        return f"{provider.title()} - {model_names[0]}"
    if len(model_names) == 2:
        return f"{provider.title()} - {model_names[0]} & {model_names[1]}"
    # For more than 2 models, show first 2 and indicate there are more
    return f"{provider.title()} - {model_names[0]}, {model_names[1]} & more"
