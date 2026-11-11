"""
UDIP v3 — modules/multi_model_ai.py
Multi-model AI support with cost management and provider flexibility.
"""

import streamlit as st
import requests
import json
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import hashlib


@dataclass
class AIProvider:
    """AI provider configuration."""
    name: str
    api_key_env: str
    base_url: str
    models: List[Dict[str, Any]]
    cost_per_1k_tokens: Dict[str, float]  # input, output costs
    free_tier_limit: Optional[int] = None
    requires_payment: bool = False
    setup_instructions: str = ""


@dataclass
class ModelUsage:
    """Model usage tracking."""
    provider: str
    model: str
    tokens_used: int
    cost_estimate: float
    timestamp: float


class MultiModelAI:
    """Multi-model AI management system."""
    
    def __init__(self):
        if "ai_providers" not in st.session_state:
            st.session_state.ai_providers = self._init_providers()
        if "selected_provider" not in st.session_state:
            st.session_state.selected_provider = "groq"
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = "llama3-70b-8192"
        if "usage_tracking" not in st.session_state:
            st.session_state.usage_tracking = []
        if "cost_settings" not in st.session_state:
            st.session_state.cost_settings = {
                "daily_budget": 10.0,
                "monthly_budget": 100.0,
                "cost_alerts": True,
                "free_tier_priority": True
            }
    
    def _init_providers(self) -> Dict[str, AIProvider]:
        """Initialize all supported AI providers."""
        return {
            "groq": AIProvider(
                name="Groq",
                api_key_env="GROQ_API_KEY",
                base_url="https://api.groq.com/openai/v1",
                models=[
                    {"id": "llama3-70b-8192", "name": "Llama 3 70B", "context": 8192, "speed": "Fast"},
                    {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B", "context": 32768, "speed": "Fast"},
                    {"id": "gemma-7b-it", "name": "Gemma 7B", "context": 8192, "speed": "Very Fast"}
                ],
                cost_per_1k_tokens={"input": 0.0, "output": 0.0},  # Free tier
                free_tier_limit=1000000,  # 1M tokens/month free
                requires_payment=False,
                setup_instructions="Get free API key from https://console.groq.com"
            ),
            "openai": AIProvider(
                name="OpenAI",
                api_key_env="OPENAI_API_KEY",
                base_url="https://api.openai.com/v1",
                models=[
                    {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "context": 128000, "speed": "Fast"},
                    {"id": "gpt-4", "name": "GPT-4", "context": 8192, "speed": "Medium"},
                    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "context": 16384, "speed": "Very Fast"}
                ],
                cost_per_1k_tokens={
                    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
                    "gpt-4": {"input": 0.03, "output": 0.06},
                    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
                },
                free_tier_limit=None,
                requires_payment=True,
                setup_instructions="Get API key from https://platform.openai.com/api-keys"
            ),
            "anthropic": AIProvider(
                name="Anthropic (Claude)",
                api_key_env="ANTHROPIC_API_KEY",
                base_url="https://api.anthropic.com/v1",
                models=[
                    {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "context": 200000, "speed": "Fast"},
                    {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku", "context": 200000, "speed": "Very Fast"},
                    {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "context": 200000, "speed": "Medium"}
                ],
                cost_per_1k_tokens={
                    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
                    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
                    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075}
                },
                free_tier_limit=None,
                requires_payment=True,
                setup_instructions="Get API key from https://console.anthropic.com/"
            ),
            "google": AIProvider(
                name="Google Gemini",
                api_key_env="GOOGLE_API_KEY",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                models=[
                    {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro", "context": 2097152, "speed": "Fast"},
                    {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "context": 1048576, "speed": "Very Fast"},
                    {"id": "gemini-pro", "name": "Gemini Pro", "context": 32768, "speed": "Medium"}
                ],
                cost_per_1k_tokens={
                    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
                    "gemini-1.5-flash": {"input": 0.000075, "output": 0.00015},
                    "gemini-pro": {"input": 0.0005, "output": 0.0015}
                },
                free_tier_limit=15000000,  # 15M tokens/month free
                requires_payment=False,
                setup_instructions="Get API key from https://aistudio.google.com/app/apikey"
            ),
            "local": AIProvider(
                name="Local Models",
                api_key_env="",
                base_url="http://localhost:11434/v1",
                models=[
                    {"id": "llama3-8b", "name": "Llama 3 8B (Local)", "context": 8192, "speed": "Medium"},
                    {"id": "mistral-7b", "name": "Mistral 7B (Local)", "context": 32768, "speed": "Fast"},
                    {"id": "phi3-mini", "name": "Phi-3 Mini (Local)", "context": 4096, "speed": "Very Fast"}
                ],
                cost_per_1k_tokens={"input": 0.0, "output": 0.0},  # Free after setup
                free_tier_limit=None,
                requires_payment=False,
                setup_instructions="Install Ollama: https://ollama.ai/download"
            )
        }
    
    def get_provider(self, provider_name: str) -> Optional[AIProvider]:
        """Get provider by name."""
        return st.session_state.ai_providers.get(provider_name)
    
    def get_available_providers(self) -> List[AIProvider]:
        """Get list of available providers with API keys."""
        available = []
        for name, provider in st.session_state.ai_providers.items():
            if name == "local":
                # Check if local server is running
                try:
                    response = requests.get(f"{provider.base_url}/models", timeout=2)
                    if response.status_code == 200:
                        available.append(provider)
                except:
                    pass
            else:
                # Check if API key is set
                api_key = os.getenv(provider.api_key_env) or st.session_state.get(f"{provider.api_key_env}_key")
                if api_key:
                    available.append(provider)
        
        return available
    
    def render_provider_selection(self):
        """Render provider and model selection interface."""
        st.markdown("### 🤖 AI Model Configuration")
        
        available_providers = self.get_available_providers()
        
        if not available_providers:
            st.warning("⚠️ No AI providers configured. Please set up API keys below.")
            return
        
        # Provider selection
        provider_names = [p.name for p in available_providers]
        provider_name_map = {p.name: name for name, p in st.session_state.ai_providers.items() 
                          for p in available_providers}
        
        selected_provider_name = st.selectbox(
            "Select AI Provider",
            options=provider_names,
            index=provider_names.index(st.session_state.ai_providers[st.session_state.selected_provider].name) 
                   if st.session_state.selected_provider in st.session_state.ai_providers else 0,
            key="ai_provider_select"
        )
        
        # Update selected provider
        selected_provider_key = provider_name_map.get(selected_provider_name)
        if selected_provider_key and selected_provider_key != st.session_state.selected_provider:
            st.session_state.selected_provider = selected_provider_key
            # Reset model selection
            provider = st.session_state.ai_providers[selected_provider_key]
            if provider.models:
                st.session_state.selected_model = provider.models[0]["id"]
        
        # Model selection
        provider = st.session_state.ai_providers[st.session_state.selected_provider]
        if provider.models:
            model_names = [m["name"] for m in provider.models]
            model_id_map = {m["name"]: m["id"] for m in provider.models}
            
            selected_model_name = st.selectbox(
                "Select Model",
                options=model_names,
                index=model_names.index(next(m["name"] for m in provider.models 
                                         if m["id"] == st.session_state.selected_model)),
                key="ai_model_select"
            )
            
            selected_model_id = model_id_map.get(selected_model_name)
            if selected_model_id != st.session_state.selected_model:
                st.session_state.selected_model = selected_model_id
            
            # Display model info
            selected_model = next(m for m in provider.models if m["id"] == selected_model_id)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Context", f"{selected_model['context']:,}")
            with col2:
                st.metric("Speed", selected_model['speed'])
            with col3:
                cost_per_1k = provider.cost_per_1k_tokens.get(selected_model_id, {"input": 0, "output": 0})
                if cost_per_1k["input"] == 0:
                    st.metric("Cost", "Free")
                else:
                    st.metric("Cost/1K", f"${cost_per_1k['input']:.4f}")
    
    def render_api_key_management(self):
        """Render API key management interface."""
        st.markdown("### 🔑 API Key Management")
        
        for name, provider in st.session_state.ai_providers.items():
            if name == "local":
                continue
            
            with st.expander(f"{provider.name} Configuration", expanded=False):
                st.markdown(f"**Setup Instructions:** {provider.setup_instructions}")
                
                # API key input
                current_key = os.getenv(provider.api_key_env) or st.session_state.get(f"{provider.api_key_env}_key", "")
                
                new_key = st.text_input(
                    f"{provider.name} API Key",
                    value=current_key,
                    type="password",
                    key=f"{provider.api_key_env}_key_input",
                    help=f"Enter your {provider.name} API key"
                )
                
                # Save key
                if new_key != current_key:
                    st.session_state[f"{provider.api_key_env}_key"] = new_key
                    # Update environment
                    os.environ[provider.api_key_env] = new_key
                    st.success(f"✅ {provider.name} API key updated!")
                
                # Key status
                if new_key:
                    st.success(f"🔑 {provider.name} API key is configured")
                    
                    # Show cost info
                    if provider.requires_payment:
                        st.warning(f"💳 {provider.name} requires payment - check pricing")
                    else:
                        st.info(f"🆓 {provider.name} has free tier available")
                else:
                    st.error(f"❌ {provider.name} API key not configured")
    
    def render_cost_tracking(self):
        """Render cost tracking and budget management."""
        st.markdown("### 💰 Cost Management")
        
        settings = st.session_state.cost_settings
        
        col1, col2 = st.columns(2)
        with col1:
            settings["daily_budget"] = st.number_input(
                "Daily Budget ($)",
                min_value=0.0,
                max_value=1000.0,
                value=settings["daily_budget"],
                step=1.0
            )
        
        with col2:
            settings["monthly_budget"] = st.number_input(
                "Monthly Budget ($)",
                min_value=0.0,
                max_value=10000.0,
                value=settings["monthly_budget"],
                step=5.0
            )
        
        settings["cost_alerts"] = st.checkbox(
            "Enable Cost Alerts",
            value=settings["cost_alerts"]
        )
        
        settings["free_tier_priority"] = st.checkbox(
            "Prioritize Free Tiers",
            value=settings["free_tier_priority"],
            help="Always use free tier options when available"
        )
        
        st.session_state.cost_settings = settings
        
        # Show usage statistics
        self._render_usage_stats()
    
    def _render_usage_stats(self):
        """Render usage statistics and cost estimates."""
        if not st.session_state.usage_tracking:
            st.info("No usage data available yet.")
            return
        
        # Calculate current month usage
        current_time = time.time()
        month_start = current_time - (30 * 24 * 3600)  # 30 days ago
        
        current_month_usage = [
            u for u in st.session_state.usage_tracking 
            if u.timestamp > month_start
        ]
        
        if current_month_usage:
            total_cost = sum(u.cost_estimate for u in current_month_usage)
            total_tokens = sum(u.tokens_used for u in current_month_usage)
            
            # Usage by provider
            provider_costs = {}
            for usage in current_month_usage:
                if usage.provider not in provider_costs:
                    provider_costs[usage.provider] = 0
                provider_costs[usage.provider] += usage.cost_estimate
            
            st.markdown("#### 📊 Current Month Usage")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Cost", f"${total_cost:.2f}")
            with col2:
                st.metric("Total Tokens", f"{total_tokens:,}")
            with col3:
                daily_avg = total_cost / 30
                st.metric("Daily Avg", f"${daily_avg:.2f}")
            
            # Provider breakdown
            if provider_costs:
                st.markdown("**Cost by Provider:**")
                for provider, cost in provider_costs.items():
                    percentage = (cost / total_cost) * 100 if total_cost > 0 else 0
                    st.markdown(f"- {provider}: ${cost:.2f} ({percentage:.1f}%)")
            
            # Budget alerts
            monthly_budget = st.session_state.cost_settings["monthly_budget"]
            if monthly_budget > 0:
                budget_used = (total_cost / monthly_budget) * 100
                if budget_used > 80:
                    st.error(f"⚠️ {budget_used:.1f}% of monthly budget used!")
                elif budget_used > 60:
                    st.warning(f"⚠️ {budget_used:.1f}% of monthly budget used")
                else:
                    st.success(f"✅ {budget_used:.1f}% of monthly budget used")
    
    def call_ai(
        self, 
        messages: List[Dict[str, str]], 
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, float]:
        """Call AI API and return response with cost estimate."""
        
        provider_name = provider or st.session_state.selected_provider
        model_id = model or st.session_state.selected_model
        
        provider = st.session_state.ai_providers.get(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name} not found")
        
        # Get API key
        if provider_name == "local":
            api_key = ""
        else:
            api_key = os.getenv(provider.api_key_env) or st.session_state.get(f"{provider.api_key_env}_key")
            if not api_key:
                raise ValueError(f"No API key configured for {provider_name}")
        
        # Prepare request
        headers = {"Content-Type": "application/json"}
        if api_key:
            if provider_name == "anthropic":
                headers["x-api-key"] = api_key
            elif provider_name == "google":
                headers["x-goog-api-key"] = api_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        
        # Format request based on provider
        if provider_name == "anthropic":
            # Claude API format
            data = {
                "model": model_id,
                "max_tokens": max_tokens or 4096,
                "temperature": temperature,
                "messages": messages
            }
            endpoint = f"{provider.base_url}/messages"
        elif provider_name == "google":
            # Gemini API format
            data = {
                "contents": [{"parts": [{"text": msg["content"]}] for msg in messages}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens or 2048
                }
            }
            endpoint = f"{provider.base_url}/models/{model_id}:generateContent"
        else:
            # OpenAI-compatible format (Groq, OpenAI, Local)
            data = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096
            }
            endpoint = f"{provider.base_url}/chat/completions"
        
        # Make request
        try:
            response = requests.post(endpoint, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract response and token count
            if provider_name == "anthropic":
                content = result["content"][0]["text"]
                usage = result.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
            elif provider_name == "google":
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                usage = result.get("usageMetadata", {})
                input_tokens = usage.get("promptTokenCount", 0)
                output_tokens = usage.get("candidatesTokenCount", 0)
            else:
                content = result["choices"][0]["message"]["content"]
                usage = result.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
            
            # Calculate cost
            cost_per_1k = provider.cost_per_1k_tokens.get(model_id, {"input": 0, "output": 0})
            cost = (input_tokens / 1000) * cost_per_1k["input"] + \
                   (output_tokens / 1000) * cost_per_1k["output"]
            
            # Track usage
            usage_record = ModelUsage(
                provider=provider_name,
                model=model_id,
                tokens_used=input_tokens + output_tokens,
                cost_estimate=cost,
                timestamp=time.time()
            )
            st.session_state.usage_tracking.append(usage_record)
            
            # Limit usage history
            if len(st.session_state.usage_tracking) > 1000:
                st.session_state.usage_tracking = st.session_state.usage_tracking[-1000:]
            
            return content, cost
            
        except Exception as e:
            st.error(f"AI API call failed: {e}")
            raise e
    
    def get_recommended_model(self, task_type: str = "general") -> str:
        """Get recommended model based on task type and availability."""
        available_providers = self.get_available_providers()
        
        # Prioritize free tiers if enabled
        if st.session_state.cost_settings["free_tier_priority"]:
            free_providers = [p for p in available_providers if not p.requires_payment or p.free_tier_limit]
            if free_providers:
                available_providers = free_providers
        
        # Task-based recommendations
        recommendations = {
            "general": ["llama3-70b-8192", "gpt-4-turbo", "claude-3-5-sonnet-20241022"],
            "analysis": ["gpt-4", "claude-3-opus-20240229", "gemini-1.5-pro"],
            "coding": ["gpt-4-turbo", "claude-3-5-sonnet-20241022", "llama3-70b-8192"],
            "fast": ["gemini-1.5-flash", "claude-3-haiku-20240307", "gpt-3.5-turbo"],
            "local": ["llama3-8b", "mistral-7b", "phi3-mini"]
        }
        
        preferred_models = recommendations.get(task_type, recommendations["general"])
        
        # Find first available preferred model
        for provider in available_providers:
            for model in provider.models:
                if model["id"] in preferred_models:
                    return model["id"]
        
        # Fallback to first available model
        if available_providers:
            return available_providers[0].models[0]["id"]
        
        return "llama3-70b-8192"  # Default fallback


# Global multi-model AI instance
_multi_model_ai = MultiModelAI()


def get_multi_model_ai() -> MultiModelAI:
    """Get the global multi-model AI instance."""
    return _multi_model_ai


def call_ai_with_fallback(
    messages: List[Dict[str, str]],
    task_type: str = "general",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> Tuple[str, float]:
    """Call AI with automatic fallback to available models."""
    
    ai = get_multi_model_ai()
    
    try:
        # Try selected model first
        return ai.call_ai(messages, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        st.warning(f"Primary model failed: {e}. Trying fallback...")
        
        # Get recommended model and retry
        recommended_model = ai.get_recommended_model(task_type)
        try:
            return ai.call_ai(messages, model=recommended_model, temperature=temperature, max_tokens=max_tokens)
        except Exception as e2:
            st.error(f"All AI models failed: {e2}")
            raise e2


def render_ai_configuration():
    """Render complete AI configuration interface."""
    ai = get_multi_model_ai()
    
    ai.render_provider_selection()
    st.divider()
    ai.render_api_key_management()
    st.divider()
    ai.render_cost_tracking()
