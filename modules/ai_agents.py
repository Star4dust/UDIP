"""
UDIP v3 — modules/ai_agents.py
Advanced AI Agent System with specialized roles and unified memory.
"""

import streamlit as st
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from modules.ai_engine import ask_groq, update_ai_memory
from modules.core import log, load_config, save_config

class AIAgent:
    """Base class for AI agents with specialized roles."""

    def __init__(self, name: str, role: str, expertise: str, system_prompt: str):
        from modules.core import DATA_DIR
        self.name = name
        self.role = role
        self.expertise = expertise
        self.system_prompt = system_prompt
        self.memory_file = DATA_DIR / f"{name.lower().replace(' ', '_')}_memory.json"
        self.memory = self._load_memory()

    def _load_memory(self) -> List[Dict]:
        """Load agent-specific memory."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            log.error(f"Failed to load {self.name} memory: {e}")
        return []

    def _save_memory(self):
        """Save agent memory to file."""
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2, default=str)
        except Exception as e:
            log.error(f"Failed to save {self.name} memory: {e}")

    def add_to_memory(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add interaction to memory."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "agent": self.name
        }
        if metadata:
            entry.update(metadata)
        self.memory.append(entry)
        self._save_memory()

        # Also update global AI memory
        update_ai_memory(f"{self.name} interaction", content)

    def get_context(self, limit: int = 10) -> str:
        """Get recent context from memory."""
        recent = self.memory[-limit:] if self.memory else []
        context = f"\n{self.name} Memory:\n"
        for entry in recent:
            context += f"[{entry['timestamp']}] {entry['role']}: {entry['content'][:200]}...\n"
        return context

    def ask(
        self,
        user_query: str,
        context: str = "",
        extra_context: str = "",
        groq_key: str | None = None,
        selected_dataset_id: str | None = None,
        task_mode: str = "auto",
        response_style: str = "balanced",
        focus_notes: str = "",
        allow_web_search: bool = True,
    ) -> str:
        """Ask the agent a question with full context."""
        task_mode = (task_mode or "auto").strip().lower()
        response_style = (response_style or "balanced").strip().lower()
        focus_notes = (focus_notes or "").strip()

        mode_guidance = {
            "plan": "Start with a short plan, then list the recommended next steps in order.",
            "build": "Return runnable code, implementation steps, or a concrete artifact the user can use immediately.",
            "debug": "Focus on root cause, likely fix, and how to verify the fix.",
            "research": "Synthesize the best available information clearly and cite sources when web context is present.",
            "rewrite": "Rewrite the answer for clarity, structure, and polish without changing the meaning.",
            "review": "Act as a reviewer. Inspect the draft for correctness, missing assumptions, and risks before answering.",
            "explain": "Explain clearly and simply, with examples if useful.",
            "auto": "Choose the best response format for the user's request.",
        }
        style_guidance = {
            "concise": "Keep the answer tight and high signal.",
            "balanced": "Be complete but not verbose.",
            "deep": "Go deeper, include edge cases, and explain tradeoffs.",
        }

        directive_block = [
            "TASK DIRECTIVES:",
            f"- Mode: {task_mode}",
            f"- Style: {response_style}",
            f"- Mode guidance: {mode_guidance.get(task_mode, mode_guidance['auto'])}",
            f"- Style guidance: {style_guidance.get(response_style, style_guidance['balanced'])}",
        ]
        if focus_notes:
            directive_block.append(f"- Focus notes: {focus_notes}")

        context_sections = [context.strip(), extra_context.strip(), self.get_context().strip()]
        merged_context = "\n\n".join(section for section in context_sections if section)

        full_prompt = f"""{self.system_prompt}

{chr(10).join(directive_block)}

CONTEXT INFORMATION:
{merged_context}

USER QUERY: {user_query}

Please provide a detailed, expert response as {self.name}."""

        groq_key = groq_key or load_config().get("groq_key", "")
        try:
            response, _, _ = ask_groq(
                full_prompt,
                groq_key,
                context=merged_context,
                selected_dataset_id=selected_dataset_id,
                allow_web_search=allow_web_search,
            )
            self.add_to_memory("user", user_query)
            self.add_to_memory("assistant", response, {"query": user_query})
            return response
        except Exception as e:
            err_msg = str(e)
            log.error(f"{self.name} error: {err_msg}")
            if "invalid_api_key" in err_msg.lower():
                return "❌ Invalid Groq API Key. Please check your Settings."
            if "context_length_exceeded" in err_msg.lower():
                return "📏 The dataset is too complex for the current AI model. Try selecting a smaller subset of columns."
            return f"I apologize, but I'm experiencing technical difficulties: {err_msg[:100]}..."

# Specialized Agent Classes
class DataScientistAgent(AIAgent):
    def __init__(self):
        super().__init__(
            name="Dr. Data",
            role="Senior Data Scientist",
            expertise="Statistical analysis, machine learning, data visualization",
            system_prompt="""You are Dr. Data, a data scientist. Provide clear, actionable insights on data analysis, statistics, and ML. Keep responses concise and practical."""
        )

class CodeArchitectAgent(AIAgent):
    def __init__(self):
        super().__init__(
            name="Code Architect",
            role="Software Architect",
            expertise="System design, code optimization, best practices",
            system_prompt="""You are Code Architect, a software engineer. Focus on clean code, optimization, and best practices. Provide practical coding solutions."""
        )

class BusinessAnalystAgent(AIAgent):
    def __init__(self):
        super().__init__(
            name="Biz Analyst",
            role="Business Intelligence Analyst",
            expertise="Business metrics, KPI analysis, strategic insights",
            system_prompt="""You are Biz Analyst, a business analyst. Translate data into business insights, KPIs, and strategic recommendations."""
        )

class MLResearcherAgent(AIAgent):
    def __init__(self):
        super().__init__(
            name="ML Researcher",
            role="Machine Learning Researcher",
            expertise="Advanced ML algorithms, research, cutting-edge techniques",
            system_prompt="""You are ML Researcher, an ML expert. Explain advanced algorithms and provide cutting-edge ML insights."""
        )

class QASentinelAgent(AIAgent):
    def __init__(self):
        super().__init__(
            name="QA Sentinel",
            role="Quality Reviewer",
            expertise="Accuracy checking, risk detection, clarity review, code and response auditing",
            system_prompt="""You are QA Sentinel, UDIP's quality reviewer. Inspect a draft answer for factual accuracy, missing assumptions, unsafe code, unclear wording, and hallucinated dataset columns. Be concise, practical, and explicit about any fixes needed."""
        )

class UdipOrchestratorAgent(AIAgent):
    def __init__(self):
        super().__init__(
            name="UDIP Orchestrator",
            role="Multi-Agent Orchestrator",
            expertise="Planning, delegation, synthesis, self-review, and end-to-end task completion",
            system_prompt="""You are UDIP Orchestrator, the most capable agent inside UDIP.

You combine the best habits of a senior planner, coder, reviewer, tutor, and research assistant:
- Planner: break complex requests into a short internal plan before answering.
- Builder: provide exact, runnable code or precise implementation steps when code is needed.
- Reviewer: check your own answer for mistakes, missing assumptions, and unsafe claims.
- Tutor: explain clearly and adapt to the user's skill level.
- Researcher: if the question needs current or external knowledge, use the available web context naturally.

Operating rules:
- Treat the selected dataset as primary when dataset context is present.
- Never invent dataset IDs, column names, metrics, or results.
- If the request is ambiguous and a wrong answer would be risky, ask one precise clarifying question.
- For simple tasks, be concise and direct.
- For complex tasks, structure the answer with a recommendation, rationale, and next steps.
- When writing code, include imports, use exact column names, and keep the solution ready to paste into UDIP.
- Before finalizing, perform a brief self-check for correctness, safety, and completeness.
- Do not reveal hidden reasoning. Provide the polished answer only.
"""
        )

class DataEngineerAgent(AIAgent):
    def __init__(self):
        super().__init__(
            name="Data Engineer",
            role="Data Infrastructure Engineer",
            expertise="ETL pipelines, databases, big data systems",
            system_prompt="""You are Data Engineer, an infrastructure expert. Design data pipelines, databases, and scalable systems."""
        )

# Agent Registry
AGENTS = {
    "udip_orchestrator": UdipOrchestratorAgent(),
    "qa_sentinel": QASentinelAgent(),
    "data_scientist": DataScientistAgent(),
    "code_architect": CodeArchitectAgent(),
    "business_analyst": BusinessAnalystAgent(),
    "ml_researcher": MLResearcherAgent(),
    "data_engineer": DataEngineerAgent(),
}

def get_agent(agent_name: str) -> Optional[AIAgent]:
    """Get an agent by name."""
    return AGENTS.get(agent_name.lower().replace(" ", "_"))

def get_all_agents() -> Dict[str, AIAgent]:
    """Get all available agents."""
    return AGENTS

def get_agent_for_task(task_description: str) -> str:
    """Determine the best agent for a given task."""
    task_lower = task_description.lower()

    if any(word in task_lower for word in ["orchestrator", "agent", "assistant", "plan", "workflow", "roadmap", "architecture", "strategy", "end-to-end", "end to end"]):
        return "udip_orchestrator"
    if any(word in task_lower for word in ["review", "audit", "check", "validate", "critique", "quality"]):
        return "qa_sentinel"
    if any(word in task_lower for word in ["statistics", "analysis", "visualization", "correlation", "regression"]):
        return "data_scientist"
    elif any(word in task_lower for word in ["code", "programming", "function", "class", "optimization", "refactor"]):
        return "code_architect"
    elif any(word in task_lower for word in ["business", "kpi", "metrics", "roi", "strategy", "market"]):
        return "business_analyst"
    elif any(word in task_lower for word in ["neural", "deep learning", "transformer", "research", "advanced ml"]):
        return "ml_researcher"
    elif any(word in task_lower for word in ["database", "etl", "pipeline", "warehouse", "big data"]):
        return "data_engineer"
    else:
        return "udip_orchestrator"  # Default to the most capable generalist

def _build_shared_context(primary_agent: str, selected_dataset_id: str = None) -> str:
    """Build shared context from other agents and the active dataset."""
    context = ""
    for name, ag in AGENTS.items():
        if name != primary_agent:
            context += ag.get_context(3)

    if st.session_state.get("datasets"):
        if selected_dataset_id and selected_dataset_id in st.session_state.datasets:
            selected_ds_info = st.session_state.datasets[selected_dataset_id]
            df = selected_ds_info["df"]
            context += (
                f"PRIMARY DATASET: {selected_ds_info['name']} (ID: {selected_dataset_id})\n"
                f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}\n"
                f"Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}\n\n"
            )
        context += "\nAvailable Datasets:\n"
        for ds_id, ds_info in st.session_state.datasets.items():
            df = ds_info["df"]
            context += f"- {ds_info['name']}: {df.shape[0]} rows, {df.shape[1]} columns\n"
            context += f"  Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}\n"

    return context

def collaborative_response(
    user_query: str,
    primary_agent: str = None,
    selected_dataset_id: str = None,
    deep_review: bool = False,
    task_mode: str = "auto",
    response_style: str = "balanced",
    focus_notes: str = "",
    extra_context: str = "",
    allow_web_search: bool = True,
) -> Dict[str, Any]:
    """Get collaborative response from multiple agents if needed."""
    if not primary_agent:
        primary_agent = get_agent_for_task(user_query)

    agent = get_agent(primary_agent)
    if not agent:
        return {"error": f"Agent {primary_agent} not found"}

    context = _build_shared_context(primary_agent, selected_dataset_id)

    response = agent.ask(
        user_query,
        context=context,
        extra_context=extra_context,
        groq_key=load_config().get("groq_key", ""),
        selected_dataset_id=selected_dataset_id,
        task_mode=task_mode,
        response_style=response_style,
        focus_notes=focus_notes,
        allow_web_search=allow_web_search,
    )

    result = {
        "agent": agent.name,
        "role": agent.role,
        "response": response,
        "expertise": agent.expertise,
    }

    if deep_review:
        reviewer = get_agent("qa_sentinel")
        if reviewer:
            review_prompt = f"""Review the draft answer for the user request below.

User request:
{user_query}

Draft answer:
{response}

Focus on:
- correctness
- missing assumptions
- hallucinated dataset columns or results
- unsafe or brittle code
- clarity and usefulness

Return a concise review with:
Verdict: pass or fix
Confidence: 0-100
Issues: ...
Suggested improvements: ...
"""
            try:
                review = reviewer.ask(
                    review_prompt,
                    context=context,
                    extra_context=extra_context,
                    groq_key=load_config().get("groq_key", ""),
                    selected_dataset_id=selected_dataset_id,
                    task_mode="review",
                    response_style="concise",
                    focus_notes="Focus on correctness, risks, and hallucinations.",
                    allow_web_search=False,
                )
                result["review"] = review
                result["reviewer"] = reviewer.name
            except Exception as e:
                result["review_error"] = str(e)

    return result
