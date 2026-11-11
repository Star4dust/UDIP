"""
UDIP v3 — modules/ai_maturity_enhancer.py
AI maturity and orchestration hardening utilities.

This module provides enterprise-grade AI capabilities with execution validation,
hallucination safeguards, workflow sanity checks, and confidence scoring.
"""

import streamlit as st
import time
import logging
import json
import hashlib
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


class ExecutionValidationLevel(Enum):
    """Execution validation levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"


class AIConfidenceLevel(Enum):
    """AI confidence levels."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class ExecutionPlan:
    """AI execution plan with validation."""
    plan_id: str
    steps: List[Dict[str, Any]]
    confidence: AIConfidenceLevel
    created_at: float = field(default_factory=time.time)
    validated: bool = False
    execution_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """AI execution result with validation."""
    plan_id: str
    step_id: str
    success: bool
    output: Any
    confidence: AIConfidenceLevel
    execution_time: float
    validated: bool = False
    error_message: Optional[str] = None
    safety_checks: List[Dict[str, bool]] = field(default_factory=list)


class AIMaturityEnhancer:
    """Enhances AI maturity with enterprise-grade capabilities."""
    
    def __init__(self):
        self.execution_plans = {}
        self.execution_history = []
        self.validation_level = ExecutionValidationLevel.MODERATE
        self.confidence_thresholds = {
            AIConfidenceLevel.VERY_HIGH: 0.9,
            AIConfidenceLevel.HIGH: 0.8,
            AIConfidenceLevel.MEDIUM: 0.7,
            AIConfidenceLevel.LOW: 0.6,
            AIConfidenceLevel.VERY_LOW: 0.5
        }
        self.safety_checks = {
            "code_injection": True,
            "data_leakage": True,
            "resource_exhaustion": True,
            "infinite_loop": True,
            "hallucination_detection": True
        }
        self.performance_monitor = {}
        
    def create_execution_plan(self, user_request: str, ai_response: str, 
                         context: Dict[str, Any] = None) -> ExecutionPlan:
        """Create validated execution plan."""
        plan_id = f"plan_{int(time.time())}"
        
        # Parse AI response into steps
        steps = self._parse_ai_response_to_steps(ai_response)
        
        # Validate each step
        validated_steps = []
        for step in steps:
            validated_step = self._validate_execution_step(step, user_request, context)
            validated_steps.append(validated_step)
        
        # Calculate overall confidence
        confidence = self._calculate_plan_confidence(validated_steps)
        
        plan = ExecutionPlan(
            plan_id=plan_id,
            steps=validated_steps,
            confidence=confidence
        )
        
        self.execution_plans[plan_id] = plan
        logger.info(f"Created execution plan {plan_id} with {len(steps)} steps")
        
        return plan
    
    def _parse_ai_response_to_steps(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI response into executable steps."""
        steps = []
        
        # Simple parsing - in production, this would use more sophisticated NLP
        lines = ai_response.strip().split('\n')
        current_step = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect step patterns
            if line.startswith('1.') or line.startswith('2.') or line.startswith('3.'):
                if current_step:
                    steps.append(current_step)
                current_step = {"step": line, "description": line[3:], "type": "action"}
            elif line.startswith('-') or line.startswith('*'):
                current_step["description"] = current_step.get("description", "") + line[1:] + " "
            elif any(keyword in line.lower() for keyword in ['import', 'def', 'class', 'function']):
                current_step["type"] = "code"
                current_step["description"] = current_step.get("description", "") + line + "\n"
            else:
                current_step["description"] = current_step.get("description", "") + line + " "
        
        if current_step:
            steps.append(current_step)
        
        return steps
    
    def _validate_execution_step(self, step: Dict[str, Any], user_request: str, 
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate individual execution step."""
        validated_step = step.copy()
        
        # Add validation metadata
        validated_step["validation"] = {
            "timestamp": time.time(),
            "user_request": user_request,
            "risk_level": self._assess_step_risk(step),
            "safety_checks": self._perform_safety_checks(step)
        }
        
        # Check for common issues
        issues = []
        
        # Check for code injection risks
        if step.get("type") == "code":
            code_content = step.get("description", "")
            if self._check_code_injection_risk(code_content):
                issues.append("Code injection risk detected")
        
        # Check for data leakage
        if "export" in step.get("description", "").lower():
            if self._check_data_leakage_risk(step, context):
                issues.append("Data leakage risk detected")
        
        # Check for infinite loops
        if "while" in step.get("description", "").lower():
            if self._check_infinite_loop_risk(step.get("description", "")):
                issues.append("Infinite loop risk detected")
        
        validated_step["validation"]["issues"] = issues
        validated_step["validation"]["safe"] = len(issues) == 0
        
        return validated_step
    
    def _assess_step_risk(self, step: Dict[str, Any]) -> str:
        """Assess risk level of execution step."""
        step_type = step.get("type", "unknown")
        description = step.get("description", "").lower()
        
        # High-risk patterns
        high_risk_patterns = [
            "delete", "drop", "truncate", "format", "exec", "eval",
            "system", "subprocess", "shell", "sudo", "admin"
        ]
        
        medium_risk_patterns = [
            "modify", "update", "insert", "create", "write",
            "download", "upload", "import", "export"
        ]
        
        for pattern in high_risk_patterns:
            if pattern in description:
                return "high"
        
        for pattern in medium_risk_patterns:
            if pattern in description:
                return "medium"
        
        return "low"
    
    def _check_code_injection_risk(self, code: str) -> bool:
        """Check for code injection risks."""
        dangerous_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
            r'subprocess\.',
            r'os\.system',
            r'shell\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        
        return False
    
    def _check_data_leakage_risk(self, step: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check for data leakage risks."""
        description = step.get("description", "").lower()
        
        # Check for sensitive data patterns
        sensitive_patterns = [
            "password", "token", "key", "secret", "credential",
            "private", "confidential", "internal", "proprietary"
        ]
        
        # Check if sensitive data is being exposed
        for pattern in sensitive_patterns:
            if pattern in description:
                return True
        
        # Check context for sensitive data
        if context:
            context_str = str(context).lower()
            for pattern in sensitive_patterns:
                if pattern in context_str:
                    return True
        
        return False
    
    def _check_infinite_loop_risk(self, description: str) -> bool:
        """Check for infinite loop risks."""
        # Simple heuristic for infinite loops
        loop_patterns = [
            r'while\s+true\s*:',
            r'for\s+.*\s+range\s*\(',
            r'while\s+.*\s+<',
            r'recursion\s+without\s+base\s+case'
        ]
        
        for pattern in loop_patterns:
            if re.search(pattern, description, re.IGNORECASE):
                return True
        
        return False
    
    def _perform_safety_checks(self, step: Dict[str, Any]) -> Dict[str, bool]:
        """Perform comprehensive safety checks."""
        checks = {}
        
        # Code injection check
        if step.get("type") == "code":
            checks["code_injection"] = not self._check_code_injection_risk(
                step.get("description", "")
            )
        
        # Data leakage check
        checks["data_leakage"] = not self._check_data_leakage_risk(
                step, {}
            )
        
        # Infinite loop check
        checks["infinite_loop"] = not self._check_infinite_loop_risk(
                step.get("description", "")
            )
        
        # Resource exhaustion check
        checks["resource_exhaustion"] = self._check_resource_exhaustion_risk(step)
        
        # Hallucination check
        checks["hallucination"] = self._check_hallucination_risk(step)
        
        return checks
    
    def _check_resource_exhaustion_risk(self, step: Dict[str, Any]) -> bool:
        """Check for resource exhaustion risks."""
        description = step.get("description", "").lower()
        
        # Check for resource-intensive operations
        resource_patterns = [
            r'range\s*\(\s*\d+\s*,\s*\d+\s*\)',
            r'for\s+.*\s+in\s+.*:',
            r'while\s+.*:',
            r'list\s*\[\s*.*\s*for\s+.*\s+in\s+.*:',
            r'numpy\.',
            r'pandas\.',
            r'torch\.',
            r'tensorflow\.'
        ]
        
        for pattern in resource_patterns:
            if re.search(pattern, description):
                return True
        
        return False
    
    def _check_hallucination_risk(self, step: Dict[str, Any]) -> bool:
        """Check for hallucination risks."""
        description = step.get("description", "").lower()
        
        # Check for hallucination indicators
        hallucination_patterns = [
            "i think", "i believe", "probably", "might be",
            "seems like", "appears to be", "i'm not sure",
            "fictional", "imaginary", "made up",
            "hypothetical", "theoretically"
        ]
        
        for pattern in hallucination_patterns:
            if pattern in description:
                return True
        
        return False
    
    def _calculate_plan_confidence(self, steps: List[Dict[str, Any]]) -> AIConfidenceLevel:
        """Calculate confidence level for execution plan."""
        if not steps:
            return AIConfidenceLevel.VERY_LOW
        
        # Count risk levels
        risk_counts = {"high": 0, "medium": 0, "low": 0}
        
        for step in steps:
            risk_level = step.get("validation", {}).get("risk_level", "low")
            risk_counts[risk_level] += 1
        
        # Calculate confidence based on risk distribution
        total_steps = len(steps)
        high_risk_ratio = risk_counts["high"] / total_steps if total_steps > 0 else 0
        
        if high_risk_ratio > 0.3:
            return AIConfidenceLevel.VERY_LOW
        elif high_risk_ratio > 0.1:
            return AIConfidenceLevel.LOW
        elif high_risk_ratio > 0.05:
            return AIConfidenceLevel.MEDIUM
        elif high_risk_ratio > 0.01:
            return AIConfidenceLevel.HIGH
        else:
            return AIConfidenceLevel.VERY_HIGH
    
    def execute_plan(self, plan_id: str, context: Dict[str, Any] = None) -> ExecutionResult:
        """Execute validated execution plan."""
        if plan_id not in self.execution_plans:
            logger.error(f"Execution plan not found: {plan_id}")
            return ExecutionResult(
                plan_id=plan_id,
                step_id="",
                success=False,
                output=None,
                confidence=AIConfidenceLevel.VERY_LOW,
                execution_time=0.0,
                validated=False,
                error_message=f"Plan not found: {plan_id}"
            )
        
        plan = self.execution_plans[plan_id]
        
        start_time = time.time()
        executed_steps = []
        
        for i, step in enumerate(plan.steps):
            step_id = f"{plan_id}_step_{i}"
            
            # Pre-execution validation
            if not step.get("validation", {}).get("safe", False):
                logger.warning(f"Skipping unsafe step: {step_id}")
                continue
            
            # Execute step
            step_start_time = time.time()
            try:
                result = self._execute_step(step, context)
                execution_time = time.time() - step_start_time
                
                executed_step = {
                    "step_id": step_id,
                    "success": True,
                    "output": result,
                    "execution_time": execution_time,
                    "timestamp": step_start_time
                }
                
                executed_steps.append(executed_step)
                logger.info(f"Step {step_id} executed successfully in {execution_time:.3f}s")
                
            except Exception as e:
                execution_time = time.time() - step_start_time
                executed_step = {
                    "step_id": step_id,
                    "success": False,
                    "output": None,
                    "execution_time": execution_time,
                    "timestamp": step_start_time,
                    "error_message": str(e)
                }
                
                executed_steps.append(executed_step)
                logger.error(f"Step {step_id} failed: {e}")
        
        total_execution_time = time.time() - start_time
        
        # Update plan status
        plan.validated = True
        plan.execution_history = executed_steps
        
        # Store execution result
        result = ExecutionResult(
            plan_id=plan_id,
            step_id="",
            success=all(step["success"] for step in executed_steps),
            output=[step["output"] for step in executed_steps if step["success"]],
            confidence=plan.confidence,
            execution_time=total_execution_time,
            validated=True
        )
        
        self.execution_history.append(result)
        logger.info(f"Execution plan {plan_id} completed in {total_execution_time:.3f}s")
        
        return result
    
    def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute individual step safely."""
        step_type = step.get("type", "unknown")
        description = step.get("description", "")
        
        # Route to appropriate executor
        if step_type == "code":
            return self._execute_code_step(description, context)
        elif step_type == "action":
            return self._execute_action_step(description, context)
        else:
            logger.warning(f"Unknown step type: {step_type}")
            return None
    
    def _execute_code_step(self, code: str, context: Dict[str, Any]) -> Any:
        """Execute code step safely."""
        # In production, this would use a sandboxed execution environment
        # For now, we'll just validate and return the code
        
        logger.info(f"Executing code step: {code[:50]}...")
        
        # Basic validation
        if not code.strip():
            return None
        
        # Check for dangerous operations
        if self._check_code_injection_risk(code):
            logger.error("Code step blocked due to injection risk")
            return None
        
        # In production, this would execute in a sandbox
        # For now, return the code for review
        return {"code": code, "executed": False, "review_required": True}
    
    def _execute_action_step(self, description: str, context: Dict[str, Any]) -> Any:
        """Execute action step safely."""
        logger.info(f"Executing action step: {description[:50]}...")
        
        # Parse action
        if "import" in description.lower():
            return self._execute_import_action(description, context)
        elif "export" in description.lower():
            return self._execute_export_action(description, context)
        else:
            return {"action": description, "executed": False, "review_required": True}
    
    def _execute_import_action(self, description: str, context: Dict[str, Any]) -> Any:
        """Execute import action safely."""
        # Validate import
        import_match = re.search(r'import\s+(\w+)', description)
        if not import_match:
            return {"error": "Invalid import syntax"}
        
        module_name = import_match.group(1)
        
        # Check against safe imports
        safe_modules = [
            "pandas", "numpy", "matplotlib", "seaborn", "sklearn",
            "plotly", "streamlit", "json", "pathlib", "datetime",
            "re", "logging", "typing", "dataclasses", "enum"
        ]
        
        if module_name not in safe_modules:
            return {"error": f"Module {module_name} not in safe list"}
        
        return {"import": module_name, "safe": True, "executed": False}
    
    def _execute_export_action(self, description: str, context: Dict[str, Any]) -> Any:
        """Execute export action safely."""
        # Validate export
        export_match = re.search(r'export\s+(\w+)', description)
        if not export_match:
            return {"error": "Invalid export syntax"}
        
        export_type = export_match.group(1)
        
        # Check for sensitive data
        if self._check_data_leakage_risk({"description": description}, context):
            return {"error": "Export blocked due to data leakage risk"}
        
        return {"export": export_type, "safe": True, "executed": False}
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.execution_history
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            "total_plans": len(self.execution_plans),
            "total_executions": len(self.execution_history),
            "validation_level": self.validation_level.value,
            "safety_checks_enabled": all(self.safety_checks.values()),
            "confidence_thresholds": self.confidence_thresholds
        }
    
    def set_validation_level(self, level: ExecutionValidationLevel):
        """Set execution validation level."""
        self.validation_level = level
        logger.info(f"Validation level set to: {level.value}")
    
    def enable_safety_check(self, check_name: str):
        """Enable specific safety check."""
        if check_name in self.safety_checks:
            self.safety_checks[check_name] = True
            logger.info(f"Safety check enabled: {check_name}")
        else:
            logger.warning(f"Unknown safety check: {check_name}")
    
    def disable_safety_check(self, check_name: str):
        """Disable specific safety check."""
        if check_name in self.safety_checks:
            self.safety_checks[check_name] = False
            logger.info(f"Safety check disabled: {check_name}")
        else:
            logger.warning(f"Unknown safety check: {check_name}")
    
    def get_ai_confidence_score(self, plan_id: str) -> float:
        """Get AI confidence score for a plan."""
        if plan_id not in self.execution_plans:
            return 0.0
        
        plan = self.execution_plans[plan_id]
        confidence_value = self.confidence_thresholds.get(plan.confidence, 0.5)
        
        return confidence_value


# Global instance
ai_maturity_enhancer = AIMaturityEnhancer()


def get_ai_maturity_enhancer() -> AIMaturityEnhancer:
    """Get global AI maturity enhancer."""
    return ai_maturity_enhancer


def initialize_ai_maturity_system():
    """Initialize AI maturity enhancement system."""
    enhancer = get_ai_maturity_enhancer()
    enhancer.set_validation_level(ExecutionValidationLevel.MODERATE)
    
    # Enable all safety checks
    for check_name in enhancer.safety_checks:
        enhancer.enable_safety_check(check_name)
    
    logger.info("AI maturity enhancement system initialized")
