"""
UDIP v3 — modules/execution_reliability.py
Execution and pipeline reliability hardening utilities.

This module provides enterprise-grade execution reliability with standardized
schemas, retry policies, timeout safeguards, and comprehensive error handling.
"""

import time
import logging
import threading
import traceback
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


class RetryPolicy(Enum):
    """Retry policy enumeration."""
    NO_RETRY = "no_retry"
    LINEAR_BACKOFF = "linear_backoff"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_INTERVAL = "fixed_interval"


@dataclass
class ExecutionSchema:
    """Standardized execution schema."""
    action_id: str
    action_type: str
    parameters: Dict[str, Any]
    expected_outputs: List[str]
    timeout_seconds: float
    retry_policy: RetryPolicy
    max_retries: int
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    rollback_handler: Optional[str] = None


@dataclass
class ExecutionResult:
    """Standardized execution result."""
    action_id: str
    status: ExecutionStatus
    output: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    validation_results: Dict[str, bool] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    rollback_successful: bool = False
    trace_id: Optional[str] = None


class ExecutionReliabilityManager:
    """Manages execution reliability with comprehensive error handling."""
    
    def __init__(self):
        self.execution_schemas = {}
        self.execution_history = []
        self.active_executions = {}
        self.retry_handlers = {}
        self.timeout_handlers = {}
        self.rollback_handlers = {}
        self.performance_metrics = {}
        self.execution_traces = {}
        
    def register_execution_schema(self, schema: ExecutionSchema):
        """Register an execution schema."""
        self.execution_schemas[schema.action_id] = schema
        logger.info(f"Registered execution schema: {schema.action_id}")
    
    def execute_with_reliability(self, action_id: str, parameters: Dict[str, Any] = None) -> ExecutionResult:
        """Execute action with full reliability guarantees."""
        if action_id not in self.execution_schemas:
            logger.error(f"Execution schema not found: {action_id}")
            return ExecutionResult(
                action_id=action_id,
                status=ExecutionStatus.FAILED,
                error_message=f"Schema not found: {action_id}",
                execution_time=0.0
            )
        
        schema = self.execution_schemas[action_id]
        execution_id = self._generate_execution_id(action_id)
        
        # Create execution trace
        trace_id = self._create_execution_trace(execution_id, action_id, parameters)
        
        # Execute with reliability
        result = self._execute_with_retry_and_timeout(schema, parameters, execution_id, trace_id)
        
        # Store execution result
        self.execution_history.append(result)
        
        logger.info(f"Execution completed: {execution_id} - {result.status.value}")
        return result
    
    def _execute_with_retry_and_timeout(self, schema: ExecutionSchema, parameters: Dict[str, Any], 
                                     execution_id: str, trace_id: str) -> ExecutionResult:
        """Execute with retry logic and timeout handling."""
        start_time = time.time()
        retry_count = 0
        last_error = None
        
        while retry_count <= schema.max_retries:
            # Check if execution is cancelled
            if execution_id in self.active_executions:
                if self.active_executions[execution_id].get("cancelled", False):
                    return ExecutionResult(
                        action_id=schema.action_id,
                        status=ExecutionStatus.CANCELLED,
                        error_message="Execution cancelled",
                        execution_time=time.time() - start_time,
                        retry_count=retry_count,
                        trace_id=trace_id
                    )
            
            # Update execution status
            if retry_count > 0:
                self._update_execution_status(execution_id, ExecutionStatus.RETRYING)
            
            # Execute with timeout
            try:
                result = self._execute_with_timeout(schema, parameters, execution_id, retry_count)
                
                # Validate result
                if self._validate_execution_result(result, schema):
                    # Success
                    return ExecutionResult(
                        action_id=schema.action_id,
                        status=ExecutionStatus.COMPLETED,
                        output=result,
                        execution_time=time.time() - start_time,
                        retry_count=retry_count,
                        validation_results=self._get_validation_results(result, schema),
                        performance_metrics=self._get_performance_metrics(execution_id),
                        trace_id=trace_id
                    )
                else:
                    # Validation failed
                    last_error = "Result validation failed"
                    retry_count += 1
                    
            except TimeoutError:
                last_error = "Execution timeout"
                retry_count += 1
                
            except Exception as e:
                last_error = str(e)
                retry_count += 1
            
            # Apply retry delay
            if retry_count <= schema.max_retries:
                delay = self._calculate_retry_delay(retry_count, schema.retry_policy)
                time.sleep(delay)
        
        # All retries exhausted
        return ExecutionResult(
            action_id=schema.action_id,
            status=ExecutionStatus.FAILED,
            error_message=last_error,
            execution_time=time.time() - start_time,
            retry_count=retry_count,
            trace_id=trace_id
        )
    
    def _execute_with_timeout(self, schema: ExecutionSchema, parameters: Dict[str, Any], 
                           execution_id: str, retry_count: int) -> Any:
        """Execute with timeout protection."""
        # Update execution status
        self._update_execution_status(execution_id, ExecutionStatus.RUNNING)
        
        # Create execution context
        execution_context = {
            "execution_id": execution_id,
            "action_id": schema.action_id,
            "parameters": parameters,
            "retry_count": retry_count,
            "schema": schema
        }
        
        # Execute in thread with timeout
        result = None
        exception = None
        
        def target_function():
            nonlocal result, exception
            try:
                result = self._execute_action(schema, parameters, execution_context)
            except Exception as e:
                exception = e
        
        # Start execution thread
        execution_thread = threading.Thread(target=target_function)
        execution_thread.daemon = True
        execution_thread.start()
        
        # Wait for completion or timeout
        execution_thread.join(timeout=schema.timeout_seconds)
        
        if execution_thread.is_alive():
            # Timeout occurred
            logger.error(f"Execution timeout: {execution_id}")
            raise TimeoutError(f"Execution timed out after {schema.timeout_seconds} seconds")
        
        if exception:
            raise exception
        
        return result
    
    def _execute_action(self, schema: ExecutionSchema, parameters: Dict[str, Any], 
                      execution_context: Dict[str, Any]) -> Any:
        """Execute the actual action."""
        # This would be implemented by specific action handlers
        # For now, we'll simulate execution
        
        action_type = schema.action_type
        
        if action_type == "data_processing":
            return self._execute_data_processing_action(parameters, execution_context)
        elif action_type == "ai_execution":
            return self._execute_ai_action(parameters, execution_context)
        elif action_type == "pipeline_execution":
            return self._execute_pipeline_action(parameters, execution_context)
        elif action_type == "hirax_processing":
            return self._execute_hirax_action(parameters, execution_context)
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    def _execute_data_processing_action(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute data processing action."""
        # Simulate data processing
        logger.info(f"Executing data processing action: {parameters}")
        
        # In production, this would process actual data
        result = {
            "processed": True,
            "rows_processed": 1000,
            "execution_time": 0.5,
            "memory_used": 50.0
        }
        
        return result
    
    def _execute_ai_action(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute AI action."""
        logger.info(f"Executing AI action: {parameters}")
        
        # Simulate AI execution
        result = {
            "ai_response": "AI processing completed",
            "confidence": 0.85,
            "tokens_used": 150,
            "execution_time": 1.2
        }
        
        return result
    
    def _execute_pipeline_action(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute pipeline action."""
        logger.info(f"Executing pipeline action: {parameters}")
        
        # Simulate pipeline execution
        result = {
            "pipeline_completed": True,
            "nodes_executed": 5,
            "execution_time": 2.0,
            "success_rate": 1.0
        }
        
        return result
    
    def _execute_hirax_action(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute HIRAX action."""
        logger.info(f"Executing HIRAX action: {parameters}")
        
        # Simulate HIRAX processing
        result = {
            "hirax_processed": True,
            "spectra_generated": 100,
            "rfi_detected": 25,
            "execution_time": 3.5,
            "scientific_valid": True
        }
        
        return result
    
    def _validate_execution_result(self, result: Any, schema: ExecutionSchema) -> bool:
        """Validate execution result against schema."""
        if not schema.validation_rules:
            return True
        
        for rule in schema.validation_rules:
            if not self._apply_validation_rule(result, rule):
                logger.warning(f"Validation rule failed: {rule}")
                return False
        
        return True
    
    def _apply_validation_rule(self, result: Any, rule: Dict[str, Any]) -> bool:
        """Apply individual validation rule."""
        rule_type = rule.get("type", "existence")
        
        if rule_type == "existence":
            # Check if required output exists
            required_output = rule.get("output", "")
            if isinstance(result, dict) and required_output in result:
                return True
            return False
        
        elif rule_type == "type_check":
            # Check output type
            expected_type = rule.get("expected_type", "dict")
            return isinstance(result, eval(expected_type))
        
        elif rule_type == "value_range":
            # Check if value is in expected range
            min_val = rule.get("min", float('-inf'))
            max_val = rule.get("max", float('inf'))
            
            if isinstance(result, (int, float)):
                return min_val <= result <= max_val
            return False
        
        elif rule_type == "custom":
            # Custom validation function
            validator = rule.get("validator")
            if callable(validator):
                return validator(result)
            return False
        
        return True
    
    def _get_validation_results(self, result: Any, schema: ExecutionSchema) -> Dict[str, bool]:
        """Get validation results for all rules."""
        validation_results = {}
        
        for i, rule in enumerate(schema.validation_rules):
            rule_name = f"rule_{i}"
            validation_results[rule_name] = self._apply_validation_rule(result, rule)
        
        return validation_results
    
    def _get_performance_metrics(self, execution_id: str) -> Dict[str, float]:
        """Get performance metrics for execution."""
        # In production, this would collect actual metrics
        return {
            "cpu_usage": 25.5,
            "memory_usage": 150.2,
            "io_wait_time": 0.1,
            "context_switches": 1000
        }
    
    def _calculate_retry_delay(self, retry_count: int, policy: RetryPolicy) -> float:
        """Calculate retry delay based on policy."""
        if policy == RetryPolicy.NO_RETRY:
            return 0.0
        elif policy == RetryPolicy.FIXED_INTERVAL:
            return 5.0  # Fixed 5 second delay
        elif policy == RetryPolicy.LINEAR_BACKOFF:
            return retry_count * 2.0  # Linear increase
        elif policy == RetryPolicy.EXPONENTIAL_BACKOFF:
            return min(60.0, 2.0 ** retry_count)  # Exponential backoff, max 60 seconds
        else:
            return 1.0
    
    def _generate_execution_id(self, action_id: str) -> str:
        """Generate unique execution ID."""
        timestamp = int(time.time() * 1000)
        return f"{action_id}_{timestamp}"
    
    def _create_execution_trace(self, execution_id: str, action_id: str, parameters: Dict[str, Any]) -> str:
        """Create execution trace for debugging."""
        trace_data = {
            "execution_id": execution_id,
            "action_id": action_id,
            "parameters": parameters,
            "start_time": time.time(),
            "trace_events": []
        }
        
        trace_id = hashlib.md5(json.dumps(trace_data, sort_keys=True).hexdigest())
        self.execution_traces[trace_id] = trace_data
        
        return trace_id
    
    def _update_execution_status(self, execution_id: str, status: ExecutionStatus):
        """Update execution status."""
        if execution_id not in self.active_executions:
            self.active_executions[execution_id] = {}
        
        self.active_executions[execution_id]["status"] = status
        self.active_executions[execution_id]["status_updated"] = time.time()
        
        logger.debug(f"Execution status updated: {execution_id} -> {status.value}")
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution."""
        if execution_id in self.active_executions:
            self.active_executions[execution_id]["cancelled"] = True
            logger.info(f"Execution cancelled: {execution_id}")
            return True
        return False
    
    def rollback_execution(self, execution_id: str) -> bool:
        """Rollback a failed execution."""
        # Find execution from history
        execution_result = None
        for result in self.execution_history:
            if result.action_id == execution_id:
                execution_result = result
                break
        
        if not execution_result:
            logger.error(f"Execution not found for rollback: {execution_id}")
            return False
        
        # Get schema for rollback handler
        action_id = execution_result.action_id
        if action_id in self.execution_schemas:
            schema = self.execution_schemas[action_id]
            if schema.rollback_handler:
                try:
                    # Execute rollback handler
                    rollback_result = self._execute_rollback_handler(schema.rollback_handler, execution_result)
                    
                    if rollback_result:
                        logger.info(f"Execution rollback successful: {execution_id}")
                        return True
                    else:
                        logger.error(f"Execution rollback failed: {execution_id}")
                        return False
                        
                except Exception as e:
                    logger.error(f"Rollback handler error: {e}")
                    return False
        
        logger.warning(f"No rollback handler found for: {execution_id}")
        return False
    
    def _execute_rollback_handler(self, handler_name: str, execution_result: ExecutionResult) -> bool:
        """Execute rollback handler."""
        # This would be implemented by specific rollback handlers
        # For now, we'll simulate successful rollback
        logger.info(f"Executing rollback handler: {handler_name}")
        return True
    
    def get_execution_history(self) -> List[ExecutionResult]:
        """Get execution history."""
        return self.execution_history
    
    def get_active_executions(self) -> Dict[str, Dict[str, Any]]:
        """Get active executions."""
        return self.active_executions
    
    def get_execution_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get execution trace."""
        return self.execution_traces.get(trace_id)
    
    def get_reliability_metrics(self) -> Dict[str, Any]:
        """Get reliability metrics."""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "timeout_rate": 0.0,
                "retry_rate": 0.0
            }
        
        total_executions = len(self.execution_history)
        successful_executions = len([r for r in self.execution_history if r.status == ExecutionStatus.COMPLETED])
        timeout_executions = len([r for r in self.execution_history if r.status == ExecutionStatus.TIMEOUT])
        retry_executions = len([r for r in self.execution_history if r.retry_count > 0])
        
        total_execution_time = sum(r.execution_time for r in self.execution_history)
        average_execution_time = total_execution_time / total_executions if total_executions > 0 else 0.0
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0.0,
            "timeout_executions": timeout_executions,
            "timeout_rate": timeout_executions / total_executions if total_executions > 0 else 0.0,
            "retry_executions": retry_executions,
            "retry_rate": retry_executions / total_executions if total_executions > 0 else 0.0,
            "average_execution_time": average_execution_time
        }


# Global instance
execution_reliability_manager = ExecutionReliabilityManager()


def get_execution_reliability_manager() -> ExecutionReliabilityManager:
    """Get global execution reliability manager."""
    return execution_reliability_manager


def initialize_execution_reliability():
    """Initialize execution reliability system."""
    logger.info("Initializing execution reliability system")
    
    # Register common execution schemas
    reliability_manager = get_execution_reliability_manager()
    
    # Data processing schema
    data_processing_schema = ExecutionSchema(
        action_id="data_processing",
        action_type="data_processing",
        parameters={},
        expected_outputs=["processed_data"],
        timeout_seconds=30.0,
        retry_policy=RetryPolicy.EXPONENTIAL_BACKOFF,
        max_retries=3,
        validation_rules=[
            {"type": "existence", "output": "processed_data"},
            {"type": "type_check", "expected_type": "dict"}
        ]
    )
    reliability_manager.register_execution_schema(data_processing_schema)
    
    # AI execution schema
    ai_execution_schema = ExecutionSchema(
        action_id="ai_execution",
        action_type="ai_execution",
        parameters={},
        expected_outputs=["ai_response"],
        timeout_seconds=60.0,
        retry_policy=RetryPolicy.LINEAR_BACKOFF,
        max_retries=2,
        validation_rules=[
            {"type": "existence", "output": "ai_response"},
            {"type": "value_range", "min": 0.0, "max": 1.0, "expected_type": "confidence"}
        ]
    )
    reliability_manager.register_execution_schema(ai_execution_schema)
    
    # Pipeline execution schema
    pipeline_execution_schema = ExecutionSchema(
        action_id="pipeline_execution",
        action_type="pipeline_execution",
        parameters={},
        expected_outputs=["pipeline_result"],
        timeout_seconds=120.0,
        retry_policy=RetryPolicy.EXPONENTIAL_BACKOFF,
        max_retries=1,
        validation_rules=[
            {"type": "existence", "output": "pipeline_result"},
            {"type": "custom", "validator": lambda x: isinstance(x, dict) and x.get("pipeline_completed", False)}
        ]
    )
    reliability_manager.register_execution_schema(pipeline_execution_schema)
    
    # HIRAX processing schema
    hirax_processing_schema = ExecutionSchema(
        action_id="hirax_processing",
        action_type="hirax_processing",
        parameters={},
        expected_outputs=["hirax_result"],
        timeout_seconds=300.0,
        retry_policy=RetryPolicy.NO_RETRY,
        max_retries=0,
        validation_rules=[
            {"type": "existence", "output": "hirax_result"},
            {"type": "custom", "validator": lambda x: isinstance(x, dict) and x.get("scientific_valid", False)}
        ],
        rollback_handler="hirax_rollback"
    )
    reliability_manager.register_execution_schema(hirax_processing_schema)
    
    logger.info("Execution reliability system initialized")
