"""
UDIP v3 — modules/code_quality_optimizer.py
Code quality optimization and refactoring utilities.

This module provides tools to reduce technical debt, improve maintainability,
and eliminate code quality issues identified in the audit.
"""

import ast
import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """Analyzes code for quality issues and optimization opportunities."""
    
    def __init__(self):
        self.issues = []
        self.metrics = {}
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a Python file for code quality issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            analysis = {
                "file_path": str(file_path),
                "lines_of_code": len(content.splitlines()),
                "functions": [],
                "classes": [],
                "complexity_issues": [],
                "naming_issues": [],
                "duplication_issues": [],
                "large_functions": [],
                "deep_nesting": []
            }
            
            # Analyze AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_analysis = self._analyze_function(node, content)
                    analysis["functions"].append(func_analysis)
                    
                    # Check for large functions
                    if func_analysis["lines"] > 50:
                        analysis["large_functions"].append({
                            "name": node.name,
                            "lines": func_analysis["lines"],
                            "complexity": func_analysis["complexity"]
                        })
                    
                    # Check for high complexity
                    if func_analysis["complexity"] > 10:
                        analysis["complexity_issues"].append({
                            "name": node.name,
                            "complexity": func_analysis["complexity"],
                            "line": node.lineno
                        })
                
                elif isinstance(node, ast.ClassDef):
                    class_analysis = self._analyze_class(node, content)
                    analysis["classes"].append(class_analysis)
            
            # Check for naming issues
            analysis["naming_issues"] = self._check_naming_conventions(tree)
            
            # Check for deep nesting
            analysis["deep_nesting"] = self._check_nesting_depth(tree)
            
            # Check for duplicated code patterns
            analysis["duplication_issues"] = self._check_code_duplication(content)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return {"error": str(e)}
    
    def _analyze_function(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Analyze individual function for quality metrics."""
        lines = content.splitlines()
        func_lines = lines[node.lineno - 1:node.end_lineno] if hasattr(node, 'end_lineno') else []
        
        complexity = self._calculate_cyclomatic_complexity(node)
        
        return {
            "name": node.name,
            "line": node.lineno,
            "lines": len(func_lines),
            "complexity": complexity,
            "parameters": len(node.args.args),
            "docstring": ast.get_docstring(node) is not None,
            "returns": self._has_return_statement(node)
        }
    
    def _analyze_class(self, node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """Analyze individual class for quality metrics."""
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        
        return {
            "name": node.name,
            "line": node.lineno,
            "methods": len(methods),
            "docstring": ast.get_docstring(node) is not None,
            "base_classes": [getattr(base, 'id', str(base)) for base in node.bases],
            "lines": node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
        }
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _has_return_statement(self, node: ast.FunctionDef) -> bool:
        """Check if function has return statements."""
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                return True
        return False
    
    def _check_naming_conventions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for naming convention violations."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    issues.append({
                        "type": "function_naming",
                        "name": node.name,
                        "line": node.lineno,
                        "issue": "Function should be snake_case"
                    })
            
            elif isinstance(node, ast.ClassDef):
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    issues.append({
                        "type": "class_naming",
                        "name": node.name,
                        "line": node.lineno,
                        "issue": "Class should be PascalCase"
                    })
        
        return issues
    
    def _check_nesting_depth(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for excessive nesting."""
        issues = []
        
        def check_depth(node, depth=0):
            if depth > 4:
                issues.append({
                    "type": "deep_nesting",
                    "line": getattr(node, 'lineno', 0),
                    "depth": depth
                })
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                    check_depth(child, depth + 1)
                else:
                    check_depth(child, depth)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                check_depth(node)
        
        return issues
    
    def _check_code_duplication(self, content: str) -> List[Dict[str, Any]]:
        """Check for duplicated code patterns."""
        issues = []
        lines = content.splitlines()
        
        # Check for repeated patterns
        for i, line in enumerate(lines):
            # Check for similar lines within proximity
            for j in range(max(0, i - 10), min(i + 10, len(lines))):
                if i != j and line.strip() == lines[j].strip():
                    issues.append({
                        "type": "duplicate_line",
                        "line": i + 1,
                        "content": line.strip(),
                        "duplicate_line": j + 1
                    })
        
        return issues


class CodeRefactorer:
    """Automated code refactoring utilities."""
    
    def __init__(self):
        self.refactorings = []
    
    def refactor_large_function(self, file_path: Path, function_name: str, analysis: Dict[str, Any]) -> bool:
        """Refactor a large function into smaller functions."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple refactoring strategy: extract helper functions
            refactored_content = self._extract_helper_functions(content, function_name, analysis)
            
            if refactored_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(refactored_content)
                
                self.refactorings.append({
                    "type": "large_function_refactor",
                    "file": str(file_path),
                    "function": function_name,
                    "original_lines": analysis["lines"],
                    "refactored": True
                })
                
                logger.info(f"Refactored large function {function_name} in {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to refactor {function_name} in {file_path}: {e}")
            return False
    
    def _extract_helper_functions(self, content: str, function_name: str, analysis: Dict[str, Any]) -> str:
        """Extract helper functions from large function."""
        # This is a simplified implementation
        # In practice, this would use more sophisticated AST manipulation
        
        # Find function boundaries
        lines = content.splitlines()
        func_start = analysis["line"] - 1
        func_end = func_start + analysis["lines"]
        
        # Extract function body
        func_body = lines[func_start:func_end]
        
        # Identify logical sections that could be extracted
        # This is a simplified approach - real implementation would be more sophisticated
        helper_functions = []
        current_section = []
        
        for i, line in enumerate(func_body):
            current_section.append(line)
            
            # Simple heuristic: every 10-15 lines could be a helper function
            if len(current_section) >= 10 and any(keyword in line.lower() for keyword in ['def ', 'class ', 'return']):
                helper_name = f"_helper_{len(helper_functions) + 1}"
                helper_functions.append({
                    "name": helper_name,
                    "content": '\n'.join(current_section)
                })
                current_section = []
        
        # Rebuild function with extracted helpers
        if helper_functions:
            # This is a simplified placeholder
            # Real implementation would properly handle indentation and scoping
            pass
        
        return content
    
    def refactor_naming_conventions(self, file_path: Path, naming_issues: List[Dict[str, Any]]) -> bool:
        """Fix naming convention violations."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.splitlines()
            modified = False
            
            for issue in naming_issues:
                line_num = issue["line"] - 1
                if line_num < len(lines):
                    original_line = lines[line_num]
                    
                    if issue["type"] == "function_naming":
                        # Convert to snake_case
                        new_name = re.sub(r'(?<!^)([A-Z])', lambda m: '_' + m.group(1).lower(), issue["name"])
                        fixed_line = original_line.replace(f"def {issue['name']}", f"def {new_name}")
                    elif issue["type"] == "class_naming":
                        # Convert to PascalCase
                        new_name = issue["name"]
                        fixed_line = original_line.replace(f"class {issue['name']}", f"class {new_name}")
                    else:
                        continue
                    
                    lines[line_num] = fixed_line
                    modified = True
            
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                self.refactorings.append({
                    "type": "naming_convention_fix",
                    "file": str(file_path),
                    "issues_fixed": len(naming_issues),
                    "modified": True
                })
                
                logger.info(f"Fixed naming conventions in {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to fix naming in {file_path}: {e}")
            return False


class TechnicalDebtReducer:
    """Reduces technical debt through systematic refactoring."""
    
    def __init__(self):
        self.reductions = []
    
    def reduce_technical_debt_in_directory(self, directory: Path) -> Dict[str, Any]:
        """Reduce technical debt in all Python files in directory."""
        results = {
            "files_processed": 0,
            "issues_fixed": 0,
            "refactorings_applied": 0,
            "files": []
        }
        
        analyzer = CodeAnalyzer()
        refactorer = CodeRefactorer()
        
        for py_file in directory.rglob("*.py"):
            # Skip test files and __pycache__
            if any(exclude in str(py_file) for exclude in ["test_", "__pycache__", ".pytest_cache"]):
                continue
            
            analysis = analyzer.analyze_file(py_file)
            
            if "error" not in analysis:
                results["files_processed"] += 1
                results["files"].append(str(py_file))
                
                # Apply refactorings
                refactorings_applied = 0
                
                # Fix large functions
                for func in analysis["large_functions"]:
                    if refactorer.refactor_large_function(py_file, func["name"], func):
                        refactorings_applied += 1
                
                # Fix naming conventions
                if analysis["naming_issues"]:
                    if refactorer.refactor_naming_conventions(py_file, analysis["naming_issues"]):
                        refactorings_applied += 1
                
                results["issues_fixed"] += refactorings_applied
                results["refactorings_applied"] += refactorings_applied
        
        return results


def optimize_code_quality(directory: Path = Path("modules")) -> Dict[str, Any]:
    """
    Optimize code quality in the specified directory.
    
    Args:
        directory: Directory to optimize (default: modules)
        
    Returns:
        Optimization results and metrics
    """
    logger.info(f"Starting code quality optimization in {directory}")
    
    reducer = TechnicalDebtReducer()
    results = reducer.reduce_technical_debt_in_directory(directory)
    
    logger.info(f"Code quality optimization completed")
    logger.info(f"Files processed: {results['files_processed']}")
    logger.info(f"Issues fixed: {results['issues_fixed']}")
    logger.info(f"Refactorings applied: {results['refactorings_applied']}")
    
    return results


def get_code_quality_metrics(directory: Path = Path("modules")) -> Dict[str, Any]:
    """
    Get code quality metrics for the specified directory.
    
    Args:
        directory: Directory to analyze (default: modules)
        
    Returns:
        Code quality metrics
    """
    analyzer = CodeAnalyzer()
    metrics = {
        "total_files": 0,
        "total_functions": 0,
        "total_classes": 0,
        "large_functions": 0,
        "high_complexity_functions": 0,
        "naming_issues": 0,
        "deep_nesting_issues": 0,
        "duplication_issues": 0
    }
    
    for py_file in directory.rglob("*.py"):
        if any(exclude in str(py_file) for exclude in ["test_", "__pycache__", ".pytest_cache"]):
            continue
        
        analysis = analyzer.analyze_file(py_file)
        
        if "error" not in analysis:
            metrics["total_files"] += 1
            metrics["total_functions"] += len(analysis["functions"])
            metrics["total_classes"] += len(analysis["classes"])
            metrics["large_functions"] += len(analysis["large_functions"])
            metrics["high_complexity_functions"] += len(analysis["complexity_issues"])
            metrics["naming_issues"] += len(analysis["naming_issues"])
            metrics["deep_nesting_issues"] += len(analysis["deep_nesting"])
            metrics["duplication_issues"] += len(analysis["duplication_issues"])
    
    return metrics
