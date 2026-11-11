import pytest
import pandas as pd
import streamlit as st
from modules.ai_engine import should_search_web, extract_search_query, run_automl, validate_code, validate_ai_response, preview_code_execution

def test_should_search_web():
    """Test web search trigger detection."""
    assert should_search_web("What is the latest pandas version?") == True
    assert should_search_web("How to clean data in pandas?") == True
    assert should_search_web("Plot a histogram") == False

def test_extract_search_query():
    """Test extracting search query from prompt."""
    query = extract_search_query("What is the latest version of scikit-learn?")
    assert "scikit-learn" in query

def test_automl_import():
    """Test that AutoML function can be imported."""
    # Just test that the function exists and can handle basic input
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    result = run_automl(df, 'b', time_limit=1)  # Very short time limit
    assert isinstance(result, dict)
    assert 'success' in result

def test_validate_code_security():
    """Test security pattern detection in code."""
    # Safe code
    valid, _ = validate_code("df = datasets['id']['df']\nprint(df.head())")
    assert valid == True
    
    # Unsafe code
    unsafe_snippets = [
        "import os; os.system('ls')",
        "import subprocess; subprocess.run(['ls'])",
        "eval('print(1)')",
        "open('test.txt', 'w').write('hello')",
        "import requests; requests.get('http://google.com')",
    ]
    for snippet in unsafe_snippets:
        valid, msg = validate_code(snippet)
        assert valid == False
        assert "Security" in msg

def test_validate_ai_response():
    """Test detection of hallucinated columns."""
    ctx = "=== MyData ===\nCOLUMNS:\n- 'Age'\n- 'Salary'\n"
    
    # Valid response
    valid, _ = validate_ai_response("Show age", "df['Age'].mean()", ctx)
    assert valid == True
    
    # Hallucinated response
    valid, msg = validate_ai_response("Show height", "df['Height'].mean()", ctx)
    assert valid == False
    assert "Height" in msg

def test_preview_code_execution_dataset_id():
    """Test that preview mode flags unknown dataset IDs."""
    # Mock streamlit session state
    if not hasattr(st, "session_state"):
        # In pytest, session_state might not be initialized
        from types import SimpleNamespace
        st.session_state = SimpleNamespace(datasets={"real_id": {"name": "Real Data"}})
    else:
        st.session_state.datasets = {"real_id": {"name": "Real Data"}}
    
    # Code with real ID
    preview = preview_code_execution("df = datasets['real_id']['df']")
    assert "Real Data" in preview
    
    # Code with fake ID
    preview = preview_code_execution("df = datasets['fake_id']['df']")
    assert "⚠️ Access UNKNOWN dataset ID: fake_id" in preview

def test_ask_groq_missing_key_returns_friendly_message(monkeypatch):
    """ask_groq should not crash when no Groq key is available."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(st, "session_state", {}, raising=False)
    monkeypatch.setattr("modules.ai_engine.load_config", lambda: {}, raising=False)

    from modules.ai_engine import ask_groq

    reply, code, confidence = ask_groq("analyze this dataset", groq_key=None, include_dataset_context=False, allow_web_search=False)
    assert "Groq API key is not configured" in reply
    assert code is None
    assert confidence == 0.0