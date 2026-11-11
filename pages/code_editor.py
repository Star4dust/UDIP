import streamlit as st
import pandas as pd
import numpy as np
import time
from streamlit_ace import st_ace
from modules.core import save_code, add_history, log

try:
    from modules.ai_engine import run_code as ai_run_code
except ImportError:
    ai_run_code = None

def render():
    st.markdown('<div class="main-header">🐍 Python Code Editor</div>', unsafe_allow_html=True)

    if not st.session_state.datasets:
        st.warning("⚠️ No datasets loaded. Code editor works best with data!")
    
    st.markdown("""
    **Global Environment:**
    - `datasets`: Dict of all loaded datasets `{id: {name, df, ...}}`
    - `pd`: Pandas
    - `np`: Numpy
    - `st`: Streamlit (create your own UI!)
    """)

    # Editor Layout
    code_input = st_ace(
        value=st.session_state.editor_code,
        language="python",
        theme="monokai" if st.session_state.get("dark_mode") else "chrome",
        keybinding="vscode",
        font_size=14,
        tab_size=4,
        min_lines=20,
        max_lines=40,
        show_gutter=True,
        wrap=True,
        auto_update=False,
        key="ace_editor"
    )

    if code_input != st.session_state.editor_code:
        st.session_state.editor_code = code_input
        save_code(code_input)

    col_run, col_clear, _ = st.columns([1, 1, 4])
    
    if col_run.button("▶️ Run Code", type="primary", width="stretch"):
        with st.spinner("Executing..."):
            start = time.time()
            # Prepare namespace
            namespace = {
                "pd": pd, "np": np, "st": st,
                "datasets": st.session_state.datasets,
                "plt": None # Optional
            }
            try:
                import matplotlib.pyplot as plt
                namespace["plt"] = plt
            except ImportError: pass

            output, error = ai_run_code(code_input, namespace)
            elapsed = time.time() - start
            
            if error:
                st.error(f"Execution Error ({elapsed:.2f}s):\n{error}")
            else:
                st.success(f"Execution Successful ({elapsed:.2f}s)")
                if output:
                    st.markdown("#### Console Output")
                    st.code(output)
                add_history("Code run", f"Manual script — {len(code_input)} chars")

    if col_clear.button("🗑️ Clear Output", width="stretch"):
        st.rerun()

    with st.expander("💡 Helper: Quick Code Snippets"):
        snippets = {
            "Load first dataset": "df = list(datasets.values())[0]['df']\nst.write(df.head())",
            "Simple Plot": "import matplotlib.pyplot as plt\ndf = list(datasets.values())[0]['df']\nfig, ax = plt.subplots()\ndf.select_dtypes('number').iloc[:,0].plot(kind='hist', ax=ax)\nst.pyplot(fig)",
            "Add column": "df = list(datasets.values())[0]['df']\ndf['new_col'] = df.iloc[:,0] * 2\nst.success('Added column!')"
        }
        for name, snip in snippets.items():
            if st.button(name):
                st.session_state.editor_code = snip
                save_code(snip)
                st.rerun()
