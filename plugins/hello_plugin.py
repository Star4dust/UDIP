"""
UDIP Plugin: Hello World
Example plugin to demonstrate extensibility.
"""

import streamlit as st

def register():
    """Register the plugin with UDIP."""
    return "Hello Plugin", "👋", render

def render():
    """Render the plugin page."""
    st.title("👋 Welcome to Plugins")
    st.markdown("This page was loaded dynamically from the `plugins/` directory.")
    
    st.balloons()
    
    st.info("You can build your own custom UDIP modules by dropping .py files into the plugins folder!")
    
    if st.button("Click me!"):
        st.write("Plugin interaction works!")
