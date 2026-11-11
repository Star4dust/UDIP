"""Agent Pipeline page wrapper.

The pipeline builder remains implemented in modules.pipeline_builder. This file
only gives the centralized router a real page target.
"""

from modules.pipeline_builder import render_pipeline_builder_with_context


def render():
    render_pipeline_builder_with_context()
