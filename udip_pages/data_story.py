"""
UDIP v3 — udip_pages/data_story.py
Report Studio — Collect artifacts, add notes/images/links/KPIs,
                tag & organise, reorder, export.
"""

import base64
import datetime
import re

import streamlit as st

from modules.core import load_config
from modules.state_manager import get_state_manager


# ── Tag colour palette (cycles through for variety) ───────────────────────────
_TAG_COLOURS = [
    "#2a5298", "#7c3aed", "#0891b2", "#059669",
    "#d97706", "#dc2626", "#db2777", "#65a30d",
]

def _tag_colour(tag: str) -> str:
    return _TAG_COLOURS[hash(tag) % len(_TAG_COLOURS)]


def _tag_pills(tags: list) -> str:
    """Return an HTML string of coloured tag pills."""
    return " ".join(
        f'<span style="background:{_tag_colour(t)};color:#fff;'
        f'padding:2px 10px;border-radius:12px;font-size:0.76em;'
        f'margin-right:4px;display:inline-block">{t}</span>'
        for t in tags
    )


def _parse_tags(raw: str) -> list:
    return [t.strip() for t in raw.split(",") if t.strip()]


# ── Export builders ────────────────────────────────────────────────────────────

def _build_markdown(artifacts: list) -> str:
    lines = [
        "# UDIP Report",
        f"_Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
    ]
    for item in artifacts:
        title = item.get("title", "Untitled")
        type_ = item.get("type", "unknown")
        data  = item.get("data", {})
        ts    = item.get("timestamp", "")
        tags  = data.get("tags", [])

        lines.append(f"---\n## {title}")
        if ts:
            lines.append(f"_Pinned: {ts}_")
        if tags:
            lines.append(f"**Tags:** {', '.join(tags)}")
        lines.append("")

        if type_ == "note":
            lines.append(data.get("content", ""))
        elif type_ == "link":
            url  = data.get("url", "")
            desc = data.get("description", "")
            lines.append(f"🔗 [{url}]({url})")
            if desc:
                lines.append(f"\n{desc}")
        elif type_ == "kpi":
            label = data.get("label", "")
            value = data.get("value", "")
            delta = data.get("delta", "")
            note  = data.get("note", "")
            lines.append(f"**{label}:** {value}" + (f"  _(Δ {delta})_" if delta else ""))
            if note:
                lines.append(f"_Source: {note}_")
        elif type_ == "image":
            fname = data.get("filename", "image")
            cap   = data.get("caption", "")
            lines.append(f"_[Image: {fname}{(' — ' + cap) if cap else ''} — view in UDIP Report Studio]_")
        elif type_ == "chart":
            ct = data.get("chart_type", "Chart")
            ds = data.get("dataset", "")
            lines.append(f"📊 **Chart type:** {ct}" + (f"  |  **Dataset:** {ds}" if ds else ""))
        elif type_ in ("analysis", "insight"):
            lines.append(data.get("content", ""))
        elif type_ == "code":
            lines.append(f"```python\n{data.get('code', '')}\n```")
        else:
            if "content" in data:
                lines.append(data["content"])
        lines.append("")

    return "\n".join(lines)


def _build_html(md_content: str) -> str:
    html = md_content
    html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
    html = re.sub(r"^# (.+)$",  r"<h1>\1</h1>", html, flags=re.MULTILINE)
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.+?)\*",     r"<em>\1</em>", html)
    html = re.sub(r"_(.+?)_",       r"<em>\1</em>", html)
    html = re.sub(r"```python\n([\s\S]+?)\n```", r"<pre><code>\1</code></pre>", html)
    html = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', html)
    html = re.sub(r"^---$", "<hr>", html, flags=re.MULTILINE)
    html = "\n".join(
        f"<p>{line}</p>" if line.strip() and not line.startswith("<") else line
        for line in html.split("\n")
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>UDIP Report</title>
<style>
  body {{ font-family: Calibri, Arial, sans-serif; max-width: 900px; margin: 40px auto;
         line-height: 1.6; color: #222; }}
  h1 {{ color: #1a3c6e; border-bottom: 2px solid #1a3c6e; padding-bottom: 6px; }}
  h2 {{ color: #2a5298; margin-top: 32px; }}
  hr {{ border: 1px solid #ccc; margin: 24px 0; }}
  pre {{ background: #f4f4f4; padding: 12px; border-radius: 4px; overflow-x: auto; }}
  code {{ font-family: Consolas, monospace; font-size: 0.9em; }}
  em {{ color: #555; }}
  a {{ color: #2a5298; }}
</style>
</head>
<body>
{html}
</body>
</html>"""


# ── Artifact renderer ─────────────────────────────────────────────────────────

def _render_artifact(item: dict, idx: int, state_mgr, total: int):
    title   = item.get("title", "Untitled Artifact")
    type_   = item.get("type", "unknown")
    ts      = item.get("timestamp", "")
    data    = item.get("data", {})
    item_id = item.get("id", str(idx))
    tags    = data.get("tags", [])

    type_labels = {
        "chart":   "📊 CHART",
        "note":    "📝 NOTE",
        "image":   "🖼️ IMAGE",
        "link":    "🔗 LINK",
        "kpi":     "📈 KPI",
        "analysis":"🔍 ANALYSIS",
        "insight": "💡 INSIGHT",
        "code":    "🐍 CODE",
        "table":   "📋 TABLE",
    }
    badge = type_labels.get(type_, type_.upper())

    with st.expander(f"#{idx + 1}  ·  {badge}  ·  {title}", expanded=True):

        # Position indicator + timestamps
        meta_parts = []
        if ts:
            meta_parts.append(f"📅 {ts}")
        meta_parts.append(f"Position {idx + 1} of {total}")
        st.caption("  |  ".join(meta_parts))

        # Tag pills
        if tags:
            st.markdown(_tag_pills(tags), unsafe_allow_html=True)
            st.write("")  # spacing

        # ── Content by type ────────────────────────────────────────────────
        if type_ == "chart":
            fig_json = data.get("plotly_json")
            if fig_json:
                try:
                    import plotly.io as pio
                    fig = pio.from_json(fig_json)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not re-render chart: {e}")
            else:
                ct = data.get("chart_type", "Chart")
                ds = data.get("dataset", "")
                st.info(
                    f"📊 **{ct}**" + (f"  from dataset `{ds}`" if ds else "") +
                    "\n\nThis chart renders interactively — return to Visualize to view it."
                )

        elif type_ == "note":
            st.markdown(data.get("content", "_Empty note_"))

        elif type_ == "image":
            b64  = data.get("base64", "")
            mime = data.get("mime", "image/png")
            fname = data.get("filename", "image")
            cap   = data.get("caption", "")
            if b64:
                try:
                    st.image(base64.b64decode(b64), caption=cap or fname, use_column_width=True)
                except Exception as e:
                    st.error(f"Could not display image: {e}")
            else:
                st.warning("Image data not found.")

        elif type_ == "link":
            url  = data.get("url", "")
            desc = data.get("description", "")
            st.markdown(f"🔗 [{url}]({url})")
            if desc:
                st.caption(desc)

        elif type_ == "kpi":
            kc1, kc2 = st.columns([1, 2])
            kc1.metric(
                data.get("label", "Metric"),
                data.get("value", "—"),
                delta=data.get("delta") or None,
            )
            note = data.get("note", "")
            if note:
                kc2.caption(f"Source: {note}")

        elif type_ in ("analysis", "insight"):
            st.markdown(data.get("content", ""))

        elif type_ == "code":
            st.code(data.get("code", ""), language="python")

        else:
            if "content" in data:
                st.markdown(data["content"])

        # ── Action buttons: reorder + remove ──────────────────────────────
        st.write("")
        ac1, ac2, ac3, _spacer = st.columns([1, 1, 1, 4])

        if ac1.button("⬆️ Move Up",   key=f"up_{item_id}_{idx}",
                      disabled=(idx == 0),         help="Move item up in report"):
            state_mgr.reorder_item(item_id, "up")
            st.rerun()

        if ac2.button("⬇️ Move Down", key=f"down_{item_id}_{idx}",
                      disabled=(idx == total - 1), help="Move item down in report"):
            state_mgr.reorder_item(item_id, "down")
            st.rerun()

        if ac3.button("🗑️ Remove",    key=f"del_{item_id}_{idx}"):
            state_mgr.unpin_item(item_id)
            st.rerun()


# ── Tag input helper (reused across all Add Content types) ────────────────────

def _tags_input(key: str) -> list:
    raw = st.text_input(
        "🏷️ Tags (comma-separated, optional)",
        placeholder="e.g. critical, follow-up, external-source",
        key=key,
    )
    return _parse_tags(raw)


# ── Main render ────────────────────────────────────────────────────────────────

def render():
    st.markdown('<div class="main-header">📖 Report Studio</div>', unsafe_allow_html=True)
    st.markdown("Build your data story — pin charts and insights, add notes, images, links, and KPI cards.")

    state_mgr = get_state_manager()
    all_artifacts = state_mgr.get_pinned_items()

    # ── Header panel ──────────────────────────────────────────────────────────
    n_charts = len([a for a in all_artifacts if a.get("type") == "chart"])
    n_custom = len([a for a in all_artifacts if a.get("type") in ("note", "image", "link", "kpi")])

    st.markdown(
        f"""
        <div class="udip-panel">
            <div class="udip-panel__eyebrow">Executive Briefing</div>
            <div class="udip-panel__title">Report Studio</div>
            <div class="udip-panel__text">
                Pin charts, SQL results, and AI insights from any UDIP page — then enrich your report
                with notes, screenshots, external links, and KPI cards. Tag, reorder, and export.
            </div>
            <div class="udip-stat-strip">
                <div class="udip-stat-tile">
                    <div class="udip-stat-tile__value">{len(all_artifacts)}</div>
                    <div class="udip-stat-tile__label">Total Items</div>
                </div>
                <div class="udip-stat-tile">
                    <div class="udip-stat-tile__value">{n_charts}</div>
                    <div class="udip-stat-tile__label">Charts</div>
                </div>
                <div class="udip-stat-tile">
                    <div class="udip-stat-tile__value">{n_custom}</div>
                    <div class="udip-stat-tile__label">Custom Content</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_board, tab_add, tab_export = st.tabs(["📋 Report Board", "➕ Add Content", "📤 Export"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — REPORT BOARD
    # ══════════════════════════════════════════════════════════════════════════
    with tab_board:
        if not all_artifacts:
            st.info(
                "📌 Your report is empty. Pin charts, SQL queries, and AI insights from other pages, "
                "or use the **➕ Add Content** tab to add notes, images, and links."
            )
        else:
            # ── Top action bar ─────────────────────────────────────────────
            top1, top2, _sp = st.columns([1, 1, 2])

            with top1:
                if st.button("🗑️ Clear All", width="stretch"):
                    state_mgr.clear_pinned_items()
                    st.rerun()

            with top2:
                cfg      = load_config()
                groq_key = cfg.get("groq_key", "")
                if st.button("🤖 AI Summary", width="stretch", type="primary"):
                    if not groq_key:
                        st.error("Configure your Groq API key in Settings first.")
                    else:
                        with st.spinner("Synthesising Executive Summary..."):
                            try:
                                from groq import Groq
                                client = Groq(api_key=groq_key)
                                context = "PINNED DATA INSIGHTS:\n\n"
                                for item in all_artifacts:
                                    t = item.get("type", "unknown")
                                    d = item.get("data", {})
                                    context += f"--- {item.get('title', 'Untitled')} ({t}) ---\n"
                                    if t == "chart":
                                        context += f"Chart: {d.get('chart_type','Plot')} on {d.get('dataset','')}\n\n"
                                    elif t in ("analysis", "insight", "note"):
                                        context += f"{d.get('content', '')}\n\n"
                                    elif t == "code":
                                        context += f"Code: {d.get('code', '')}\n\n"
                                    elif t == "kpi":
                                        context += f"KPI — {d.get('label','')}: {d.get('value','')} (Δ {d.get('delta','')})\n\n"
                                    elif t == "link":
                                        context += f"External link: {d.get('url','')} — {d.get('description','')}\n\n"
                                prompt = (
                                    "You are a Chief Data Officer. Write an Executive Summary based "
                                    "on the following pinned data insights.\n\n" + context +
                                    "\nFormat with: 1. High-Level Overview  2. Key Findings  "
                                    "3. Strategic Recommendations\n\n"
                                    "Use Markdown. Do not hallucinate data not present in the insights."
                                )
                                resp = client.chat.completions.create(
                                    model="llama-3.3-70b-versatile",
                                    messages=[{"role": "user", "content": prompt}],
                                    max_tokens=2048,
                                )
                                st.session_state._executive_summary = resp.choices[0].message.content
                            except Exception as e:
                                st.error(f"Error generating summary: {e}")

            # AI summary display
            if st.session_state.get("_executive_summary"):
                with st.expander("📑 AI Executive Summary", expanded=True):
                    st.markdown(st.session_state._executive_summary)
                    if st.button("🗑️ Clear Summary", key="clear_ai_summary"):
                        del st.session_state._executive_summary
                        st.rerun()

            st.divider()

            # ── Tag filter ─────────────────────────────────────────────────
            all_tags = sorted({
                tag
                for item in all_artifacts
                for tag in item.get("data", {}).get("tags", [])
            })

            filtered_artifacts = all_artifacts
            if all_tags:
                selected_tags = st.multiselect(
                    "🏷️ Filter by tag",
                    options=all_tags,
                    default=[],
                    key="report_tag_filter",
                    placeholder="Show all items, or pick tags to filter…",
                )
                if selected_tags:
                    filtered_artifacts = [
                        item for item in all_artifacts
                        if any(t in item.get("data", {}).get("tags", []) for t in selected_tags)
                    ]
                    st.caption(f"Showing {len(filtered_artifacts)} of {len(all_artifacts)} items matching selected tags.")

            # ── Render each artifact ───────────────────────────────────────
            total = len(filtered_artifacts)
            for idx, item in enumerate(filtered_artifacts):
                _render_artifact(item, idx, state_mgr, total)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — ADD CONTENT
    # ══════════════════════════════════════════════════════════════════════════
    with tab_add:

        # ── Notes ─────────────────────────────────────────────────────────────
        st.markdown("### 📝 Add a Note")
        st.caption("Write observations, external findings, or any context you want to remember. Supports **Markdown**.")

        note_title = st.text_input("Note title", placeholder="e.g. Market research finding", key="note_title_input")
        note_text  = st.text_area(
            "Note content",
            height=160,
            placeholder=(
                "Write anything here — findings from Google, observations, conclusions...\n\n"
                "Tip: use **bold**, *italic*, `code`, and bullet lists."
            ),
            key="note_text_input",
        )
        note_tags  = _tags_input("note_tags_input")

        if st.button("➕ Add Note", key="btn_add_note", type="primary"):
            if note_text.strip():
                state_mgr.pin_item(
                    "note",
                    {"content": note_text.strip(), "tags": note_tags},
                    note_title.strip() or f"Note — {datetime.datetime.now().strftime('%H:%M')}",
                )
                st.success("📝 Note added to Report Board!")
                st.rerun()
            else:
                st.warning("Write something in the note first.")

        st.divider()

        # ── Images ────────────────────────────────────────────────────────────
        st.markdown("### 🖼️ Add an Image or Screenshot")
        st.caption("Upload a screenshot from your browser, a chart from an external tool, or any image.")

        img_title   = st.text_input("Image title", placeholder="e.g. Google Trends screenshot", key="img_title_input")
        img_caption = st.text_input("Caption (optional)", placeholder="Brief description of what this shows", key="img_caption_input")
        img_tags    = _tags_input("img_tags_input")
        uploaded    = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg", "webp", "gif"],
            key="img_uploader",
        )

        if uploaded is not None:
            st.image(uploaded, caption="Preview", use_column_width=True)
            if st.button("➕ Add Image to Report", key="btn_add_image", type="primary"):
                img_bytes = uploaded.read()
                img_b64   = base64.b64encode(img_bytes).decode()
                state_mgr.pin_item(
                    "image",
                    {
                        "base64":   img_b64,
                        "filename": uploaded.name,
                        "mime":     uploaded.type,
                        "caption":  img_caption.strip(),
                        "tags":     img_tags,
                    },
                    img_title.strip() or uploaded.name,
                )
                st.success("🖼️ Image added to Report Board!")
                st.rerun()

        st.divider()

        # ── Links ─────────────────────────────────────────────────────────────
        st.markdown("### 🔗 Add an External Link")
        st.caption("Save a reference to an article, report, dashboard, or any external source.")

        lc1, lc2 = st.columns(2)
        link_url   = lc1.text_input("URL", placeholder="https://...", key="link_url_input")
        link_title = lc2.text_input("Link title", placeholder="e.g. Competitor analysis article", key="link_title_input")
        link_desc  = st.text_input("Brief description", placeholder="What this link contains or why it's relevant", key="link_desc_input")
        link_tags  = _tags_input("link_tags_input")

        if st.button("➕ Add Link", key="btn_add_link", type="primary"):
            if link_url.strip():
                state_mgr.pin_item(
                    "link",
                    {"url": link_url.strip(), "description": link_desc.strip(), "tags": link_tags},
                    link_title.strip() or link_url.strip(),
                )
                st.success("🔗 Link added to Report Board!")
                st.rerun()
            else:
                st.warning("Please enter a URL.")

        st.divider()

        # ── KPI Cards ─────────────────────────────────────────────────────────
        st.markdown("### 📈 Add a KPI / Metric Card")
        st.caption("Record a key number from an external source — e.g. from a Google report or industry benchmark.")

        kc1, kc2, kc3 = st.columns(3)
        kpi_label = kc1.text_input("Metric name",      placeholder="e.g. Market share",   key="kpi_label_input")
        kpi_value = kc2.text_input("Value",             placeholder="e.g. 23.4%",          key="kpi_value_input")
        kpi_delta = kc3.text_input("Change (optional)", placeholder="e.g. +2.1% YoY",     key="kpi_delta_input")
        kpi_note  = st.text_input("Source / context",   placeholder="e.g. Statista 2024", key="kpi_note_input")
        kpi_tags  = _tags_input("kpi_tags_input")

        if st.button("➕ Add KPI Card", key="btn_add_kpi", type="primary"):
            if kpi_label.strip() and kpi_value.strip():
                state_mgr.pin_item(
                    "kpi",
                    {
                        "label": kpi_label.strip(),
                        "value": kpi_value.strip(),
                        "delta": kpi_delta.strip(),
                        "note":  kpi_note.strip(),
                        "tags":  kpi_tags,
                    },
                    f"KPI: {kpi_label.strip()}",
                )
                st.success("📈 KPI card added to Report Board!")
                st.rerun()
            else:
                st.warning("Metric name and value are required.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — EXPORT
    # ══════════════════════════════════════════════════════════════════════════
    with tab_export:
        st.markdown("### 📤 Export Your Report")

        st.info(
            "💡 **Both formats open directly in Microsoft Word:**\n\n"
            "- **HTML** → File → Open → select the `.html` file. Word preserves all headings, "
            "bold, links, and layout. Then **Save As .docx** and export to PDF.\n\n"
            "- **Markdown** → Word 2019+ opens `.md` files directly, or paste the content "
            "into any Markdown editor. Free tool [Pandoc](https://pandoc.org) converts "
            "`.md → .docx` with one terminal command: `pandoc report.md -o report.docx`"
        )

        if not all_artifacts:
            st.warning("Your report is empty. Add content from the Report Board or Add Content tabs first.")
        else:
            md_content   = _build_markdown(all_artifacts)
            html_content = _build_html(md_content)
            ts_str       = datetime.datetime.now().strftime("%Y%m%d_%H%M")

            ec1, ec2 = st.columns(2)
            ec1.download_button(
                label="⬇️ Download as HTML (.html)",
                data=html_content,
                file_name=f"udip_report_{ts_str}.html",
                mime="text/html",
                width="stretch",
                help="Open in Microsoft Word → Save As .docx → Export to PDF.",
            )
            ec2.download_button(
                label="⬇️ Download as Markdown (.md)",
                data=md_content,
                file_name=f"udip_report_{ts_str}.md",
                mime="text/markdown",
                width="stretch",
                help="Word 2019+ opens .md directly. Also works with Pandoc.",
            )

            st.divider()
            st.markdown("### Preview (Markdown)")
            st.markdown(md_content)
