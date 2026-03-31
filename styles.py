"""
────────────────────
Centralised Streamlit CSS injection.
All pages call apply_theme() once at the top.
"""

import streamlit as st


THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp { background: #0d0d0f; }

h1, h2, h3 { font-family: 'Syne', sans-serif; }

.score-hero {
    background: linear-gradient(135deg, #1a1a24 0%, #12121a 100%);
    border: 1px solid #2a2a3a;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.score-number {
    font-family: 'DM Mono', monospace;
    font-size: 5rem;
    font-weight: 500;
    line-height: 1;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.score-label {
    color: #6b7280;
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.verdict-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-top: 1rem;
}
.verdict-strong   { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.verdict-good     { background: #042f2e; color: #2dd4bf; border: 1px solid #115e59; }
.verdict-moderate { background: #2d1a0a; color: #fb923c; border: 1px solid #7c2d12; }
.verdict-weak     { background: #2d0a0a; color: #f87171; border: 1px solid #7f1d1d; }

.section-card {
    background: #13131d;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.section-card-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #6b7280;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.section-score-bar {
    height: 4px;
    border-radius: 2px;
    margin: 0.6rem 0;
}

.pill-strong { background: #052e16; color: #4ade80; padding: 2px 10px; border-radius: 999px; font-size: 0.72rem; }
.pill-partial { background: #2d1a0a; color: #fb923c; padding: 2px 10px; border-radius: 999px; font-size: 0.72rem; }
.pill-weak    { background: #2d0a0a; color: #f87171; padding: 2px 10px; border-radius: 999px; font-size: 0.72rem; }

.insight-block {
    background: #13131d;
    border-left: 3px solid #7c3aed;
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    color: #c4c4d4;
    font-size: 0.9rem;
    line-height: 1.7;
}

.gap-high   { border-left-color: #ef4444; }
.gap-medium { border-left-color: #f59e0b; }
.gap-low    { border-left-color: #10b981; }

.strength-block {
    background: #0b1f14;
    border: 1px solid #14532d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    color: #bbf7d0;
    font-size: 0.88rem;
}

.tool-chip {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 6px;
    margin: 3px;
}
.tool-match   { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.tool-missing { background: #2d0a0a; color: #f87171; border: 1px solid #7f1d1d; }
.tool-bonus   { background: #1e1b4b; color: #a5b4fc; border: 1px solid #3730a3; }

.metric-mini {
    background: #13131d;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-mini-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.4rem;
    color: #e2e2f0;
}
.metric-mini-label {
    font-size: 0.7rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

.divider { border: none; border-top: 1px solid #1e1e2e; margin: 2rem 0; }

.rec-block {
    background: #11131f;
    border: 1px solid #2a2d4a;
    border-radius: 12px;
    padding: 1.5rem;
    color: #c4c4d4;
    font-size: 0.9rem;
    line-height: 1.8;
}
</style>
"""

def apply_theme():
    """Inject CSS theme. Call once at the top of every page."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)