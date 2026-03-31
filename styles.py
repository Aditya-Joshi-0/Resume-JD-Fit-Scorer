"""
────────────────────
Centralised Streamlit CSS injection.
All pages call apply_theme() once at the top.
"""

import streamlit as st


THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0d0d0f; }
h1,h2,h3 { font-family: 'Syne', sans-serif; }
 
.verdict-badge { display:inline-block;padding:0.3rem 1rem;border-radius:999px;font-size:0.8rem;font-weight:600;letter-spacing:0.05em; }
.verdict-strong   { background:#052e16;color:#4ade80;border:1px solid #166534; }
.verdict-good     { background:#042f2e;color:#2dd4bf;border:1px solid #115e59; }
.verdict-moderate { background:#2d1a0a;color:#fb923c;border:1px solid #7c2d12; }
.verdict-weak     { background:#2d0a0a;color:#f87171;border:1px solid #7f1d1d; }
 
.card {
    background:#13131d;border:1px solid #1e1e2e;border-radius:12px;
    padding:1.2rem 1.4rem;margin-bottom:0.9rem;
}
.card-title {
    font-family:'DM Mono',monospace;font-size:0.68rem;color:#6b7280;
    letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;
}
.metric-mini { background:#13131d;border:1px solid #1e1e2e;border-radius:10px;padding:0.9rem;text-align:center; }
.metric-mini-val { font-family:'DM Mono',monospace;font-size:1.35rem;color:#e2e2f0; }
.metric-mini-label { font-size:0.68rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.08em;margin-top:0.25rem; }
 
.pill-strong  { background:#052e16;color:#4ade80;padding:2px 9px;border-radius:999px;font-size:0.7rem; }
.pill-partial { background:#2d1a0a;color:#fb923c;padding:2px 9px;border-radius:999px;font-size:0.7rem; }
.pill-weak    { background:#1a1400;color:#fbbf24;padding:2px 9px;border-radius:999px;font-size:0.7rem; }
.pill-absent  { background:#2d0a0a;color:#f87171;padding:2px 9px;border-radius:999px;font-size:0.7rem; }
 
.tool-chip { display:inline-block;font-family:'DM Mono',monospace;font-size:0.7rem;padding:3px 9px;border-radius:6px;margin:2px; }
.tool-match   { background:#052e16;color:#4ade80;border:1px solid #166534; }
.tool-missing { background:#2d0a0a;color:#f87171;border:1px solid #7f1d1d; }
.tool-bonus   { background:#1e1b4b;color:#a5b4fc;border:1px solid #3730a3; }
 
.gap-block { background:#13131d;border-radius:0 10px 10px 0;padding:1rem 1.2rem;margin-bottom:0.7rem;font-size:0.87rem;line-height:1.6; }
.gap-high   { border-left:3px solid #ef4444; }
.gap-medium { border-left:3px solid #f59e0b; }
.gap-low    { border-left:3px solid #10b981; }
 
.strength-block { background:#0b1f14;border:1px solid #14532d;border-radius:10px;padding:0.9rem 1.1rem;margin-bottom:0.7rem;color:#bbf7d0;font-size:0.86rem; }
.rec-block { background:#11131f;border:1px solid #2a2d4a;border-radius:12px;padding:1.4rem;color:#c4c4d4;font-size:0.88rem;line-height:1.8; }
.divider { border:none;border-top:1px solid #1e1e2e;margin:1.5rem 0; }
 
/* tab polish */
.stTabs [data-baseweb="tab-list"] { background:#13131d;border-radius:10px;padding:4px;gap:4px; }
.stTabs [data-baseweb="tab"] { border-radius:8px;padding:0.4rem 1rem;font-size:0.85rem;color:#6b7280; }
.stTabs [aria-selected="true"] { background:#1e1e2e;color:#e2e2f0; }

/* Full-width segmented control */
div[data-testid="stSegmentedControl"] [role="radiogroup"] {
    gap: 0.4rem;
}

div[data-testid="stSegmentedControl"] [role="radio"] {
    flex: 1 1 0;
    justify-content: center;
}

div[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] {
    background: #1e1e2e !important;
    border-color: #3730a3 !important;
}
</style>
"""

def apply_theme():
    """Inject CSS theme. Call once at the top of every page."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)