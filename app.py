import streamlit as st
from sentence_transformers import SentenceTransformer, util
import re
import PyPDF2
import io
import plotly.graph_objects as go
import csv
from datetime import datetime

# Graceful import for mic recorder
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Exam Evaluator Pro",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.main { background-color: #0e1117; }

.stTextArea textarea {
    border-radius: 8px;
    border: 1px solid #3d444d;
    background-color: #161b22;
    color: #c9d1d9;
    font-size: 0.95rem;
}

.stButton button {
    width: 100%;
    border-radius: 6px;
    background: #1f6feb;
    color: white;
    font-weight: 600;
    font-size: 16px;
    height: 3em;
    border: none;
    letter-spacing: 0.03em;
    transition: background 0.2s ease;
}
.stButton button:hover {
    background: #388bfd;
}

[data-testid="stMetricValue"] {
    font-size: 36px;
    color: #58a6ff;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    font-size: 0.8rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Keyword badge */
.kw-badge {
    display: inline-block;
    background: #161b22;
    color: #f85149;
    padding: 3px 10px;
    border-radius: 4px;
    margin: 3px;
    border: 1px solid #3d1a1a;
    font-size: 0.82rem;
    font-family: monospace;
    font-weight: 600;
}

/* Tip card */
.tip-card {
    background: #161b22;
    border-left: 3px solid #58a6ff;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #c9d1d9;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* Highlight box — subtle green bg for matched words */
.highlight-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 18px 22px;
    line-height: 2.0;
    font-size: 0.97rem;
    color: #c9d1d9;
}
.hl-match {
    background-color: rgba(46, 160, 67, 0.18);
    color: #3fb950;
    border-radius: 3px;
    padding: 1px 4px;
    font-weight: 600;
}
.hl-filler { color: #484f58; }
.hl-normal { color: #c9d1d9; }

/* Section header rule */
.section-rule {
    border: none;
    border-top: 1px solid #21262d;
    margin: 20px 0 14px 0;
}

/* Verdict banners */
.verdict-expert {
    background: #0d2b0d;
    border: 1px solid #238636;
    border-radius: 6px;
    padding: 14px 20px;
    color: #3fb950;
    font-size: 1rem;
    font-weight: 600;
}
.verdict-average {
    background: #2b2000;
    border: 1px solid #9e6a03;
    border-radius: 6px;
    padding: 14px 20px;
    color: #d29922;
    font-size: 1rem;
    font-weight: 600;
}
.verdict-low {
    background: #2d0f0f;
    border: 1px solid #6e1a1a;
    border-radius: 6px;
    padding: 14px 20px;
    color: #f85149;
    font-size: 1rem;
    font-weight: 600;
}

/* Download button */
div[data-testid="stDownloadButton"] button {
    background: #238636 !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    height: 2.6em !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SUBJECT CONFIG
# ---------------------------------------------------------------------------
SUBJECT_CONFIG = {
    "Technical / Science": {
        "expert": 82, "average": 58,
        "description": "Strict evaluation — precise terminology matters.",
        "icon": "Science"
    },
    "Theory / Arts": {
        "expert": 72, "average": 48,
        "description": "Moderate evaluation — conceptual understanding is key.",
        "icon": "Arts"
    },
    "Coding": {
        "expert": 78, "average": 55,
        "description": "Logic and correctness focused evaluation.",
        "icon": "Coding"
    },
}

FILLER_WORDS = {
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they',
    'is', 'a', 'the', 'and', 'it', 'for', 'used', 'in', 'of', 'to',
    'are', 'was', 'were', 'an', 'be', 'by', 'on', 'at', 'as', 'or',
    'that', 'this', 'with', 'from', 'has', 'have', 'had', 'not', 'but',
    'so', 'just', 'very', 'also', 'then', 'than', 'too', 'its', 'into',
    'about', 'like', 'well', 'even', 'here', 'there', 'when', 'where',
    'which', 'who', 'will', 'would', 'could', 'should', 'may', 'might',
    'do', 'did', 'does', 'been', 'being', 'get', 'got', 'let', 'now'
}

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
if "score_history" not in st.session_state:
    st.session_state.score_history = []
if "spoken_answer" not in st.session_state:
    st.session_state["spoken_answer"] = ""
if "viva_error" not in st.session_state:
    st.session_state["viva_error"] = ""

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def extract_keywords(text: str) -> set:
    words = re.findall(r'\w+', text.lower())
    return {w for w in words if w not in FILLER_WORDS and len(w) > 2}


def extract_text_from_pdf(uploaded_file) -> str:
    try:
        file_bytes = uploaded_file.read()
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)
        if total_pages == 0:
            st.warning("The PDF appears to have no pages.")
            return ""
        bar = st.progress(0, text="Reading PDF...")
        parts = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text()
            if txt:
                parts.append(txt)
            bar.progress(
                int((i + 1) / total_pages * 100),
                text=f"Processing page {i + 1} of {total_pages}..."
            )
        bar.empty()
        return "\n".join(parts).strip()
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""


def compute_metrics(user_ans: str, ref_ans: str, raw_score: float) -> dict:
    """
    Derive four radar chart metrics from the answers.
    All values are 0-100.
    """
    ref_keys  = extract_keywords(ref_ans)
    user_keys = extract_keywords(user_ans)

    # Semantic Accuracy — direct from model similarity
    semantic_accuracy = round(raw_score, 1)

    # Keyword Coverage — % of reference keywords present in user answer
    if ref_keys:
        kw_coverage = round(len(ref_keys & user_keys) / len(ref_keys) * 100, 1)
    else:
        kw_coverage = 100.0

    # Sentence Clarity — penalise very short or run-on sentences
    sentences = [s.strip() for s in re.split(r'[.!?]', user_ans) if s.strip()]
    if sentences:
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        # Ideal sentence length 10-25 words; score drops outside that range
        if 10 <= avg_len <= 25:
            clarity = 90.0
        elif avg_len < 5:
            clarity = 30.0
        elif avg_len > 40:
            clarity = 50.0
        else:
            clarity = round(max(40, 90 - abs(avg_len - 17) * 2), 1)
    else:
        clarity = 20.0

    # Concept Depth — ratio of user word count to reference word count, capped at 100
    user_wc = len(user_ans.split())
    ref_wc  = len(ref_ans.split())
    depth   = round(min(user_wc / ref_wc * 100, 100), 1) if ref_wc > 0 else 50.0

    return {
        "Semantic Accuracy": semantic_accuracy,
        "Keyword Coverage":  kw_coverage,
        "Sentence Clarity":  clarity,
        "Concept Depth":     depth,
    }


def build_gauge_chart(score: float, subject: str) -> go.Figure:
    cfg = SUBJECT_CONFIG[subject]
    expert_thresh = cfg["expert"]
    avg_thresh    = cfg["average"]
    bar_color = (
        "#3fb950" if score >= expert_thresh
        else ("#d29922" if score >= avg_thresh else "#f85149")
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={"suffix": "%", "font": {"size": 40, "color": "#e6edf3"}},
        delta={
            "reference": expert_thresh,
            "increasing": {"color": "#3fb950"},
            "decreasing": {"color": "#f85149"}
        },
        title={"text": "Readiness Score", "font": {"size": 16, "color": "#8b949e"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": "#8b949e",
                "tickfont": {"color": "#8b949e", "size": 11}
            },
            "bar": {"color": bar_color, "thickness": 0.22},
            "bgcolor": "#161b22",
            "borderwidth": 0,
            "steps": [
                {"range": [0, avg_thresh],            "color": "#1a0d0d"},
                {"range": [avg_thresh, expert_thresh], "color": "#1a1500"},
                {"range": [expert_thresh, 100],        "color": "#0d1f0d"},
            ],
            "threshold": {
                "line": {"color": "#8b949e", "width": 2},
                "thickness": 0.7,
                "value": expert_thresh
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117",
        font_color="#8b949e",
        height=290,
        margin=dict(t=40, b=10, l=20, r=20)
    )
    return fig


def build_wordcount_bar(user_ans: str, ref_ans: str) -> go.Figure:
    user_wc = len(user_ans.split())
    ref_wc  = len(ref_ans.split())
    fig = go.Figure(data=[
        go.Bar(
            name="Reference", x=["Word Count"], y=[ref_wc],
            marker_color="#388bfd", text=[ref_wc], textposition="outside",
            textfont={"color": "#c9d1d9"}
        ),
        go.Bar(
            name="Your Answer", x=["Word Count"], y=[user_wc],
            marker_color="#f85149", text=[user_wc], textposition="outside",
            textfont={"color": "#c9d1d9"}
        ),
    ])
    fig.update_layout(
        barmode="group",
        title={"text": "Word Count Comparison", "font": {"color": "#8b949e", "size": 14}},
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#8b949e", height=280,
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(bgcolor="#161b22", font={"color": "#c9d1d9"}),
        yaxis=dict(gridcolor="#21262d", color="#8b949e"),
        xaxis=dict(color="#8b949e")
    )
    return fig


def build_radar_chart(metrics: dict) -> go.Figure:
    categories = list(metrics.keys())
    values     = list(metrics.values())
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed     = values     + [values[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(56, 139, 253, 0.12)",
        line=dict(color="#388bfd", width=2),
        marker=dict(color="#388bfd", size=6),
        name="Your Performance"
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont={"color": "#8b949e", "size": 10},
                gridcolor="#21262d",
                linecolor="#30363d",
                tickvals=[20, 40, 60, 80, 100]
            ),
            angularaxis=dict(
                tickfont={"color": "#c9d1d9", "size": 12},
                gridcolor="#21262d",
                linecolor="#30363d"
            )
        ),
        paper_bgcolor="#0d1117",
        font_color="#c9d1d9",
        title={"text": "Performance Radar", "font": {"color": "#8b949e", "size": 14}},
        height=320,
        margin=dict(t=50, b=20, l=40, r=40),
        showlegend=False
    )
    return fig


def build_trend_chart(history: list) -> go.Figure:
    labels = [h[0] for h in history]
    scores = [h[1] for h in history]
    fig = go.Figure(go.Scatter(
        x=labels, y=scores,
        mode="lines+markers+text",
        text=[f"{s:.1f}%" for s in scores],
        textposition="top center",
        textfont={"color": "#c9d1d9", "size": 11},
        line=dict(color="#388bfd", width=2),
        marker=dict(size=8, color="#388bfd", line=dict(color="#e6edf3", width=1.5))
    ))
    fig.update_layout(
        title={"text": "Improvement Trend — Last 3 Attempts",
               "font": {"color": "#8b949e", "size": 14}},
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#8b949e", height=250,
        margin=dict(t=50, b=20, l=20, r=20),
        yaxis=dict(range=[0, 105], gridcolor="#21262d", color="#8b949e"),
        xaxis=dict(gridcolor="#21262d", color="#8b949e")
    )
    return fig


def highlight_answer(user_ans: str, ref_keys: set) -> str:
    """
    Render user answer HTML with:
      - subtle green background highlight for matched keywords
      - dimmed colour for filler words
      - normal colour for other content words
    """
    tokens = re.findall(r'\w+|\W+', user_ans)
    parts  = []
    for token in tokens:
        lower = token.lower().strip()
        if not re.match(r'\w+', token):
            parts.append(token)
        elif lower in ref_keys:
            parts.append(f"<span class='hl-match'>{token}</span>")
        elif lower in FILLER_WORDS or len(lower) <= 2:
            parts.append(f"<span class='hl-filler'>{token}</span>")
        else:
            parts.append(f"<span class='hl-normal'>{token}</span>")
    return "".join(parts)


def depth_suggestions(user_ans: str, ref_ans: str, subject: str) -> list:
    tips   = []
    u_wc   = len(user_ans.split())
    r_wc   = len(ref_ans.split())
    ratio  = u_wc / r_wc if r_wc > 0 else 1.0

    if ratio < 0.4:
        tips.append(
            f"Answer length is significantly below the reference (~{u_wc} vs ~{r_wc} words). "
            "Expand your response with definitions, examples, or supporting evidence."
        )
    elif ratio < 0.7:
        tips.append(
            f"Your answer covers approximately {int(ratio*100)}% of the expected depth. "
            "Add more detail to strengthen your response."
        )

    subject_tips = {
        "Technical / Science": (
            "Include quantitative values, units, or formula references "
            "to demonstrate technical precision."
        ),
        "Theory / Arts": (
            "Cite relevant authors, movements, or real-world examples "
            "to support your argument."
        ),
        "Coding": (
            "Address time and space complexity, edge cases, "
            "and any relevant design patterns."
        ),
    }
    tips.append(subject_tips.get(subject, ""))

    if not [t for t in tips if t]:
        tips = ["Answer length and structure are well-proportioned. Focus on terminology precision."]

    return [t for t in tips if t]


def build_csv_report(
    subject: str,
    score: float,
    verdict: str,
    metrics: dict,
    missing_kw: set,
    tips: list,
    user_ans: str
) -> str:
    """Return a CSV string of the evaluation report."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Field", "Value"])
    writer.writerow(["Date", datetime.now().strftime("%Y-%m-%d %H:%M")])
    writer.writerow(["Subject", subject])
    writer.writerow(["Readiness Score (%)", f"{score:.1f}"])
    writer.writerow(["Verdict", verdict])
    writer.writerow([])

    writer.writerow(["Metric", "Score (%)"])
    for metric, val in metrics.items():
        writer.writerow([metric, f"{val:.1f}"])
    writer.writerow([])

    writer.writerow(["Missing Keywords"])
    if missing_kw:
        for kw in sorted(missing_kw):
            writer.writerow([kw])
    else:
        writer.writerow(["None — full coverage"])
    writer.writerow([])

    writer.writerow(["Improvement Suggestions"])
    for tip in tips:
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', tip)
        writer.writerow([clean])
    writer.writerow([])

    writer.writerow(["Your Answer"])
    writer.writerow([user_ans])

    return output.getvalue()

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### AI Exam Evaluator")
    st.markdown("---")

    subject = st.selectbox(
        "Subject Category",
        options=list(SUBJECT_CONFIG.keys()),
        help="Adjusts evaluation strictness based on subject type."
    )
    cfg = SUBJECT_CONFIG[subject]
    st.caption(cfg["description"])

    st.markdown("---")
    st.markdown(
        "**Score Thresholds**\n"
        f"- Expert  : >= {cfg['expert']}%\n"
        f"- Average : >= {cfg['average']}%\n"
        f"- Low     : < {cfg['average']}%"
    )

    if st.session_state.score_history:
        st.markdown("---")
        st.markdown("**Session History**")
        for label, sc in st.session_state.score_history:
            st.markdown(f"- {label}: `{sc:.1f}%`")
        if st.button("Clear History"):
            st.session_state.score_history = []
            st.rerun()

    st.markdown("---")
    st.caption("AI Evaluator Pro v5.0")

# ---------------------------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------------------------
st.title("AI Exam Readiness Analyzer")
st.caption("Semantic evaluation powered by Sentence Transformers")
st.markdown("---")

# ---------------------------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Reference Answer")
    ref_input_mode = st.radio(
        "Input method:",
        options=["Type / Paste Text", "Upload PDF"],
        horizontal=True,
        label_visibility="collapsed"
    )
    ref_ans = ""
    if ref_input_mode == "Upload PDF":
        uploaded_pdf = st.file_uploader(
            "Upload a PDF containing the reference answer:",
            type=["pdf"],
            help="Text is extracted automatically from all pages."
        )
        if uploaded_pdf is not None:
            ref_ans = extract_text_from_pdf(uploaded_pdf)
            if ref_ans:
                st.success(f"PDF loaded — {len(ref_ans):,} characters extracted.")
                with st.expander("Preview extracted text"):
                    st.write(ref_ans[:1200] + ("..." if len(ref_ans) > 1200 else ""))
            else:
                st.warning("No readable text found in the PDF.")
    else:
        ref_ans = st.text_area(
            "Paste the standard textbook answer here:",
            height=220,
            placeholder="Enter the ideal definition or explanation..."
        )
    if ref_ans:
        st.caption(f"Reference word count: {len(ref_ans.split())}")

with col2:
    st.subheader("Your Response")

    viva_tab, text_tab = st.tabs(["Viva Mode (Speak)", "Type Answer"])

    with viva_tab:
        if not MIC_AVAILABLE:
            st.error(
                "streamlit-mic-recorder is not installed.\n\n"
                "Run: `pip install streamlit-mic-recorder` then restart the app."
            )
        else:
            st.markdown(
                "<div class='tip-card'>"
                "<strong>How to use Viva Mode:</strong><br>"
                "1. Click Start Recording below.<br>"
                "2. Allow microphone access when prompted by your browser.<br>"
                "3. Speak your answer clearly, then click Stop Recording.<br>"
                "4. The transcribed text will appear in the editable box below."
                "</div>",
                unsafe_allow_html=True
            )
            with st.expander("Microphone not working? See browser fix"):
                st.markdown(
                    "**Chrome:** Lock icon in address bar > Site settings > Microphone > Allow\n\n"
                    "**Firefox:** Microphone icon in address bar > Allow\n\n"
                    "**Edge:** Lock icon > Permissions for this site > Microphone > Allow\n\n"
                    "Refresh the page after changing permissions."
                )
            audio = mic_recorder(
                start_prompt="Start Recording",
                stop_prompt="Stop Recording",
                just_once=True,
                use_container_width=True,
                key="mic_recorder"
            )
            if audio is not None:
                transcribed = audio.get("text", "").strip()
                if transcribed:
                    st.session_state["spoken_answer"] = transcribed
                    st.session_state["viva_error"] = ""
                    st.success("Speech captured. Review and edit in the box below.")
                else:
                    st.session_state["viva_error"] = (
                        "Speech transcription failed. Possible causes:\n"
                        "- Microphone permission denied\n"
                        "- Browser does not support Web Speech API (use Chrome or Edge)\n\n"
                        "Fix: Allow microphone access in browser settings, then refresh."
                    )
            if st.session_state["viva_error"]:
                st.warning(st.session_state["viva_error"])

    with text_tab:
        st.caption("Type or paste your answer below.")

    user_ans = st.text_area(
        "Your Answer (editable):",
        value=st.session_state["spoken_answer"],
        height=200,
        placeholder="Your answer will appear here after speaking, or type directly...",
        key="final_answer_box"
    )
    st.session_state["spoken_answer"] = user_ans

    if user_ans:
        st.caption(f"Word count: {len(user_ans.split())}")

    if st.session_state["spoken_answer"]:
        if st.button("Clear Answer"):
            st.session_state["spoken_answer"] = ""
            st.session_state["viva_error"] = ""
            st.rerun()

# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------
if st.button("Run Evaluation"):
    if ref_ans.strip() and user_ans.strip():

        # ── Staged progress indicators ────────────────────────────────────
        stage_bar = st.progress(0, text="Initialising evaluation pipeline...")

        stage_bar.progress(15, text="Extracting keywords from reference...")
        ref_keys  = extract_keywords(ref_ans)

        stage_bar.progress(30, text="Extracting keywords from your answer...")
        user_keys = extract_keywords(user_ans)
        missing   = ref_keys - user_keys

        stage_bar.progress(50, text="Calculating semantic similarity...")
        ref_emb   = model.encode(ref_ans)
        user_emb  = model.encode(user_ans)
        raw_score = util.cos_sim(ref_emb, user_emb).item() * 100

        stage_bar.progress(70, text="Analysing sentence clarity and concept depth...")
        metrics = compute_metrics(user_ans, ref_ans, raw_score)

        stage_bar.progress(85, text="Generating improvement recommendations...")
        tips = depth_suggestions(user_ans, ref_ans, subject)

        stage_bar.progress(100, text="Evaluation complete.")
        stage_bar.empty()

        # Verdict
        expert_thresh = cfg["expert"]
        avg_thresh    = cfg["average"]
        if raw_score >= expert_thresh:
            verdict = "EXPERT"
        elif raw_score >= avg_thresh:
            verdict = "AVERAGE"
        else:
            verdict = "NEEDS IMPROVEMENT"

        # Session history
        attempt_num = len(st.session_state.score_history) + 1
        st.session_state.score_history.append((f"Attempt {attempt_num}", raw_score))
        if len(st.session_state.score_history) > 3:
            st.session_state.score_history = st.session_state.score_history[-3:]

        st.markdown("---")

        # ── Verdict banner ────────────────────────────────────────────────
        verdict_class = {
            "EXPERT":            "verdict-expert",
            "AVERAGE":           "verdict-average",
            "NEEDS IMPROVEMENT": "verdict-low",
        }[verdict]

        verdict_text = {
            "EXPERT":            f"Verdict: EXPERT  |  Score: {raw_score:.1f}%  |  Strong command of the subject demonstrated.",
            "AVERAGE":           f"Verdict: AVERAGE  |  Score: {raw_score:.1f}%  |  Core understanding present. Improve technical precision.",
            "NEEDS IMPROVEMENT": f"Verdict: NEEDS IMPROVEMENT  |  Score: {raw_score:.1f}%  |  Significant gaps detected. Review the reference material.",
        }[verdict]

        st.markdown(
            f"<div class='{verdict_class}'>{verdict_text}</div>",
            unsafe_allow_html=True
        )

        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

        # ── Gauge + Radar ─────────────────────────────────────────────────
        gauge_col, radar_col = st.columns(2)
        with gauge_col:
            st.plotly_chart(
                build_gauge_chart(raw_score, subject),
                use_container_width=True
            )
        with radar_col:
            st.plotly_chart(
                build_radar_chart(metrics),
                use_container_width=True
            )

        # ── Metric row ────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Semantic Accuracy",  f"{metrics['Semantic Accuracy']:.1f}%")
        m2.metric("Keyword Coverage",   f"{metrics['Keyword Coverage']:.1f}%")
        m3.metric("Sentence Clarity",   f"{metrics['Sentence Clarity']:.1f}%")
        m4.metric("Concept Depth",      f"{metrics['Concept Depth']:.1f}%")

        # ── Word count bar ────────────────────────────────────────────────
        st.plotly_chart(
            build_wordcount_bar(user_ans, ref_ans),
            use_container_width=True
        )

        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

        # ── Improvement Trend ─────────────────────────────────────────────
        if len(st.session_state.score_history) >= 1:
            st.subheader("Improvement Trend")
            st.plotly_chart(
                build_trend_chart(st.session_state.score_history),
                use_container_width=True
            )
            st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

        # ── Answer Highlighter ────────────────────────────────────────────
        st.subheader("Answer Analysis")
        st.caption(
            "Green highlight = matched reference keyword  |  "
            "Dimmed = filler word  |  Normal = content word"
        )
        highlighted_html = highlight_answer(user_ans, ref_keys)
        st.markdown(
            f"<div class='highlight-box'>{highlighted_html}</div>",
            unsafe_allow_html=True
        )

        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

        # ── Missing Keywords ──────────────────────────────────────────────
        st.subheader("Missing Keywords")
        if missing:
            badge_html = "".join(
                f"<span class='kw-badge'>{w}</span>" for w in sorted(missing)
            )
            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown(
                "<div class='tip-card' style='margin-top:10px;'>"
                "Including these terms will align your answer more closely "
                "with the marking scheme."
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='tip-card' style='border-left-color:#3fb950;'>"
                "Full keyword coverage — all essential terms are present."
                "</div>",
                unsafe_allow_html=True
            )

        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

        # ── How to Improve ────────────────────────────────────────────────
        st.subheader("Recommendations")
        for tip in tips:
            st.markdown(f"<div class='tip-card'>{tip}</div>", unsafe_allow_html=True)

        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

        # ── Export ────────────────────────────────────────────────────────
        st.subheader("Export Results")
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            csv_data = build_csv_report(
                subject, raw_score, verdict, metrics, missing, tips, user_ans
            )
            st.download_button(
                label="Download Analysis Report (.csv)",
                data=csv_data,
                file_name=f"exam_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with exp_col2:
            # Plain text report as fallback / additional export
            txt_lines = [
                "AI EXAM READINESS ANALYZER — RESULT REPORT",
                "=" * 50,
                f"Date    : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"Subject : {subject}",
                f"Score   : {raw_score:.1f}%",
                f"Verdict : {verdict}",
                "",
                "METRICS",
            ]
            for k, v in metrics.items():
                txt_lines.append(f"  {k}: {v:.1f}%")
            txt_lines += [
                "",
                "MISSING KEYWORDS",
                (", ".join(sorted(missing)) if missing else "None"),
                "",
                "RECOMMENDATIONS",
            ]
            for tip in tips:
                txt_lines.append(f"  - {re.sub(r'\\*\\*(.+?)\\*\\*', r'\\1', tip)}")
            txt_lines += ["", "YOUR ANSWER", user_ans]

            st.download_button(
                label="Download Text Report (.txt)",
                data="\n".join(txt_lines),
                file_name=f"exam_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    else:
        st.warning("Provide both a reference answer and your response to run the evaluation.")

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("AI Evaluator Pro v5.0  |  Python · Streamlit · Sentence-Transformers · Plotly")
