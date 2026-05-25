import streamlit as st
import random
import html

from module import MODEL_OPTIONS, predict_text

st.set_page_config(
    page_title="Tweet Safety Classifier",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');


    /* Remove Streamlit top header / deploy bar */
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0 !important;
    }

    div[data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    div[data-testid="stDecoration"] {
        display: none !important;
    }

    div[data-testid="stStatusWidget"] {
        display: none !important;
    }

    .stDeployButton {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden !important;
    }

    footer {
        visibility: hidden !important;
    }

    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f9fafb;
        color: #111827;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 860px;
    }

    .app-title {
        font-size: 2rem;
        font-weight: 600;
        color: #111827;
        letter-spacing: -0.03em;
        margin-bottom: 0.3rem;
    }

    .app-subtitle {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }

    .result-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 1.4rem 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .result-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }

    .result-class {
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: -0.03em;
        text-transform: capitalize;
        margin-bottom: 0.5rem;
    }

    .class-neither {
        color: #16a34a;
    }

    .class-offensive {
        color: #d97706;
    }

    .class-hate {
        color: #dc2626;
    }

    .status-pill {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
    }

    .pill-neither {
        color: #15803d;
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
    }

    .pill-offensive {
        color: #b45309;
        background-color: #fffbeb;
        border: 1px solid #fde68a;
    }

    .pill-hate {
        color: #b91c1c;
        background-color: #fef2f2;
        border: 1px solid #fecaca;
    }

    .metric-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-top: 0.75rem;
    }

    .metric-title {
        font-size: 0.72rem;
        color: #9ca3af;
        font-weight: 600;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .metric-value {
        font-size: 1.25rem;
        color: #111827;
        font-weight: 600;
    }

    /* Text area */
    .stTextArea textarea {
        border-radius: 14px;
        border: 1px solid #d1d5db;
        font-size: 0.95rem;
        padding: 0.9rem 1rem;
        background-color: #ffffff;
        color: #111827;
        font-family: 'Inter', sans-serif;
    }

    .stTextArea textarea::placeholder {
        color: #9ca3af;
    }

    .stTextArea textarea:focus {
        border-color: #6b7280;
        box-shadow: none;
    }

    /* Selectbox */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 14px !important;
        color: #111827 !important;
    }

    .stSelectbox div[data-baseweb="select"] span {
        color: #111827 !important;
    }

    .stSelectbox svg {
        fill: #111827 !important;
    }

    div[data-baseweb="popover"] {
        background-color: #ffffff !important;
    }

    ul[role="listbox"] {
        background-color: #ffffff !important;
        color: #111827 !important;
    }

    li[role="option"] {
        background-color: #ffffff !important;
        color: #111827 !important;
    }

    li[role="option"]:hover {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 2.9rem;
        border-radius: 12px;
        background-color: #111827;
        color: #f9fafb;
        font-weight: 600;
        font-size: 0.9rem;
        border: none;
        transition: all 0.15s ease;
        font-family: 'Inter', sans-serif;
    }

    .stButton > button:hover {
        background-color: #1f2937;
        transform: translateY(-1px);
    }

    .stButton > button:active {
        background-color: #374151;
        transform: translateY(0px);
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        background-color: #ffffff;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
    }

    /* Alert */
    div[data-testid="stAlert"] {
        background-color: #fffbeb;
        color: #92400e;
        border-radius: 12px;
        border: 1px solid #fde68a;
    }

    /* Bar chart */
    div[data-testid="stVegaLiteChart"] {
        background-color: #ffffff;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        padding: 1rem;
    }

    hr {
        border-color: #e5e7eb;
        margin-top: 1.75rem;
        margin-bottom: 1.75rem;
    }

    .footer {
        color: #9ca3af;
        text-align: center;
        font-size: 0.82rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_class_css(predicted_class: str):
    predicted_class = predicted_class.lower()

    if predicted_class == "neither":
        return "class-neither", "pill-neither", "Safe to post"

    if predicted_class == "offensive":
        return "class-offensive", "pill-offensive", "Potentially offensive"

    if predicted_class == "hate":
        return "class-hate", "pill-hate", "Unsafe content detected"

    return "", "", "Unknown"


st.markdown(
    '<div class="app-title">What do you want to share today?</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">Analyze your content posts descriptions as hate, offensive, or neither before posting.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)

selected_model = st.selectbox(
    label="Select model", options=MODEL_OPTIONS, label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="section-label">Compose Content</div>', unsafe_allow_html=True)

tweet_text = st.text_area(
    label="Tweet content",
    placeholder="What is happening?",
    height=160,
    label_visibility="collapsed",
)

char_count = len(tweet_text)


char_count = len(tweet_text)

st.markdown(
    f"""
    <div style="color:#9ca3af; font-size:0.85rem; padding-top:0.9rem; padding-bottom:0.9rem;">
        {char_count} characters
    </div>
    """,
    unsafe_allow_html=True,
)


analyze_button = st.button("Analyze Post")

if analyze_button:
    if not tweet_text.strip():
        st.warning("Please write a tweet before analyzing.")
    else:
        # inference
        output = predict_text(tweet_text, selected_model)

        predicted_class = output["label"]
        predicted_probability = output["score"]
        probabilities = output["scores"]

        class_css, pill_css, status_text = get_class_css(predicted_class)

        st.markdown(
            '<div class="app-title">Text Analysis</div>', unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Prediction Result</div>
                <div class="result-class {class_css}">{predicted_class}</div>
                <div class="status-pill {pill_css}">{status_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-title">Selected Model</div>
                    <div class="metric-value">{selected_model}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-title">Predicted Class Score</div>
                    <div class="metric-value">{predicted_probability * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-label">Probability Scores</div>',
            unsafe_allow_html=True,
        )

        for label, prob in probabilities.items():
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-title">{label}</div>
                    <div class="metric-value">{prob * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


st.markdown(
    """
    <div class="footer">
        Final Project for Natural Language Processing - Group 7 | Developed by Muhammad Nizwa, Lintang Anggowoyuono, and Jason Alvaro Gouw
    </div>
    """,
    unsafe_allow_html=True,
)
