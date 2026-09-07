"""
Conspiracy Radar — Streamlit demo app.

Paste any text and get a breakdown of conspiracy-style rhetorical
patterns detected in it, plus a blended rule-based + ML score.

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
from detector import analyze, label_for_score, get_model

st.set_page_config(page_title="Conspiracy Radar", page_icon="🕵️", layout="centered")

st.title("🕵️ Conspiracy Radar")
st.caption(
    "Paste any piece of text and see how many conspiracy-style rhetorical "
    "patterns it uses — hidden-agent framing, us-vs-them language, false "
    "certainty, and more. Built to be explainable, not a black box."
)

model_status = "✅ Loaded" if get_model() is not None else "⚠️ Not trained yet — run train_model.py"
st.sidebar.markdown("### About")
st.sidebar.write(
    "This tool blends a rule-based linguistic pattern detector with a "
    "TF-IDF + Logistic Regression classifier trained on a labeled starter "
    "dataset. Score reflects *rhetorical style*, not factual accuracy — "
    "a text can be true and still use persuasive rhetorical patterns, "
    "or false and use neutral language."
)
st.sidebar.markdown(f"**Model status:** {model_status}")

example_texts = {
    "-- pick an example --": "",
    "Neutral news snippet": "The city council approved funding for two new public libraries after a review of community feedback.",
    "Conspiracy-style snippet": "They don't want you to know this, but nothing that happens is random. Wake up and connect the dots before it's too late.",
}

choice = st.selectbox("Try an example, or paste your own text below:", list(example_texts.keys()))
default_text = example_texts[choice]

text = st.text_area("Text to analyze", value=default_text, height=160, placeholder="Paste a paragraph, post, or article snippet here...")

analyze_clicked = st.button("Analyze", type="primary")

if analyze_clicked and text.strip():
    result = analyze(text)
    score = result["final_score"]

    st.subheader("Result")
    st.progress(min(max(score, 0.0), 1.0))
    st.markdown(f"**Score: {score:.2f} / 1.00** — {label_for_score(score)}")

    col1, col2 = st.columns(2)
    col1.metric("Rule-based signal", f"{result['rule_score']:.2f}")
    col2.metric("ML classifier signal", f"{result['ml_score']:.2f}" if result["ml_score"] is not None else "N/A")

    st.subheader("Pattern breakdown")
    cat_scores = result["category_scores"]
    if any(v > 0 for v in cat_scores.values()):
        df = pd.DataFrame(
            {"Pattern category": list(cat_scores.keys()), "Matches": list(cat_scores.values())}
        ).set_index("Pattern category")
        st.bar_chart(df)

        st.subheader("Matched phrases")
        for category, phrases in result["breakdown"].items():
            if phrases:
                st.markdown(f"**{category}**")
                for p in set(phrases):
                    st.markdown(f"- _\"{p}\"_")
    else:
        st.info("No strong rule-based rhetorical patterns detected in this text.")

elif analyze_clicked:
    st.warning("Paste some text first!")

st.divider()
st.caption(
    "Built by Rithya · Educational project analyzing rhetorical style, not a "
    "fact-checker. Always verify claims through credible sources."
)
