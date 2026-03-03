# Hogwarts Reviews — Hard Negatives Finder
# This app demonstrates a Computational Linguistics approach to sentiment analysis.
# It combines:
# - TF-IDF vector space representation
# - Linear classification (SGD logistic regression)
# - Class imbalance handling
# - Discourse marker detection (e.g., "but", "however")
# - Mixed polarity lexical cues
# - Uncertainty-based slicing of difficult cases

import re
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "sentiment_pipeline.joblib"

# Discourse markers indicate contrast or concession.
# Example:
# "I love the world but it crashes constantly."
# In English discourse, sentiment after "but" often dominates interpretation.
DISCOURSE_MARKERS = [
    "but", "however", "although", "though", "yet",
    "nevertheless", "nonetheless", "still",
    "even though", "despite", "in spite of"
]

# Lexical sentiment cues.
# These are surface-level polarity indicators.
# Example:
# "amazing graphics" → positive cue
# "boring gameplay" → negative cue
POS_CUES = {
    "amazing", "great", "good", "love", "loved", "fun",
    "awesome", "excellent", "beautiful", "perfect",
    "enjoy", "enjoyed", "fantastic", "incredible"
}

NEG_CUES = {
    "boring", "bug", "bugs", "glitch", "glitches",
    "crash", "crashes", "crashing", "lag",
    "stutter", "stuttering", "unplayable",
    "broken", "worst", "bad", "terrible",
    "refund", "waste", "disappointing"
}

# Real-world CSV files often contain messy column names.
# For example: "_Reviews_", "review_", "Reviews"
# This function normalizes them so they map to "review".
def _normalize_colname(col: str) -> str:
    s = str(col).strip().lower()
    s = s.replace(" ", "")
    s = re.sub(r"_+", "", s)
    s = re.sub(r"[^a-z]", "", s)
    if s.endswith("s"):
        s = s[:-1]
    return s

def find_review_column(columns):
    for col in columns:
        if _normalize_colname(col) == "review":
            return col
    return None

# Basic text normalization.
# Example:
# "   AMAZING Game!!!  " → "amazing game!!!"
# This reduces superficial variation.
def normalize_text(t: str) -> str:
    t = str(t).lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t

# Counts contrastive discourse markers.
# Example:
# "I liked it but it crashes" → score = 1
# Higher scores suggest structural polarity shift.
def discourse_score(text: str) -> int:
    t = normalize_text(text)
    score = 0
    for m in DISCOURSE_MARKERS:
        score += t.count(m)
    return score

# Counts positive and negative lexical cues.
# Example:
# "Amazing world but boring combat"
# pos_ct = 1 ("amazing")
# neg_ct = 1 ("boring")
# This signals mixed polarity.
def cue_counts(text: str):
    tokens = re.findall(r"[a-z']+", normalize_text(text))
    pos = sum(1 for w in tokens if w in POS_CUES)
    neg = sum(1 for w in tokens if w in NEG_CUES)
    return pos, neg

# Loads the trained sentiment model.
# The model was trained using:
# - TF-IDF representation
# - SGDClassifier with logistic loss
# - class_weight="balanced" to correct class imbalance
def load_model():
    return joblib.load(MODEL_PATH)

# Generates predictions and probability distributions.
# Probability values allow uncertainty estimation.
# Example:
# Positive: 0.51, Negative: 0.49 → ambiguous case
def predict_with_probs(model, texts):
    probs = model.predict_proba(texts)
    classes = list(model.named_steps["clf"].classes_)
    preds = model.predict(texts)
    return preds, probs, classes

def confidence_of_pred(pred, prob_row, classes):
    prob_map = dict(zip(classes, prob_row))
    return float(prob_map.get(pred, 0.0))

st.set_page_config(page_title="Hard Negatives Finder", layout="wide")
st.title("Hogwarts Reviews — Hard Negatives Finder")
st.caption("Discourse-aware sentiment classification with uncertainty slicing.")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Run train.py first to generate sentiment_pipeline.joblib")
    st.stop()

st.sidebar.header("Settings")

# Uncertainty threshold defines what counts as a linguistically difficult case.
# Example:
# If threshold = 0.55
# Any prediction below 0.55 confidence is marked as uncertain.
uncertainty_threshold = st.sidebar.slider(
    "Uncertainty threshold",
    min_value=0.50, max_value=0.80, value=0.55, step=0.01
)

st.subheader("Single Review Analysis")
text = st.text_area("Paste a review to analyze sentiment:", height=140)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Paste a review first.")
    else:
        pred, probs, classes = predict_with_probs(model, [text])
        pred = pred[0]
        conf = confidence_of_pred(pred, probs[0], classes)
        disc = discourse_score(text)
        pos_ct, neg_ct = cue_counts(text)

        st.write("Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sentiment", pred)
        c2.metric("Confidence", f"{conf:.2f}")
        c3.metric("Discourse markers", disc)
        c4.metric("Lexical polarity (pos/neg)", f"{pos_ct}/{neg_ct}")

        # Hard case detection logic
        # A review is difficult if:
        # - model confidence is low
        # - discourse contrast appears
        # - both positive and negative cues appear
        flags = []
        if conf < uncertainty_threshold:
            flags.append("UNCERTAIN CLASSIFICATION")
        if disc > 0:
            flags.append("CONTRASTIVE STRUCTURE")
        if pos_ct > 0 and neg_ct > 0:
            flags.append("MIXED POLARITY")

        if flags:
            st.warning("Hard-case signals: " + " | ".join(flags))
        else:
            st.success("No hard-case signals detected.")

        st.write("Probability distribution:")
        st.json({cls: float(p) for cls, p in zip(classes, probs[0])})

st.subheader("Batch Analysis (Upload CSV with 'Review' column)")
upload = st.file_uploader("Upload CSV", type=["csv"])

if upload:
    df = pd.read_csv(upload)

    review_col = find_review_column(df.columns)
    if review_col is None:
        st.error(f"Review column not found. Columns: {list(df.columns)}")
        st.stop()

    df = df.rename(columns={review_col: "Review"})
    df["Review"] = df["Review"].fillna("").astype(str)

    preds, probs, classes = predict_with_probs(model, df["Review"].tolist())
    df["Predicted_Sentiment"] = preds
    df["Confidence"] = [
        confidence_of_pred(p, pr, classes) for p, pr in zip(preds, probs)
    ]

    df["DiscourseMarkers"] = df["Review"].apply(discourse_score)
    df["PosCues"], df["NegCues"] = zip(*df["Review"].apply(cue_counts))

    df["UNCERTAIN"] = df["Confidence"] < uncertainty_threshold
    df["CONTRASTIVE"] = df["DiscourseMarkers"] > 0
    df["MIXED_CUES"] = (df["PosCues"] > 0) & (df["NegCues"] > 0)

    # Hard-negative candidate example:
    # "Great graphics but constant crashes"
    # Predicted Positive
    # Contains negative cue + discourse marker
    df["HARD_NEGATIVE_CANDIDATE"] = (
        (df["Predicted_Sentiment"] == "Positive") &
        (df["NegCues"] > 0) &
        (df["CONTRASTIVE"] | df["UNCERTAIN"] | df["MIXED_CUES"])
    )

    st.write("Corpus-level statistics")
    k1, k2, k3 = st.columns(3)
    k1.metric("Uncertain cases", int(df["UNCERTAIN"].sum()))
    k2.metric("Contrastive reviews", int(df["CONTRASTIVE"].sum()))
    k3.metric("Hard-negative candidates", int(df["HARD_NEGATIVE_CANDIDATE"].sum()))

    st.write("Confidence distribution")
    st.bar_chart(df["Confidence"].round(2).value_counts().sort_index())

    st.write("Hard Negative Candidates")
    st.dataframe(
        df[df["HARD_NEGATIVE_CANDIDATE"]]
        .sort_values("Confidence")
        .head(100),
        use_container_width=True
    )

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download enriched CSV",
        data=csv_bytes,
        file_name="hard_negative_analysis.csv",
        mime="text/csv"
    )