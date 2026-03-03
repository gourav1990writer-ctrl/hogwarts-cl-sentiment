# Discourse-Aware Sentiment Analysis of Game Reviews (Streamlit)

This project is a sentiment analysis on game reviews.
It goes beyond “Positive/Negative” by surfacing **hard cases** using discourse markers, mixed polarity cues, and model uncertainty.

# What the app does

# 1) Sentiment classification (TF-IDF + SGD)
- **Feature:** Predict review sentiment (Positive/Negative) with probability.
- **Implemented by functions:**
  - `load_model()` → loads the trained TF-IDF + SGD pipeline from `sentiment_pipeline.joblib`
  - `predict_with_probs(model, texts)` → returns predictions + `predict_proba` distributions
  - `confidence_of_pred(pred, prob_row, classes)` → extracts the confidence for the predicted label

Example:
- Input: `"This game is boring and full of bugs"`
- Output: `Negative` with a probability score

# 2) Discourse marker detection (contrast / concession)
- **CL motivation:** Contrastive markers often signal a polarity shift.
  - Example: `"I love the game, **but** it crashes constantly."`
- **Implemented by functions:**
  - `normalize_text(text)` → lowercases and normalizes whitespace
  - `discourse_score(text)` → counts occurrences of discourse markers (e.g., “but”, “however”)

# 3) Lexical polarity cue detection (mixed sentiment language)
- **CL motivation:** Reviews often contain **mixed polarity** (positive + negative words).
  - Example: `"Amazing world but boring combat"` → positive cue + negative cue
- **Implemented by functions:**
  - `cue_counts(text)` → tokenizes and counts words from `POS_CUES` and `NEG_CUES`

# 4) Uncertainty-based slicing (finding ambiguous predictions)
- **CL motivation:** Low-confidence predictions often represent linguistically hard cases (sarcasm, mixed cues, implicit sentiment).
  - Example: `Positive: 0.51 vs Negative: 0.49` → uncertain
- **Implemented by functions:**
  - `predict_with_probs(...)` + `confidence_of_pred(...)`
- **UI logic:** reviews are flagged as uncertain if:
  - `Confidence < uncertainty_threshold` (slider in the sidebar)

# 5) “Hard Negative Candidate” extraction
- **Goal:** Find reviews predicted **Positive** but showing negative signals (useful for error analysis / moderation / QA).
  - Example: `"Great game but constant crashes"` (often misclassified if text starts positive)
- **Implemented by code logic in batch mode:**
  - `df["HARD_NEGATIVE_CANDIDATE"] = (...)`
  - Uses:
    - `NegCues > 0`
    - and at least one of: `CONTRASTIVE`, `UNCERTAIN`, `MIXED_CUES`

# 6) Robust CSV column handling (real-world messy datasets)
- **Goal:** Accept review columns with names like:
  - `_review`, `review_`, `Reviews`, `_reviews_`, `_Reviews_`, `_review_`
- **Implemented by functions:**
  - `_normalize_colname(col)` → removes underscores, lowercases, removes plural “s”
  - `find_review_column(columns)` → locates a review-text column even with messy naming

Files
- `train.py` → trains and saves the sentiment pipeline as `sentiment_pipeline.joblib`
- `app.py` → Streamlit interface for single review + batch review analysis
- `requirements.txt` → dependencies

## How to run locally
```bash
pip install -r requirements.txt
python3 train.py
streamlit run app.py
