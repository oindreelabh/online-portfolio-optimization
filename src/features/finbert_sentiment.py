from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd

MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

print(model.config.id2label)
print(model.config.num_labels)

LABELS = model.config.id2label

def finbert_predict_sentiment(text: str):
    """
    Returns (sentiment_label, [prob_negative, prob_neutral, prob_positive])
    """
    if not isinstance(text, str) or not text.strip():
        return "neutral", [0.0, 1.0, 0.0]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
        label_id = int(torch.argmax(outputs.logits, dim=1))
        return LABELS[label_id], probs

def add_finbert_sentiment(df: pd.DataFrame, text_col: str, prefix: str = "sentiment"):
    # Get sentiment results
    results = df[text_col].astype(str).apply(finbert_predict_sentiment)
    # Extract labels and probabilities
    df = df.reset_index(drop=True)
    labels = results.apply(lambda x: x[0])
    probs = results.apply(lambda x: x[1]).tolist()
    probs_df = pd.DataFrame(probs, columns=[f"{prefix}_neg", f"{prefix}_neu", f"{prefix}_pos"])
    probs_df = probs_df.reset_index(drop=True)
    # Assign columns
    df[f"{prefix}_label"] = labels.values
    df[[f"{prefix}_neg", f"{prefix}_neu", f"{prefix}_pos"]] = probs_df
    df[f"{prefix}_score"] = df[f"{prefix}_pos"] - df[f"{prefix}_neg"]
    return df

