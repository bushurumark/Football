import logging

def determine_final_prediction(pred, probs):
    outcome_labels = ["Home Team Win", "Draw", "Away Team Win"]
    # If pred is already a string outcome, return it
    if isinstance(pred, str):
        if pred in outcome_labels:
            return pred
        try:
            pred_val = float(pred)
        except Exception:
            return "Home Team Win"  # fallback
    else:
        pred_val = pred
    try:
        pred_val = float(pred_val)
    except Exception:
        return "Home Team Win"  # fallback
    if 0.5 <= pred_val <= 1.4:
        model_outcome = "Home Team Win"
        class_label = 1
    elif 1.5 <= pred_val <= 2.4:
        model_outcome = "Draw"
        class_label = 2
    elif 2.5 <= pred_val <= 3.4:
        model_outcome = "Away Team Win"
        class_label = 3
    else:
        return "Home Team Win"  # fallback
    # Map class labels to outcome strings
    label_to_outcome = {1: "Home Team Win", 2: "Draw", 3: "Away Team Win"}
    # Convert all keys in probs to int if possible
    int_probs = {}
    for k, v in (probs or {}).items():
        try:
            int_k = int(k)
            int_probs[int_k] = float(v)
        except Exception:
            continue
    if not int_probs:
        return "Home Team Win"  # fallback
    # Find the class with the highest probability
    highest_label = max(int_probs, key=int_probs.get)
    highest_outcome = label_to_outcome.get(highest_label, "Home Team Win")
    if model_outcome == highest_outcome:
        return model_outcome
    tied = [label_to_outcome[k] for k, v in int_probs.items() if v == int_probs[highest_label] and k in label_to_outcome]
    if len(tied) > 1:
        return f"{model_outcome} or {tied[1]}" if tied[1] != model_outcome else f"{tied[0]} or {model_outcome}"
    return model_outcome 