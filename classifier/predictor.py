import torch

class Predictor():

    def __init__(self, model, vectorizer, mlb, lookup):
        self.vectorizer = vectorizer
        self.mlb = mlb
        self.lookup = lookup
        self.model = model

    def predict(self, text, threshold=0.1):

        vec = self.vectorizer.transform([text]).toarray()
        input_tensor = torch.tensor(vec, dtype=torch.float32)
        with torch.no_grad():
            # Extract model output (logits).
            logits = self.model(input_tensor)
            # Perform sigmoid for the model output.
            probs = torch.sigmoid(logits).numpy()[0]

        # Get predicted codes from lookup table.
        prediction_pairs = [(self.mlb.classes_[i], p) for i, p in enumerate(probs) if p >= threshold]
        prediction_pairs = sorted(prediction_pairs, key=lambda x: x[1], reverse=True)
        
        # Format text for prediction output.
        full_labels = [
            f"{code} - {self.lookup[code]['category']}, {self.lookup[code]['name']} ({p})"
            for code, p in prediction_pairs if code in self.lookup
        ]

        return full_labels