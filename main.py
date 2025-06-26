import onnx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn

from onnx2torch import convert
from decimal import Decimal
from classifier.papers import PapersDataset
from classifier.model import PaperClassifier
from classifier.predictor import Predictor
from classifier.trainer import Trainer

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-t", "--train", type=int, dest="train",
                help="Train the model, for the number of epochs specified.")

parser.add_argument("-f", "--format", type=str, dest="format",
                help="Specify output model format (ONNX | PyTorch). Default is PyTorch.")

args = parser.parse_args()
epochs = args.train
format = args.format

if format is None or (format != "ONNX" and format != "PyTorch"):
    format = "PyTorch"

print(f"Model Format: {format}")

taxonomy_path = "dataset/category_taxonomy.csv"
model_path = "models/paper_classifier"
dataset_path = "dataset/arxiv_data.csv"

prediction_threshold = 0.1
onnx_mode = False

# Load and create taxonomy file.
taxonomy_df = pd.read_csv(taxonomy_path)
taxonomy_lookup = {
    row["code"]: {
        "category": row["category"],
        "name": row["name"]
    }
    for _, row in taxonomy_df.iterrows()
}

# Load dataset CSV.
df = pd.read_csv(dataset_path)

# Combine title and summary.
df["text"] = df["titles"] + " " + df["summaries"]

# Convert label strings to actual lists.
df["terms"] = df["terms"].apply(eval)

# TF-IDF vectorization.
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df["text"])

# Encode target labels.
taxonomy_codes = list(taxonomy_lookup.keys())
mlb = MultiLabelBinarizer(classes=taxonomy_codes)
Y = mlb.fit_transform(df["terms"])

# Train the model if enabled.
if epochs is not None:
    print("Training model. Epochs: " + str(epochs) + ".")

    print("Input shape:", X.shape, "Output shape:", Y.shape)

    # Setup the dataset wrapper.
    dataset = PapersDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Instantiate the model.
    model = PaperClassifier(input_dim=dataset.X.shape[1], output_dim=dataset.Y.shape[1])
    
    # Prepare the backpropagation optimizer.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Instantiate the trainer.
    trainer = Trainer(model, model_path, dataloader, optimizer, criterion, X.shape[1], format)
    trainer.train(epochs=epochs)

# Load the model from ONNX.
if format == "PyTorch":
    model = torch.load(model_path + ".pth")
    model.eval()
else:
    onnx_model = onnx.load(model_path + ".onnx")
    model = convert(onnx_model)

# Prediction/inference loop.
predictor = Predictor(model, vectorizer, mlb, taxonomy_lookup)
while True:
    user_input = input("> ")

    if user_input.strip().lower() == "exit":
        break

    try:
        # Perform prediction.
        predictions = predictor.predict(user_input, prediction_threshold)
        print("Predicted topics:")
        for label in predictions:
            print("  - ", label)
    except Exception as e:
        print("Error:", e)