import onnx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import joblib

from onnx2torch import convert
from argparse import ArgumentParser
from pathlib import Path

from helpers.model_helper import ModelHelper
from classifier.papers import PapersDataset
from classifier.model import PaperClassifier
from classifier.predictor import Predictor
from classifier.trainer import Trainer

# Argument parsing.
parser = ArgumentParser()

parser.add_argument("-t", "--train", type=int, dest="train",
                help="Train the model, for the number of epochs specified.")

parser.add_argument("-f", "--format", type=str, dest="format",
                help="Specify output model format (ONNX | PyTorch). Default is PyTorch.")

# Create models folder.
Path("models").mkdir(parents=True, exist_ok=True)

args = parser.parse_args()
epochs = args.train
format = args.format

if format is None or (format != "ONNX" and format != "PyTorch"):
    format = "PyTorch"

print(f"Model Format: {format}")

# Load configuration.
helper = ModelHelper()
config = helper.load_config('./config.json')

taxonomy_path = config["taxonomy_path"]
model_path = config["model_path"]
dataset_path = config["dataset_path"]
vectorizer_path = config["vectorizer_path"]
binarizer_path = config["binarizer_path"]

prediction_threshold = float(config["prediction_threshold"])

# Load and create taxonomy file.
taxonomy_df = pd.read_csv(taxonomy_path)
taxonomy_lookup = {
    row["code"]: {
        "category": row["category"],
        "name": row["name"]
    }
    for _, row in taxonomy_df.iterrows()
}

# Train the model if enabled.
if epochs is not None:
    
    print("Training model. Epochs: " + str(epochs) + ".")
    
    # Load dataset CSV.
    df = pd.read_csv(dataset_path)

    # Combine title and summary.
    df["text"] = df['titles'] + " " + df['summaries']

    # Convert label strings to actual lists.
    df["terms"] = df["terms"].apply(eval)

    # TF-IDF vectorization.
    vectorizer = TfidfVectorizer(max_features=10000)

    # Encode target labels.
    taxonomy_codes = list(taxonomy_lookup.keys())
    mlb = MultiLabelBinarizer(classes=taxonomy_codes)

    X = vectorizer.fit_transform(df["text"])
    Y = mlb.fit_transform(df["terms"])

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(mlb, binarizer_path)

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
vectorizer = joblib.load(vectorizer_path)
mlb = joblib.load(binarizer_path)
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