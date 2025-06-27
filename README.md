# arXiv Paper Classifier

This is a classifier for arXiv papers. It is trained against ~50K papers from [Kaggle](https://www.kaggle.com/datasets/barclaysav/b-interview-arXiv-dataset) and is able to classify the paper category, based on the title and/or abstract/description.

## Details
The model consists of 4 Fully-Connected Linear layers, and performs classification across 154 categories. It consists of about 11M parameters. The training process utilizes backpropagation via the `Adam` optimizer and `BCEWithLogitsLoss` criterion. And introduces early stopping with `patience = 5`, when a loss larger than the already observed best loss is identified (meaning that the loss stopped minimizing).

![Model](https://github.com/user-attachments/assets/53a6c904-4fc7-4bb3-bb93-9bd0781ff41d)

## Implementation
The tool is implemented in Python, utilizing the PyTorch API, as well as the ONNX API (for the case ONNX is selected as format). We utilize TF-IDF vectorization for the tokens.

## Configuration
The system configuration is defined in `config.json`. In particular:
```
{
    "taxonomy_path": "<The dataset taxonomy relative path.>",
    "model_path": "<The base classifier model path. The extension is automatically added based on the -f parameter.>",
    "dataset_path": <The dataset relative path.>,
    "prediction_threshold": <The prediction threshold for inference.>
}
```

## Instructions

### Install
You will need Python `3.5+` and `pip`. Run `pip install -r requirements.txt`.

### Train Model
You need to donwload the dataset, and place it in the `datasets` folder.
After that, you can train the model by doing `python main.py -t <Epochs> -f <Format: (ONNX | PyTorch)>`. Once training is complete, the model will be automatically utilized for execution.

### Run Model
To load the model without training after it is training, simply do `python main.py -f <FORMAT>`. If no format is specified, a `PyTorch (.pth)` model will be loaded.
This will load the file.

### Output Format
You can output the model by specifying the `-f` parameter, specifying either `ONNX` or `PyTorch` format, e.g., `python main.py -t 100 -f ONNX`, both for training and loading purposes.
