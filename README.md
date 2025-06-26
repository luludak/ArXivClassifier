# ArXiv Paper Classifier

This is a classifier for ArXiv papers. It is trained against ~50K papers from [Kaggle](https://www.kaggle.com/datasets/barclaysav/b-interview-arxiv-dataset) and is able to classify the paper category, based on the title and/or abstract/description.

## Details
The model consists of 4 Fully-Connected Linear layers and 3 ReLU activation functions, and performs classification across 154 categories. The training process utilizes backpropagation via the `Adam` optimizer and `BCEWithLogitsLoss` criterion. And introduces early stopping when a loss larger than the already observed best loss is identified (meaning that the loss stopped minimizing).

![Model](https://github.com/user-attachments/assets/29346063-0630-48f7-bdc7-9577af460a45)

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

### Train Model
You need to donwload the dataset, and place it in the `datasets` folder.
After that, you can train the model by doing `python main.py -t <Epochs> -f <Format: (ONNX | PyTorch)>`. Once training is complete, the model will be automatically utilized for execution.

### Run Model
To load the model without training after it is training, simply do `python main.py -f <FORMAT>`. If no format is specified, a `PyTorch (.pth)` model will be loaded.
This will load the file.

### Output Format
You can output the model by specifying the `-f` parameter, specifying either `ONNX` or `PyTorch` format, e.g., `python main.py -t 100 -f ONNX`, both for training and loading purposes.
