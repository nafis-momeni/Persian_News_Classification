# Persian News Classification with Hugging Face

This project demonstrates how to perform text classification on Persian news articles using the Hugging Face library. It utilizes the `HooshvareLab/bert-fa-base-uncased-clf-persiannews` pre-trained model for the classification task.

## Project Overview

The project consists of the following main components:

1. **Data**: The training data for the text classification model should be prepared in a CSV file with two columns: 'text' and 'label'. The 'text' column contains the Persian news articles, and the 'label' column contains the corresponding class labels. The `data_extract.ipynb` file is used to convert raw files in folders to this format.

2. **Model Training**: The project uses the Hugging Face library and the `transformers` package to train the text classification model. The `Trainer` class is utilized to handle the training loop and model optimization.

3. **Evaluation**: After training the model, it is evaluated on a separate validation dataset. The accuracy, F1 score, and a confusion matrix are computed to assess the model's performance.

4. **Model Saving and Reuse**: Once the model is trained, it can be saved to drive and later loaded for inference or further fine-tuning.

## Prerequisites

Make sure you have the following dependencies installed:

- Python 3.x
- `transformers` library (install via `pip install transformers`)
- `pandas` library (install via `pip install pandas`)
- `scikit-learn` library (install via `pip install scikit-learn`)
- `matplotlib` library (install via `pip install matplotlib`)
- `seaborn` library (install via `pip install seaborn`)

## Usage

1. **Prepare the Data**:

   - Create a CSV file containing the training data with 'text' and 'label' columns. Each row should represent a news article with its corresponding class label.
   - Ensure that the Persian news articles are properly preprocessed and cleaned. The data will be tokenized in the project, but for better performance, normalizing the data can help. The Hazm library can be used for Farsi text.

2. **Training**:

   - Set up the environment with the required dependencies.
   - Update the file paths and hyperparameters in the training script (`train_model.ipynb`) according to your specific setup.
   - Run the training script: `python train_model.ipynb`
   - The trained model will be saved to drive.

3. **Model Reuse**:
   - Set up the environment with the required dependencies.
   - Convert the input data in the described format.
   - Update the file paths and hyperparameters in the test script (`test_model.ipynb`) according to your specific setup.
   - Run the test script: `python test_model.ipynb`
   - The trained model will be used to predict labels for new data.
   - The evaluation metrics accuracy, F1 score, and confusion matrix will be created.

## Resources

- Hugging Face library: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
- `HooshvareLab/bert-fa-base-uncased-clf-persiannews` model: [https://huggingface.co/HooshvareLab/bert-fa-base-uncased-clf-persiannews](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-clf-persiannews)

## License

This project is licensed under the MIT License. See the [
