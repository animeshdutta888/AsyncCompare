# Asynchronous Machine Learning Pipeline

This repository contains an asynchronous machine learning pipeline implemented in Python, designed to efficiently fetch, preprocess, and train a model on a dataset of mobile money transactions. The pipeline utilizes asynchronous programming with `aiohttp` and `asyncio` for optimal performance, making it suitable for handling large datasets.

## Features

- Asynchronous data fetching from multiple CSV files hosted online.
- Concurrent preprocessing of data using scikit-learn's `ColumnTransformer`.
- Incremental model training with `SGDClassifier` for efficient learning.
- Asynchronous predictions on test data chunks.

## Requirements

- Python 3.7 or higher
- `aiohttp`
- `asyncio`
- `pandas`
- `scikit-learn`
- `numpy`

## Installation

To get started, clone this repository and install the required packages using pip:

```bash
git clone https://github.com/animeshdutta888/AsyncCompare.git
cd AsyncCompare
pip install -r requirements.txt
```

## Usage

To run the asynchronous machine learning pipeline, execute the following command in your terminal:

```bash
python async_pipeline.py
```

## Input Data

The pipeline fetches data from the following URLs:

- [split_1.csv](http://raw.githubusercontent.com/animeshdutta888/AsyncCompare/main/split_1.csv)
- [split_2.csv](http://raw.githubusercontent.com/animeshdutta888/AsyncCompare/main/split_2.csv)
- [split_3.csv](http://raw.githubusercontent.com/animeshdutta888/AsyncCompare/main/split_3.csv)

## Output

Upon execution, the script will output:

- A classification report showing precision, recall, and F1-score.
- Overall accuracy of the model on the test dataset.
- Total execution time for the asynchronous processing.

## Contribution

Feel free to contribute by forking the repository and submitting a pull request with your improvements or features!
