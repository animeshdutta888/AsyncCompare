import aiohttp
import asyncio
import pandas as pd
from io import StringIO
from sklearn.linear_model import SGDClassifier  # Incremental learning model
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
import time
import numpy as np
from typing import Tuple, List

# Define asynchronous data fetching function
async def fetch_data_async(url: str, session: aiohttp.ClientSession) -> pd.DataFrame:
    async with session.get(url, ssl=False) as response:
        text = await response.text()
        return pd.read_csv(StringIO(text))

# Fetch all data asynchronously
async def fetch_all_data(urls: List[str]) -> pd.DataFrame:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_data_async(url, session) for url in urls]
        results = await asyncio.gather(*tasks)
        return pd.concat(results, ignore_index=True)

# Define an asynchronous preprocessing function
async def async_preprocess_chunk(chunk: pd.DataFrame, preprocessor: ColumnTransformer) -> Tuple[np.ndarray, pd.Series]:
    X_chunk = chunk.drop('isFraud', axis=1)
    y_chunk = chunk['isFraud']
    X_chunk_preprocessed = await asyncio.get_event_loop().run_in_executor(
        None, lambda: preprocessor.transform(X_chunk).toarray()
    )
    return X_chunk_preprocessed, y_chunk

# Define an asynchronous partial training function with consistent class labels
async def async_partial_fit_chunk(X_chunk: np.ndarray, y_chunk: pd.Series, model: SGDClassifier, classes: np.ndarray):
    await asyncio.get_event_loop().run_in_executor(
        None, lambda: model.partial_fit(X_chunk, y_chunk, classes=classes)
    )

# Define an asynchronous prediction function
async def async_predict_chunk(chunk: np.ndarray, model: SGDClassifier) -> np.ndarray:
    return await asyncio.get_event_loop().run_in_executor(None, lambda: model.predict(chunk))

async def async_pipeline():
    urls = [
        'http://raw.githubusercontent.com/animeshdutta888/AsyncCompare/main/split_1.csv',
        'http://raw.githubusercontent.com/animeshdutta888/AsyncCompare/main/split_2.csv',
        'http://raw.githubusercontent.com/animeshdutta888/AsyncCompare/main/split_3.csv'
    ]
    
    start_time_async = time.time()
    
    # Fetch and concatenate data asynchronously
    df = await fetch_all_data(urls)
    print("Dataframe shape: ",df.shape)
    print(df.head())
    print(df.columns.tolist())
    train_df = df.iloc[:25000]
    test_df = df.iloc[25000:30000]

    numeric_cols = train_df.select_dtypes(include=['number']).columns.drop('isFraud')
    categorical_cols = train_df.select_dtypes(exclude=['number']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Fit preprocessor on the entire dataset
    X_all = df.drop('isFraud', axis=1)
    preprocessor.fit(X_all)

    # Get unique classes from the training data
    unique_classes = np.unique(train_df['isFraud'])

    # Split data into chunks for async processing
    chunk_size = 1000
    train_chunks = [train_df.iloc[i:i + chunk_size] for i in range(0, len(train_df), chunk_size)]
    test_chunks = [test_df.iloc[i:i + chunk_size] for i in range(0, len(test_df), chunk_size)]

    # Process and train on each chunk concurrently
    train_tasks = [async_preprocess_chunk(chunk, preprocessor) for chunk in train_chunks]
    train_preprocessed_chunks = await asyncio.gather(*train_tasks)

    # Initialize the model for incremental learning
    clf = SGDClassifier(random_state=42)

    # Perform asynchronous partial fit on each chunk with consistent class labels
    train_fit_tasks = [async_partial_fit_chunk(X, y, clf, unique_classes) for X, y in train_preprocessed_chunks]
    await asyncio.gather(*train_fit_tasks)

    # Preprocess test chunks asynchronously
    test_tasks = [async_preprocess_chunk(chunk, preprocessor) for chunk in test_chunks]
    test_preprocessed_chunks = await asyncio.gather(*test_tasks)

    # Convert processed test chunks
    X_test_preprocessed = np.vstack([x for x, _ in test_preprocessed_chunks])
    y_test = pd.concat([y for _, y in test_preprocessed_chunks], ignore_index=True)

    # Predict asynchronously in chunks
    test_chunks_split = np.array_split(X_test_preprocessed, 10)  # Adjust chunk size as needed
    predict_tasks = [async_predict_chunk(chunk, clf) for chunk in test_chunks_split]
    y_pred_chunks = await asyncio.gather(*predict_tasks)

    # Concatenate prediction results
    y_pred = np.concatenate(y_pred_chunks)

    end_time_async = time.time()

    # Print results
    print("Asynchronous Classification Report:\n", classification_report(y_test, y_pred))
    print("Asynchronous Accuracy:", accuracy_score(y_test, y_pred))
    print(f"Asynchronous Execution Time: {end_time_async - start_time_async} seconds")

def main():
    asyncio.run(async_pipeline())

if __name__ == "__main__":
    main()
