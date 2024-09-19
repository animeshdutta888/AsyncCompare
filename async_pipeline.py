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
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
import os

# Define semaphore limit to control concurrency
SEMAPHORE_LIMIT = 10
semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
# Define ThreadPoolExecutor to limit number of worker threads
executor = ThreadPoolExecutor(max_workers=8)  
# Define asynchronous data fetching function
async def fetch_data_async(url: str, session: aiohttp.ClientSession) -> pd.DataFrame:
    async with session.get(url, ssl=False) as response:
        text = await response.text()
        return pd.read_csv(StringIO(text))

# Define an asynchronous preprocessing function
async def async_preprocess_chunk(chunk: pd.DataFrame, preprocessor: ColumnTransformer) -> Tuple[np.ndarray, pd.Series]:
    X_chunk = chunk.drop('isFraud', axis=1)
    y_chunk = chunk['isFraud']
    X_chunk_preprocessed = await asyncio.get_event_loop().run_in_executor(
        executor, lambda: preprocessor.transform(X_chunk).toarray()
    )
    return X_chunk_preprocessed, y_chunk

# Define an asynchronous partial training function
async def async_partial_fit_chunk(X_chunk: np.ndarray, y_chunk: pd.Series, model: SGDClassifier, classes: np.ndarray):
    await asyncio.get_event_loop().run_in_executor(
        executor, lambda: model.partial_fit(X_chunk, y_chunk, classes=classes)
    )

# Combined function to fetch, preprocess, and train chunks concurrently with semaphore
async def fetch_preprocess_train_chunks(urls: List[str], preprocessor: ColumnTransformer, model: SGDClassifier, classes: np.ndarray):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            # Fetch data asynchronously
            fetch_task = asyncio.create_task(fetch_data_async(url, session))
            
            # Preprocess and train on the fetched data concurrently
            async def preprocess_and_train(fetch_task):
                async with semaphore:  # Control concurrency
                    df_chunk = await fetch_task
                    X_chunk_preprocessed, y_chunk = await async_preprocess_chunk(df_chunk, preprocessor)
                    await async_partial_fit_chunk(X_chunk_preprocessed, y_chunk, model, classes)
                    del df_chunk  # Clear memory to avoid memory leaks
            
            tasks.append(preprocess_and_train(fetch_task))
        
        # Run all fetch, preprocess, and train tasks concurrently
        await asyncio.gather(*tasks)

# Asynchronous prediction function
async def async_predict_chunk(chunk: np.ndarray, model: SGDClassifier) -> np.ndarray:
    return await asyncio.get_event_loop().run_in_executor(None, lambda: model.predict(chunk))

async def async_pipeline():
    urls = [
        'http://raw.githubusercontent.com/animeshdutta888/AsyncCompare/main/split_1.csv',
        'http://raw.githubusercontent.com/animeshdutta888/AsyncCompare/main/split_2.csv',
        'http://raw.githubusercontent.com/animeshdutta888/AsyncCompare/main/split_3.csv'
    ]
    
    start_time_async = time.time()
    
    # Fetch the first chunk to initialize preprocessing
    async with aiohttp.ClientSession() as session:
        initial_df = await fetch_data_async(urls[0], session)
        initial_df = initial_df.drop(columns=['isFlaggedFraud'])

    # Identify columns and set up preprocessor
    numeric_cols = initial_df.select_dtypes(include=['number']).columns.drop('isFraud')
    categorical_cols = initial_df.select_dtypes(exclude=['number']).columns

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
        
    # Get all unique classes from the initial chunk
    all_classes = np.array([0, 1])

    # Fit the preprocessor on the initial data chunk
    X_initial = initial_df.drop('isFraud', axis=1)
    preprocessor.fit(X_initial)

    # Initialize the model for incremental learning
    clf = SGDClassifier(random_state=42)

    # Fetch, preprocess, and train chunks concurrently
    await fetch_preprocess_train_chunks(urls, preprocessor, clf, all_classes)

    # After training, prepare test data
    test_df = initial_df.iloc[9000:10000]
    test_chunks = [test_df.iloc[i:i + 1000] for i in range(0, len(test_df), 1000)]  # Adjust chunk size as needed

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
