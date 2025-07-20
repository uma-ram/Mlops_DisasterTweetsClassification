import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_split_data(filepath="data/raw/train.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(filepath)
    return train_test_split(df["text"], df["target"], test_size=test_size, random_state=random_state)
    

def vectorize_text(X_train, X_test, max_features=1000):
    tfidf = TfidfVectorizer(max_features=max_features)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    return X_train_vec, X_test_vec, tfidf
