from sklearn.model_selection import train_test_split

def preprocess_data(df):
    # Simple preprocessing: train-test split
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)
