import os
import config
import argparse
import pandas as pd
from Source.utils import save_file
from Source.model import vectorize
from Source.processing import process_text

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def train_model(X_train, X_test, y_train, y_test):
    """
    Function to train the model
    :param X_train
    :param X_test
    :param y_train
    :param y_test
    :return: trained model
    """
    model = LogisticRegression(C=config.logreg_C, max_iter=1000)
    model.fit(X_train, y_train)
    # Make train predictions
    train_pred = model.predict(X_train)
    # Make test predictions
    test_pred = model.predict(X_test)
    # Calculate train accuracy
    train_acc = round(accuracy_score(y_train, train_pred)*100,2)
    # Calculate test accuracy
    test_acc = round(accuracy_score(y_test, test_pred)*100,2)
    print(f"Train Accuracy: {train_acc}%" )
    print(f"Test Accuracy: {test_acc}%")
    return model


def main(args):
    # Create input data file path
    input_file = os.path.join(config.input_path, args.file_name)
    # Create vectorizer file path
    vect_file = os.path.join(config.output_path, f"{args.output_name}.pkl")
    # Create model file path
    model_file = os.path.join(config.output_path, f"{args.output_name}_lr.pkl")
    # Read raw data
    data = pd.read_excel(input_file)
    # Select text and label columns
    data = data[[config.text_col, config.label_col]]
    # Convert text column to a list of reviews
    reviews = list(data[config.text_col])
    # Pre-process the text data
    reviews = [process_text(r, config.stem) for r in reviews]
    # Create dependent variable
    y = data[config.label_col]
    # Vectorize the data and split data into train and test
    X_train, X_test, y_train, y_test, vectorizer = vectorize(reviews, y,
                                                             vect=args.vectorizer,
                                                             min_df=config.min_df,
                                                             ng_low=config.ng_low,
                                                             ng_high=config.ng_high,
                                                             test_size=config.test_size,
                                                             rs=config.rs)
    # Save the vectorizer
    save_file(vect_file, vectorizer)
    # Train the model
    model = train_model(X_train, X_test, y_train, y_test)
    # Save the model file
    save_file(model_file, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="Canva_reviews.xlsx",
                        help="Input file name")
    parser.add_argument("--vectorizer", type=str, default="bow",
                        help="Vectorizer, one of - 'bow', 'bowb', 'ng','tf'")
    parser.add_argument("--output_name", type=str, default="model",
                        help="Output file name")
    args = parser.parse_args()
    main(args)
