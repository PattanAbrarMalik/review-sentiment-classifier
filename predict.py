import os
import config
import argparse
from Source.utils import load_file
from Source.processing import process_text
from sklearn.linear_model import LogisticRegression


def main(args):
    """
    Prediction Function
    """
    # Create vectorizer path
    vect_file = os.path.join(config.output_path, f"{args.model_name}.pkl")
    # Create model path
    model_file = os.path.join(config.output_path, f"{args.model_name}_lr.pkl")
    # Load the vectorizer
    vect = load_file(vect_file)
    # Load the model
    model = load_file(model_file)
    # Tokenize the input text
    tokens = [process_text(args.text)]
    # Vectorize the tokens
    X = vect.transform(tokens)
    # Make predictions
    pred_prob = round(model.predict_proba(X)[0,1]*100, 2)
    print(f"Text: {args.text}")
    print(f"Probability of Positive Class: {pred_prob}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Test review")
    parser.add_argument("--model_name", type=str, default="n_gram",
                        help="Input file name")
    args = parser.parse_args()
    main(args)
