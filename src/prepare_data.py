import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(input_file, train_file, test_file, test_size=0.2, random_state=42):
    data = pd.read_csv(input_file)
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input dataset")
    parser.add_argument("--train", required=True, help="Path to save training dataset")
    parser.add_argument("--test", required=True, help="Path to save testing dataset")
    args = parser.parse_args()

    prepare_data(args.input, args.train, args.test)
