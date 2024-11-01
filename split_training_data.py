import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    random_seed = 20241101
    df = pd.read_csv("data/train.csv", index_col=False)
    train, test = train_test_split(
        df,
        test_size=0.1,
        random_state=random_seed,
        shuffle=True,
    )
    train.to_csv("data/train_data.csv", index=False)
    test.to_csv("data/test_data.csv", index=False)


if __name__ == "__main__":
    main()
