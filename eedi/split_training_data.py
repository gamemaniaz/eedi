import pandas as pd
from sklearn.model_selection import train_test_split

from eedi import ORIGINAL_TRAIN_CSV, SEED, TEST_SET_CSV, TRAIN_SET_CSV


def main():
    df = pd.read_csv(ORIGINAL_TRAIN_CSV, index_col=False)
    train, test = train_test_split(
        df,
        test_size=0.1,
        random_state=SEED,
        shuffle=True,
    )
    train.to_csv(TRAIN_SET_CSV, index=False)
    test.to_csv(TEST_SET_CSV, index=False)


if __name__ == "__main__":
    main()
