import numpy as np


def apk(actual: int, predicted: str, k: int = 25) -> float:
    """APK

    :param actual: target value
    :type actual: int
    :param predicted: predicted values, space separated
    :type predicted: str
    :param k: allowed no. of values to predict, defaults to 25
    :type k: int, optional
    :return: score
    :rtype: float
    """
    if not actual:
        return 0.0

    actual = [actual]
    predicted = list(map(int, predicted.split()))

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(actual: list[int], predicted: list[str], k: int = 25) -> float:
    """MAPK

    :param actual: list of target values
    :type actual: list[int]
    :param predicted: list of predicted values, space separated
    :type predicted: list[str]
    :param k: allowed no. of values to predict, defaults to 25
    :type k: int, optional
    :return: score
    :rtype: float
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
