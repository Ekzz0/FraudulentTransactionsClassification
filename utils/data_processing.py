import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def balance_the_dataset(data, y_name):
    R = np.random.RandomState(42)

    # Resampling датасета
    yes_index = np.array(data[data[f"{y_name}"] == 1].index)
    no_index = np.array(data[data[f"{y_name}"] == 0].index)
    new_no_index = R.choice(no_index, len(yes_index))

    # Сгенерируем датасет по индексам
    data = data.loc[list(np.array(new_no_index)) + list(np.array(yes_index))]

    # Перемешаем датасет
    data = shuffle(data, random_state=41)
    data.index = range(len(data))
    return data



