# FUNCTIONS DEFINITIONS
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error


def root_mean_squared_error(y_true, y_pred):
    """
    Function computes statistical error metric called "Root Mean Squared Error" (RMSE)
    given vector of real values and vector of predicted values.

    Args:
        y_true: (numeric values) real values
        y_pred: (numeric values) predicted values

    Returns:
        Root Mean Squared Error (RMSE)
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Function computes statistical error metric called "Mean Absolute Percentage Error" (MAPE)
    given vector of real values and vector of predicted values.

    Args:
        y_true: (numeric values) real values
        y_pred: (numeric values) predicted values

    Returns:
        Mean Absolute Percentage Error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def generate_time_series_cv_split(df_length=24 * 7, min_train_size=10, max_train_size=10,
                                  gap=0, test_size=1, verbose=False):
    """
    Returns list of tuples, where tuple[0] equals to indices assigned to train_set and tuple[1] assigned to test_set
    It will generate splits of minimal train set size equal to "min_train_size" which is incremented until reaching
    "max_train_size" with a constant gap between train and test sets equal to "gap". This process continues until dataset length is fully exploited.
    Upon assigning "max_train_size < 0" it will always increment train_set size until the maximum possible limit.

    Args:
        df_length: (integer) number of observations in the Time series. It will be used to generate output for
                             (train_set, test_set) indices tuples.
        min_train_size: (integer) minimal length of the train_set outputs.
        max_train_size:(integer) maximal length of the train_set outputs.
        gap: (integer) constant length of gap between time of the prediction and timepoint for which we are generating the prediction.
        test_size: (integer) constant length of the test_set outputs.
        verbose: (bool) when True, this function is printing consecutive train_set, test_set divisions.
                        Example: if df_length=5, min_train_size=3, max_train_size=3 and gap=1 it will print:
                        [1, 2, 3, 4, 5] [1, 2, 3] [5].

    Returns:
        List of tuples, where tuple[0] is a list of indices assigned to train_set and tuple[1] is a list of indices
        assigned to test_set in this particular split.
    """

    output = list()
    is_finished = False

    if max_train_size <= 0:
        max_train_size = df_length

    step = test_size
    window_length = min_train_size + gap + test_size
    window = [i for i in range(window_length)]

    while not is_finished:
        train_idx = window[:min_train_size]
        test_idx = window[min_train_size + gap:]

        # extended_train_idx = train_idx
        if (len(output) > 0) and (max_train_size > min_train_size):
            train_idx = list(np.unique(output[-1][0] + train_idx))
        train_idx = train_idx[-max_train_size:]

        output.append((train_idx, test_idx))
        if verbose:
            print(window)
            print(train_idx, test_idx, "\n")

        window = [idx + step for idx in window]

        if np.max(window) > (df_length - 1):
            is_finished = True

    return output


def get_holiday_flag(input_df):
    """
    This function takes as a parameter pandas.DataFrame which MUST have format pandas.DatetimeIndex (datetime.datetime).
    So far it handles year from 2014 until the end of 2019.

    Args:
        input_df: Data frame for which polish holidays' flag will be added.

    Returns:
        pandas.DataFrame of the same format as input_df with one additional column of holidays' flag.
    """

    df = input_df.copy()
    holidays_2014 = ["2014-01-01", "2014-01-06", "2014-04-20", "2014-04-21", "2014-05-01", "2014-05-03", "2014-06-08",
                     "2014-06-19", "2014-08-15", "2014-11-01", "2014-11-11", "2014-12-25", "2014-12-26"]
    holidays_2015 = ["2015-01-01", "2015-01-06", "2015-04-05", "2015-04-06", "2015-05-01", "2015-05-03", "2015-05-24",
                     "2015-06-04", "2015-08-15", "2015-11-01", "2015-11-11", "2015-12-25", "2015-12-26"]
    holidays_2016 = ["2016-01-01", "2016-01-06", "2016-03-27", "2016-03-28", "2016-05-01", "2016-05-03", "2016-05-15",
                     "2016-05-26", "2016-08-15", "2016-11-01", "2016-11-11", "2016-12-25", "2016-12-26"]
    holidays_2017 = ["2017-01-01", "2017-01-06", "2017-04-16", "2017-04-17", "2017-05-01", "2017-05-03", "2017-06-04",
                     "2017-06-15", "2017-08-15", "2017-11-01", "2017-11-11", "2017-12-25", "2017-12-26"]
    holidays_2018 = ["2018-01-01", "2018-01-06", "2018-04-01", "2018-04-02", "2018-05-01", "2018-05-03", "2018-05-20",
                     "2018-05-31", "2018-08-15", "2018-11-01", "2018-11-11", "2018-11-12", "2018-12-25", "2018-12-26"]
    holidays_2019 = ["2019-01-01", "2019-01-06", "2019-04-21", "2019-04-22", "2019-05-01", "2019-05-03", "2019-06-09",
                     "2019-06-20", "2019-08-15", "2019-11-01", "2019-11-11", "2019-12-25", "2019-12-26"]

    holiday_list = holidays_2014 + holidays_2015 + holidays_2016 + holidays_2017 + holidays_2018 + holidays_2019

    output = [False for i in range(len(df.index))]
    holiday_list_date = [datetime.strptime(x, "%Y-%m-%d") for x in holiday_list]

    for holiday_date in holiday_list_date:
        tmp = (df.index.year == holiday_date.year) & (df.index.month == holiday_date.month) & (
                df.index.day == holiday_date.day)
        output = [a or b for a, b in zip(output, tmp)]

    df["isHoliday"] = output
    return df
