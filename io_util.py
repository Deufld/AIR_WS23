from ast import literal_eval
import pandas as pd
from pathlib import Path


def read_csv(
    filename: str,
    relevant_columns: list[str],
    testing: bool) -> pd.DataFrame:
    """
        Args:
            filename (str): The filename of the CSV file to read
            relevant_columns (list[str]): columns to read from the csv file
            testing (bool): if true, the method will only return a portion of the entire CSV file which can then be
                            used to have faster test times
        Returns:
            data (pd.DataFrame): The data read from the csv as a DataFrame
        """

    # check if file exists
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError

    data = None
    # if no relevant columns are provided, all columns are read
    if len(relevant_columns) == 0:
        # https://stackoverflow.com/questions/23111990/pandas-dataframe-stored-list-as-string-how-to-convert-back-to-list
        data = pd.read_csv(path, encoding='latin-1', converters={'preprocessed_text': literal_eval})
    else:
        data = pd.read_csv(path, usecols=relevant_columns, encoding='latin-1', converters={'preprocessed_text': literal_eval})

    if testing:
        # only take 1/1000 of the entire data, for testing purposes only
        testing_len = int(len(data) / 1000)
        return data.head(testing_len)

    return data


def write_csv(
    filename: str,
    df: pd.DataFrame
):
    path = Path(filename)
    df.to_csv(path_or_buf=path, index=False)