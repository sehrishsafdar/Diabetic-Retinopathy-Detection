# import pandas as pd
# from sklearn.model_selection import StratifiedKFold

# def split_data(csv_path, n_clients=2):
#     df = pd.read_csv(csv_path)
#     skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=42)
#     partitions = []

#     X = df.iloc[:, 0]
#     y = df.iloc[:, 1]

#     for _, test_idx in skf.split(X, y):
#         partitions.append(df.iloc[test_idx].reset_index(drop=True))
#     return partitions

import pandas as pd
from sklearn.model_selection import StratifiedKFold

def split_data(csv_path, n_clients=2):
    """
    Splits the dataset into stratified partitions for federated clients.

    Args:
        csv_path (str): Path to the CSV file containing image IDs and labels.
        n_clients (int): Number of clients to split the data for.

    Returns:
        List[pd.DataFrame]: A list of DataFrames, each for one client.
    """
    df = pd.read_csv(csv_path)

    # Sanity check
    if df.shape[1] < 2:
        raise ValueError("CSV must contain at least two columns: image_id and label.")

    # Extract image IDs and labels
    X = df.iloc[:, 0]
    y = df.iloc[:, 1]

    # Initialize stratified splitter
    skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=42)
    partitions = []

    # Perform stratified split
    for _, test_idx in skf.split(X, y):
        client_df = df.iloc[test_idx].reset_index(drop=True)
        partitions.append(client_df)

    return partitions
