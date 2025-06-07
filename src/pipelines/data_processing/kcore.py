import pandas as pd

"""
This script applies k-core filtering to a dataset of user-item interactions.
Complete the required parameters in the main section and run the script to do the filtering.
"""


def apply_k_core(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Apply k-core filtering to the DataFrame.
    This function filters the DataFrame to retain only users and items that have at least
    k interactions. It iteratively removes users and items until no further changes are made.

    Parameters:
        df (pd.DataFrame): DataFrame with user-item interactions.
        k (int): Minimum number of interactions required for users and items to be retained.

    Returns:
        pd.DataFrame: Filtered DataFrame with only users and items with at least k interactions.
    """
    while True:
        start_shape = df.shape[0]

        # Filter users with at least k interactions
        user_counts = df[0].value_counts()
        df = df[df[0].isin(user_counts[user_counts >= k].index)]

        # Filter items with at least k interactions
        item_counts = df[1].value_counts()
        df = df[df[1].isin(item_counts[item_counts >= k].index)]

        # Check if any rows were removed
        if df.shape[0] == start_shape:
            break

    return df


def main(input_file: str, output_file: str, k: int):
    """
    Main function to load data, apply k-core filtering, and save the results.

    Parameters:
        input_file (str): Path to the input file containing user-item interactions.
        output_file (str): Path to save the filtered output file.
        k (int): Minimum number of interactions required for users and items to be retained.
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, sep='\t', header=None)

    print(f"Applying {k}-core filtering...")
    filtered_df = apply_k_core(df, k)

    print(f"Saving results in {output_file}...")
    filtered_df.to_csv(output_file, sep='\t', index=False, header=False)

    print("Completed filtering:")
    print(f"- Remaining rows: {filtered_df.shape[0]}")
    print(f"- Remaining users: {filtered_df[0].nunique()}")
    print(f"- Remaining items: {filtered_df[1].nunique()}")


if __name__ == "__main__":
    ########## CONFIGURATION ##########
    k = 10
    input_filename = "data/dataset/file.txt"
    output_filename = f"data/dataset/file_{k}.txt"

    main(input_filename, output_filename, k)
