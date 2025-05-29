import pandas as pd


def count_user_item_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of interactions for each user and item in the DataFrame and
    establishing it as the rating column. The timestamp is the one of the first
    user-item interaction.

    Parameters:
        df (pd.DataFrame): DataFrame with user-item interactions.

    Returns:
        pd.DataFrame: DataFrame with user and item counts.
    """
    # Rename columns for clarity
    df.columns = ['user', 'item', 'rating', 'timestamp', 'local_timestamp']

    # Set the rating as the number of interactions
    df['rating'] = df.groupby(['user', 'item'])['timestamp'].transform('count')
    # Set the timestamp as the first interaction
    df['timestamp'] = df.groupby(['user', 'item'])['timestamp'].transform('min')
    df['local_timestamp'] = df.groupby(['user', 'item'])['local_timestamp'].transform('min')
    # Drop duplicates
    df = df.drop_duplicates(subset=['user', 'item'])
    # Reset index
    df = df.reset_index(drop=True)
    return df[['user', 'item', 'rating', 'timestamp']]

def main(input_file, output_file):
    print(f"Loading data frmo {input_file}...")
    df = pd.read_csv(input_file, sep='\t', header=None)

    print("Counting user-item interactions...")
    filtered_df = count_user_item_interactions(df)

    print(f"Saving results in {output_file}...")
    filtered_df.to_csv(output_file, sep='\t', index=False, header=False)

    print(f"Completed filtering. Remaining rows: {filtered_df.shape[0]}")

if __name__ == "__main__":
    input_filename = "data/NewYork/US_NewYork_Processed_Shortened_10.txt"
    output_filename = "data/NewYork/US_NewYork_Processed_Shortened_Processed.txt"

    main(input_filename, output_filename)
