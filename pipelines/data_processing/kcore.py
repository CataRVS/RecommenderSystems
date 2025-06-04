import pandas as pd


def apply_k_core(df, k):
    while True:
        start_shape = df.shape[0]

        # Filtrar usuarios con al menos k interacciones
        user_counts = df[0].value_counts()
        df = df[df[0].isin(user_counts[user_counts >= k].index)]

        # Filtrar POIs con al menos k interacciones
        poi_counts = df[1].value_counts()
        df = df[df[1].isin(poi_counts[poi_counts >= k].index)]

        # Verificar si ya no hay cambios
        if df.shape[0] == start_shape:
            break

    return df


def main(input_file, output_file, k):
    print(f"Loading data frmo {input_file}...")
    df = pd.read_csv(input_file, sep='\t', header=None)

    print(f"Applying {k}-core filtering...")
    filtered_df = apply_k_core(df, k)

    print(f"Saving results in {output_file}...")
    filtered_df.to_csv(output_file, sep='\t', index=False, header=False)

    print(f"Completed filtering. Remaining rows: {filtered_df.shape[0]}")


if __name__ == "__main__":
    input_filename = "data/NewYork/US_NewYork_Processed.txt"
    k = 5
    output_filename = f"data/NewYork/US_NewYork_Processed_Shortened_kcore_{k}.txt"

    main(input_filename, output_filename, k)
