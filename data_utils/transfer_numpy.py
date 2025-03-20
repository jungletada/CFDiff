import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def transfer(csv_file, npy_file):
    # Read the CSV file (skip initial whitespace for safety, as the file has spaced formatting)
    df = pd.read_csv(csv_file, skipinitialspace=True)

    # Verify columns loaded
    print(df.columns.tolist())

    # Extract columns as 1D arrays (NumPy ndarrays)
    pressure_data    = df['pressure'].to_numpy()
    x_velocity_data  = df['x-velocity'].to_numpy()
    y_velocity_data  = df['y-velocity'].to_numpy()
    temperature_data = df['temperature'].to_numpy()

    # Determine grid dimensions (example calculation)
    num_columns = (df['y-coordinate'] == df['y-coordinate'].min()).sum()
    num_rows = len(df) // num_columns  # Ensuring grid dimensions are correct

    print(f"{num_rows} x {num_columns}")

    # Reshape each array into 2D (row-major order assumed)
    pressure_grid    = pressure_data[:num_rows * num_columns].reshape((num_rows, num_columns))
    x_velocity_grid  = x_velocity_data[:num_rows * num_columns].reshape((num_rows, num_columns))
    y_velocity_grid  = y_velocity_data[:num_rows * num_columns].reshape((num_rows, num_columns))
    temperature_grid = temperature_data[:num_rows * num_columns].reshape((num_rows, num_columns))

    # Check shapes to ensure correctness
    print("Pressure grid shape:", pressure_grid.shape)
    print("X-velocity grid shape:", x_velocity_grid.shape)
    
    # Save as a dictionary
    data_dict = {
        'pressure': pressure_grid,
        'x_velocity': x_velocity_grid,
        'y_velocity': y_velocity_grid,
        'temperature': temperature_grid
    }

    np.save(npy_file, data_dict)


def verify(npy_file):
    # Load to verify
    data_dict = np.load(npy_file, allow_pickle=True).item()
    print(data_dict.keys())  # Should print: dict_keys(['pressure', 'x_velocity', 'y_velocity', 'temperature'])

    # Define a function to plot heatmaps
    def plot_heatmap(data, title):
        plt.figure(figsize=(10, 1))
        sns.heatmap(data, cmap='coolwarm', annot=False)
        plt.title(title)
        plt.xlabel("X-axis (grid)")
        plt.ylabel("Y-axis (grid)")
        plt.savefig(title + '.png')

    # Iterate over each key-value pair and plot
    for key, value in data_dict.items():
        plot_heatmap(value, title=f"Heatmap of {key}")


if __name__ == '__main__':
    # transfer("data/case_data1/fluent_data_all_csv/107.csv", '107_data.npy')
    verify('107_data.npy')