import pandas as pd
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


# Read the csv files into an array of dataframes
def read_csvs():
    dataframes = []
    for file in os.listdir('CSV_files'):
        csv_path = os.path.join('CSV_files', file)
        df = pd.read_csv(csv_path)

        dataframes.append((file, df))

    return dataframes


# Perform data preprocessing operations
def data_preprocessing(dataframes):
    for i, (file, df) in enumerate(dataframes):
        # 1) Remove rows that have empty cells
        print("Deleting empty columns...")
        df.dropna(inplace=True)
        print("COMPLETE\n")

        # 2)a) 'climate_data.csv' contains two date columns: Date and Date1. These always have the same value, so deleting the 'Date1' column
        print("Formatting datetimes...")
        if 'Date1' in df.columns:
            df = df.drop('Date1', axis=1)

        # 2)b) Parse data columns to consistent format
        for data_column in ['Date', 'Formatted Date']:
            if data_column in df.columns:
                df[data_column] = pd.to_datetime(df[data_column], errors='coerce', dayfirst=True)
                df.rename(columns={data_column: 'Date'}, inplace=True)

        # 2)c) Remove columns that failed datetime parsing
        if 'Date' in df.columns:
            df.dropna(subset=['Date'], inplace=True)
        print("COMPLETE\n")

        # 3) Ensure all temperature recordings are stored in Celsius, not fahrenheit
        # Calculating celsius from farhenheit by the equation below:
        # Celsius = (Fahrenheit - 32) x (5/9)
        print("Converting to celsius...")
        fahrenheit_columns = [col for col in df.columns if '(°F)' in col]
        for col in fahrenheit_columns:
            celsius_col = col.replace('(°F)', '(°C)')
            df[celsius_col] = (df[col]-32) * (5/9)
            df.drop(columns=[col], inplace=True)
        print("COMPLETE\n")

        # 4) Ensure all windspeed recordings are stored in kmh, not mph
        # Calculating kmh from mph using the equation below
        # kmh = mph * 1.60934
        print("Converting to km/h...")
        mph_columns = [col for col in df.columns if 'mph' in col]
        for col in mph_columns:
            kmh_col = col.replace('(mph)', '(km_h)')
            df[kmh_col] = (df[col]) * 1.60934
            df.drop(columns=[col], inplace=True)
        print("COMPLETE\n")

        # 5) Round all values to 2 decimal places
        print("Rounding decimal points...")
        df = df.round(2)
        print("COMPLETE")

        # Update dataframes array with new dataframe
        print("Updating dataframes...")
        dataframes[i] = (file, df)
        print("COMPLETE\n")

        # Overwrite original CSV files
        print("Saving changes...")
        csv_path = os.path.join('CSV_files', file)
        df.to_csv(csv_path, index=False)
        print("COMPLETE")

    return


# Exploratory Data Analysis to understand the dataset
def EDA(dataframes):
    for file, df in dataframes:
        # Basic information
        print(f"\n\nDataframe {file}: {df}")
        print(f"Dataframe information:")
        df.info()
        print(f"\nDataframe summary: \n{df.describe()}")

        # Correlation heatmap
        os.makedirs('graphs', exist_ok=True)
        os.makedirs(os.path.join('graphs', 'heatmaps'), exist_ok=True)
        print("Creating correlation heatmap...")
        numeric_cols = df.select_dtypes(include=['float', 'int']).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            plt.figure(figsize=(10, 6))
            sns.heatmap(corr_matrix, cmap='coolwarm')
            plt.title(f"Correlation Heatmap for dataset: {file}")
            plt.tight_layout()
            heat_path = os.path.join('graphs', 'heatmaps', f"{file}_heatmap.png")
            plt.savefig(heat_path)
            plt.close()
        print("COMPLETE\n")

        # Boxplots for outliers
        os.makedirs(os.path.join('graphs', 'boxplots'), exist_ok=True)
        print("Creating boxplots...")
        for col in numeric_cols:
            # Replace any instances of the characters found in the below link with an underscore to comply with windows procedures
            # https://stackoverflow.com/questions/1976007/what-characters-are-forbidden-in-windows-and-linux-directory-names
            sanitised_file_paths = (col.replace('<', '_')
                                    .replace('>', '_')
                                    .replace(':', '_')
                                    .replace('"', '_')
                                    .replace('/', '_')
                                    .replace('\\', '_')
                                    .replace('|', '_')
                                    .replace('?', '_')
                                    .replace('*', '_'))

            plt.figure()
            plt.boxplot(df[col].dropna(), labels=[col])
            plt.title(f"Boxplot of dataset: {file}")
            plt.tight_layout()
            box_path = os.path.join('graphs', 'boxplots', f"{file}_{sanitised_file_paths}_boxplot.png")
            plt.savefig(box_path)
            plt.close()
        print("COMPLETE\n")


def main():
    print("Reading csv files...")
    dataframes = read_csvs()
    print(f"READING COMPLETE\nObtained dataframes: {dataframes}\n")
    print("Performing data preprocessing")
    data_preprocessing(dataframes)
    print("PREPROCESSING COMPLETE\n")
    print("Performing EDA...")
    EDA(dataframes)
    print("EDA COMPLETE\n")
    return


if __name__ == '__main__':
    main()