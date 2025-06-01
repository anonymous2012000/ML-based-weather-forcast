import os
import shutil


# Function made to remove quotes from strings in the csv
# Mainly made to reformat the 'MET Office Weather Data.csv' file
def remove_csv_quotes(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file_in, \
            open(output_file, 'w', encoding='utf-8') as file_out:
        for line in file_in:
            cleaned = line.replace('"', '')
            file_out.write(cleaned)


def main():
    source_dir = 'Source_data'
    temp_dir = 'CSV_temp_hold'
    output_dir = 'CSV_files'

    # Create directories if they dont exist
    print("Verifying directories...")
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    print("COMPLETE\n")

    # Moves files from 'Source data' to 'CSV_temp_hold'
    print("Copying files to project environment")
    for file in os.listdir(source_dir):
        if file.endswith('.csv'):
            src_path = os.path.join(source_dir, file)
            dest_path = os.path.join(temp_dir, file)

            # Copy the file so that the original remains in 'Source_data'
            shutil.copy2(src_path, dest_path)
    print("COMPLETE\n")

    # Formatting files and moving to final CSV directory
    print("Finalising file format...")
    for file in os.listdir(temp_dir):
        if file.endswith('.csv'):
            temp_path = os.path.join(temp_dir, file)

            # create output file of same name, suffixed with _p to show the file has been processed
            old_name, extension = os.path.splitext(file)
            new_name = f"{old_name}_p{extension}"

            output_path = os.path.join(output_dir, new_name)

            remove_csv_quotes(temp_path, output_path)

            # Delete original file from 'CSV_temp_hold'
            os.remove(temp_path)
    print("COMPLETE")
    print("Returning...")


if __name__ == '__main__':
    main()

