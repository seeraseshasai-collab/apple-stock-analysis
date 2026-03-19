import pandas as pd




# function to import data from csv file
def import_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error importing data: {e}")
        return None