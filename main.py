import sys
sys.path.append('..')

from ml_logic import data

def main():
    # Retrieve data from BigQuery
    processed_data = data.combine_clean_data()

    # Load the model

    # Make predictions

    # Return the results
    return(processed_data)

if __name__ == '__main__':
    main()
