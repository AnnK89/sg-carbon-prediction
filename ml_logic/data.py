import params
import pandas as pd

from google.cloud import bigquery

class BigQueryDataRetriever():
    """
    Retrieves dataset from BigQuery.

    Parameters
    -------
    table_id : str
        The table ID to retrieve the data from.

    Returns
    ------
    dataframe

    """
    def retrieve_data(self, table_id):
        try:
            query = f"""
                SELECT *
                FROM {params.PROJECT_ID}.{params.DATASET_ID}.{table_id}
                """

            client = bigquery.Client()
            query_job = client.query(query)
            result = query_job.result()
            df = result.to_dataframe()
            return df
        except Exception as e:
            raise Exception(f"An error occurred while retrieving '{table_id}' dataset: {str(e)}")


# Preprocess consumption data from EMA
class ConsumDataPreprocessingTransformer():
    """
    Retrieves EMA Electricity and Gas Consumption data from BigQuery and transform into the required format.

    Parameters
    -------
    table_id : str
        The table ID to retrieve the data from.

    Returns
    ------
    dataframe

    """
    def __init__(self, table_id):
        self.table_id = table_id
        self.data = BigQueryDataRetriever().retrieve_data(self.table_id)

    def clean_data(self):
        X = self.data

        try:
            X = self.filter_data(X)
            X = self.rename_columns(X)
            X = self.convert_to_numeric(X)
            X = self.merge_with_household_data(X)
            X = self.calculate_columns(X)

            return X
        except ValueError as ve:
            raise ValueError(f"An error occurred while cleaning '{self.table_id}' dataset: {str(ve)}")

    def filter_data(self, X):
        X = X[(X['month'] == 'Annual') & (X['year'] >= params.YEAR_START) & (X['year'] <= params.YEAR_END)]
        X = X.drop(columns=['month', 'Region'])
        X = X[~(X['dwelling_type'].str.lower().str.contains('housing') | (X['dwelling_type'] == 'Overall'))]
        X['year'] = X['year'].astype(int)
        return X

    def rename_columns(self, X):
        X = X.rename(columns={'Description': 'planning_area'})
        return X

    def convert_to_numeric(self, X):
        X['kwh_per_acc'] = pd.to_numeric(X['kwh_per_acc'], errors='coerce')
        X = X[(X['kwh_per_acc'].notna()) & (X['kwh_per_acc'] > 0)]
        return X

    def merge_with_household_data(self, X):
        Household_data = HouseholdDataPreprocessingTransformer().clean_data()
        X = X.merge(Household_data, on=['dwelling_type', 'year', 'planning_area'], how='left')
        return X

    def calculate_columns(self, X):
        X['kwh_total'] = X['kwh_per_acc'] * X['household_acc']
        X['co2_emissions_ton_total'] = X['kwh_total'] * params.ELEC_EMISSION_FACTOR * 0.001
        X = X.drop(columns=['kwh_per_acc', 'household_acc', 'kwh_total'])
        X = X.groupby(['planning_area', 'year']).agg({'co2_emissions_ton_total': 'sum'}).reset_index()
        X = X.pivot(index=['planning_area'], columns='year', values='co2_emissions_ton_total').reset_index()
        X = X[~(X['planning_area'].str.lower().str.contains('region') | (X['planning_area'] == 'Overall'))]
        return X


class HouseholdDataPreprocessingTransformer():
    """
    Retrieves Datagov household data from BigQuery and transform into the required format.

    Parameters
    -------
    table_id : str
        The table ID to retrieve the data from.

    Returns
    ------
    dataframe

    """
    def __init__(self):
        self.table_id = 'household'
        self.data = BigQueryDataRetriever().retrieve_data(self.table_id)

    def clean_data(self):
        X = self.data

        try:
            X = X.rename(columns={'PA': 'planning_area', 'TOD': 'dwelling_type', 'Hse': 'household_acc', 'Time': 'year'})
            X = X[(X['year'] >= params.YEAR_START) & (X['year'] <= params.YEAR_END)]
            X = X.groupby(['planning_area', 'dwelling_type', 'year']).agg({'household_acc': 'sum'}).reset_index()
            X['dwelling_type'] = X['dwelling_type'].map(params.DWELLING_TYPE_MAPPING)
            return X
        except Exception as e:
            print(f"An error occurred while cleaning '{self.table_id}' dataset: {str(e)}")
            return None


class ElecConsumDataPreprocessingTransformer(ConsumDataPreprocessingTransformer):
    def __init__(self):
        super().__init__('electricity_consumption')


class GasConsumDataPreprocessingTransformer(ConsumDataPreprocessingTransformer):
    def __init__(self):
        super().__init__('gas_consumption')


class PopulationDataPreprocessingTransformer():
    """
    Retrieves Datagov population data from BigQuery and transform into the required format.

    Parameters
    -------
    table_id : str
        The table ID to retrieve the data from.

    Returns
    ------
    dataframe

    """
    def __init__(self):
        self.table_id = 'population'
        self.data = BigQueryDataRetriever().retrieve_data(self.table_id)

    def clean_data(self):
        X = self.data

        try:
            X = X.rename(columns={'PA': 'planning_area'})
            X = X[(X['Time'] >= params.YEAR_START) & (X['Time'] <= params.YEAR_END)]
            X = X.pivot(index='planning_area', columns='Time', values='Pop').reset_index()
            return X
        except Exception as e:
            print(f"An error occurred while cleaning '{self.table_id}' dataset: {str(e)}")
            return None


class VehicleDataPreprocessingTransformer():
    """
    Retrieves Datagov vehicle data from BigQuery and transform into the required format.

    Parameters
    -------
    table_id : str
        The table ID to retrieve the data from.

    Returns
    ------
    dataframe

    """
    def __init__(self):
        self.table_id = 'vehicle'
        self.data = BigQueryDataRetriever().retrieve_data(self.table_id)

    def clean_data(self):
        X = self.data
        try:
            X = X.rename(columns={'string_field_0': 'planning_area'})
            return X
        except Exception as e:
            print(f"An error occurred while cleaning '{self.table_id}' dataset: {str(e)}")
            return None


def combine_clean_data():
    dfs = pd.DataFrame()

    # Add electricity consumption data
    dfs = add_dataset_to_list(ElecConsumDataPreprocessingTransformer(), 'electricity', dfs)

    # Add gas consumption data
    dfs = add_dataset_to_list(GasConsumDataPreprocessingTransformer(), 'gas', dfs)

    # Add population consumption data
    dfs = add_dataset_to_list(PopulationDataPreprocessingTransformer(), 'population', dfs)

    # Add vehicle consumption data
    dfs = add_dataset_to_list(VehicleDataPreprocessingTransformer(), 'vehicle', dfs)

    return dfs



def add_dataset_to_list(transformer, data_type, dfs):
    df = transformer.clean_data()
    df['planning_area'] = df['planning_area'] + f'_{data_type}'
    return pd.concat([dfs, df])
