##################  VARIABLES  ##################
PROJECT_ID = 'le-wagon-bootcamp-396204'
DATASET_ID = 'sgcarbonprediction'

# Electricity : Input value (in KWh/Yr) X 0.85 (Emission Factor) =  Output value in (Kg of CO2)
ELEC_EMISSION_FACTOR = 0.85

# Petrol: Input Value(In Litres/Yr) X 2.296(Emission Factor) =  Output value in (Kg of CO2)
# Diesel: Input Value(In Litres/Yr) X 2.653 (Emission Factor) =  Output value in (Kg of CO2)
# LPG: Input Value(In Kg/Yr) X 2.983 (Emission Factor) =  Output value in (Kg of CO2)
# Carbon emitted for vehicle = 4.6 tCO2 /year
# Carbon emitted for population = 0.365 tCO2 /year

# Start and end year for dataset filter
YEAR_START = 2005
YEAR_END = 2021

##################  CONSTANTS  #####################
DWELLING_TYPE_MAPPING = {
    'HDB 1- and 2-Room Flats': '1-room / 2-room',
    'HDB 3-Room Flats': '3-room',
    'HDB 4-Room Flats': '4-room',
    'HDB 5-Room and Executive Flats': '5-room and Executive',
    'Landed Properties': 'Landed Properties',
    'Condominiums and Other Apartments': 'Private Apartments and Condominiums'
}

PLANNING_AREA = ['Ang Mo Kio', 'Bedok', 'Bishan', 'Boon Lay', 'Bukit Batok',
       'Bukit Merah', 'Bukit Panjang', 'Bukit Timah',
       'Central Water Catchment', 'Changi', 'Changi Bay', 'Choa Chu Kang',
       'Clementi', 'Downtown Core', 'Geylang', 'Hougang', 'Jurong East',
       'Jurong West', 'Kallang', 'Lim Chu Kang', 'Mandai', 'Marina East',
       'Marina South', 'Marine Parade', 'Museum', 'Newton',
       'North-Eastern Islands', 'Novena', 'Orchard', 'Outram',
       'Pasir Ris', 'Paya Lebar', 'Pioneer', 'Punggol', 'Queenstown',
       'River Valley', 'Rochor', 'Seletar', 'Sembawang', 'Sengkang',
       'Serangoon', 'Simpang', 'Singapore River', 'Southern Islands',
       'Straits View', 'Sungei Kadut', 'Tampines', 'Tanglin', 'Tengah',
       'Toa Payoh', 'Tuas', 'Western Islands', 'Western Water Catchment',
       'Woodlands', 'Yishun']

TRANSFORMER_MAP = {
    'electricity': 'ElecConsumDataPreprocessingTransformer',
    'gas': 'GasConsumDataPreprocessingTransformer',
    'population': 'PopulationDataPreprocessingTransformer',
    'vehicle': 'VehicleDataPreprocessingTransformer',
}
