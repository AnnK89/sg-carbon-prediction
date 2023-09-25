##################  VARIABLES  ##################
PROJECT_ID = 'le-wagon-bootcamp-396204'
DATASET_ID = 'sgcarbonprediction'

# Electricity : Input value (in KWh/Yr) X 0.85 (Emission Factor) =  Output value in (Kg of CO2)
ELEC_EMISSION_FACTOR = 0.85

# Petrol: Input Value(In Litres/Yr) X 2.296(Emission Factor) =  Output value in (Kg of CO2)
# Diesel: Input Value(In Litres/Yr) X 2.653 (Emission Factor) =  Output value in (Kg of CO2)
# LPG: Input Value(In Kg/Yr) X 2.983 (Emission Factor) =  Output value in (Kg of CO2)

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
