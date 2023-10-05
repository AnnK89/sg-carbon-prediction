import streamlit as st
import plotly.express as px
import pandas as pd
from ml_logic.data import BigQueryDataLoader
import json
import requests

st.write('Carbon Emission of Different Areas in Singapore')
data = requests.get("https://carbonpredictionarbon-nmh4nih7qa-as.a.run.app/predict")
data = pd.DataFrame(json.loads(data.json()['5_years_prediction']))

df = BigQueryDataLoader.clean_pred_data(data)
areas_map = json.load(open("MasterPlan2019PlanningAreaBoundaryNoSea.geojson","r"))

fig=px.choropleth(df,
                  geojson=areas_map,
                  featureidkey='properties.Name',
                  locations='plan_area',
                  color='carbon_total',
                  color_continuous_scale='Inferno'
                  )
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(height=500, width=800)

st.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme="streamlit")
