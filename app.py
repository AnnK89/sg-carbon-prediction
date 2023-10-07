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
df['plan_area'] = df['plan_area'].apply(lambda x: x.upper())

areas_map = json.load(open("MasterPlan2019PlanningAreaBoundaryNoSea.geojson","r"))

fig=px.choropleth(df,
                  geojson=areas_map,
                  featureidkey='properties.Name',
                  locations='plan_area',
                  animation_frame = 'year',
                  color='carbon_total',
                  color_continuous_scale=[[0, 'rgb(240,240,240,0.6)'], 
                                          [0.5, 'rgb(255,0,0)'],
                                          [0.75, 'rgb(200,0,0)'],
                                          [0.80, 'rgb(150,0,0)'],
                                          [0.95, 'rgb(100,0,0)'],
                                          [1, 'rgb(50,0,0)']], 
                  range_color=(df['carbon_total'].min(), df['carbon_total'].max()),
                  )
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(height=500, width=800)

st.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme="streamlit")
