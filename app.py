import streamlit as st
import plotly.express as px
import pandas as pd
import json

st.write('Carbon Emission of Different Areas in Singapore')

df =pd.read_csv('CO2_2015_2021.csv')
areas_map = json.load(open("MasterPlan2019PlanningAreaBoundaryNoSea.geojson","r"))

fig=px.choropleth(df, 
                  geojson=areas_map, 
                  featureidkey='properties.Name', 
                  locations='plan_area', 
                  color='2015_CO2_ton', 
                  color_continuous_scale='Inferno'
                  )
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(height=500, width=800)

st.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme="streamlit")
