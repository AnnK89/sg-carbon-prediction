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
df = BigQueryDataLoader.clean_pred_data(data).reset_index(drop=True)
df['plan_area'] = df['plan_area'].apply(lambda x : x.upper())

# Set the animation duration to control the speed of the animation
animation_duration = 2000

fig=px.choropleth(df,
                  geojson=areas_map,
                  featureidkey='properties.Name',
                  locations='plan_area',
                  animation_frame = 'year',
                  color='carbon_total',
                  color_continuous_scale=[[0, '#f0f0f0'],
                                          [0.1, '#f0f0f0'],
                                          [0.3,'#008000'],
                                            [0.5, '#ffff99'],
                                            [0.75, '#ff6600'],
                                            [0.80, '#FFC0CB'],
                                            [0.95, '#ff0000'],
                                            [1, '#800080']],
                  color_continuous_midpoint=df['carbon_total'].mean(),
                  range_color=(df['carbon_total'].min(), df['carbon_total'].max()),
                  )
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(height=500, width=800)

# Create animated bar chart
fig_bar = px.bar(df,
             x='plan_area',
             y='carbon_total',
             animation_frame='year',
             color='carbon_total',
             color_continuous_scale=[
                 [0.0, '#00ff00'],      # Green
                 [0.2, '#ffff00'],      # Yellow
                 [0.4, '#ff8000'],      # Orange
                 [0.6, '#ff4000'],      # Dark Orange
                 [0.8, '#ff0000'],      # Red
                 [1.0, '#800080']       # Purple
             ],
             color_continuous_midpoint=df['carbon_total'].mean(),
             range_color=(df['carbon_total'].min(), df['carbon_total'].max()),
             )


# Update chart layout
fig_bar.update_layout(
    height=600,  # Adjust the height as needed
    margin=dict(l=50, r=50, b=100, t=50),
)



# Streamlit widget to switch between chart types
chart_type = st.radio("Select Chart Type", ["Choropleth Map","Bar Chart"])

if chart_type == "Bar Chart":
    st.plotly_chart(fig_bar, use_container_width=False, sharing="streamlit", theme="streamlit", animation_duration=animation_duration)
elif chart_type == "Choropleth Map":
    st.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme="streamlit", animation_duration=animation_duration)
