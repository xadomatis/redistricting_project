#packages
import streamlit as sl
import pandas as pd
import numpy as np
import altair as alt

# cd desktop/dspp/solo_projects/redistricting_project/app_building
# streamlit run app.py

sl.title("Xav's Redistricting Hub")

sl.write("""
## Viewing Partisan Gerrymandering Through a Non-Geographic Lens
For decades, anti-Gerrymandering advocates have condemned districts like "Maryland's Crab" or "Ohio's By-The_Lake Districts" as horribly gerrymandered distortions.. However, highlighting these districts de-emphasizes several key states that misrepresent their constituencies with their Congressional maps. With this app, I seek to show partisan distortion fully abstracted from Geography, based on history and probability
To the left, input an election year and a partisan wave, and the plots below will show the baseline distortion and a predicted election outcome.
""")

#set hexagon shape
hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"

#filepaths
converter = "/Users/xavier/Desktop/DSPP/solo_projects/redistricting_project/app_building/app_data/converter.csv"
state_data = "/Users/xavier/Desktop/DSPP/solo_projects/redistricting_project/app_building/app_data/state_data.csv"
state_coords = "/Users/xavier/Desktop/DSPP/solo_projects/redistricting_project/app_building/app_data/cartogram_coords.csv"
hourglass_coords = "/Users/xavier/Desktop/DSPP/solo_projects/redistricting_project/app_building/app_data/hourglass_coords.csv"
district_data = "/Users/xavier/Desktop/DSPP/solo_projects/redistricting_project/app_building/app_data/district_data.csv"

#read in csvs
cv = pd.read_csv(converter)
st = pd.read_csv(state_data)
sc = pd.read_csv(state_coords)
hc = pd.read_csv(hourglass_coords)
dd = pd.read_csv(district_data)

year_name = sl.sidebar.slider("Election Year", 2004, 2022, step=2)
wave_name = sl.sidebar.selectbox("Choose Wave for Predictive Plots", ("Neutral","Democratic","Republican"))


#merge state data with coordinates
st = st.merge(sc, how="left")

#build out the chart
df = st[st["year"] == year_name]

#create a function to change the probabilities based on a partisan wave

def call_wave(wave):
    '''Call a model by function name (to be used within a larger function)

    Args:
        A string that refers to the model name. Options are Linear SVC, SVC, and Logistic Regression

    Returns:
        A model as a function for later use
    '''
    #create an if/else to identify the possible models
    if wave == "Republican":
        col = cv[["logit_red"]]
    elif wave == "Democratic":
        col = cv[["logit_blue"]]
    #return a string if it fails- it will block the function from applying
    else:
        col = cv[["logit_all"]]
    index = cv[["metric"]]
    index = index.join(col)
    index.columns = ["metric","prob"]
    #return index
    return index

#################### state cartogram
#build the chart
st_cart = alt.Chart(df).mark_point().encode(
    alt.X('X',axis=None),
    alt.Y('Y',axis=None),
    alt.Text('ST'),
    alt.Color('porp_loss', scale=alt.Scale(scheme='redblue'), legend=alt.Legend(title="Porportion not Represented")),
    tooltip = [alt.Tooltip('porp_text',title='Porportion Lost'),
               alt.Tooltip('dist_text',title='District Deficit')
              ]
)
#inser labels
st_cart_labs = alt.Chart(df).mark_text(align='center').encode(
    alt.X('X'),
    alt.Y('Y'),
    alt.Text('ST'),
).properties(title=f"Porportion Misrepresented by State in {year_name}")
#display
cart = alt.layer(st_cart, st_cart_labs).configure_axis(grid=False).configure_view(strokeWidth=0).configure_point(size=250,shape=hexagon,filled=True)

#################### bubble chart

#create a bubble chart
selection = alt.selection_single();
bubble = alt.Chart(df).mark_point(filled=True).encode(
    alt.X('metric', scale=alt.Scale(zero=False), title='Porportion Republican'),
    alt.Y('porp_loss', scale=alt.Scale(zero=False), title='Porportion of Population not Represented'),
    alt.Size('dist_loss', legend=alt.Legend(title="Amount of Seats Effected")),
    alt.Order('dist_loss', sort='descending'),
    alt.Color('porp_loss', scale=alt.Scale(scheme='redblue'), legend=alt.Legend(title="Porportion not Represented")),
    tooltip = [alt.Tooltip('ST',title='State'),
               alt.Tooltip('porp_text',title='Porportion Lost'),
               alt.Tooltip('dist_text',title='District Deficit')
              ]
).add_selection(selection).properties(title='Severity of Partisan Misrepresentation')

#################### Data Prep for Predictive plots

conv = call_wave(wave_name)
dist = dd[dd["year"] == year_name]
dist = dist.merge(conv, how="left")
#create the amounts chart for the Majority Bar
amounts = pd.DataFrame(dist.prob.value_counts()).reset_index()
amounts.columns = ["prob_GOP","num_dist"]
amounts["quantity"] = amounts["prob_GOP"] * amounts["num_dist"]
expec_GOP = amounts["quantity"].sum()
expec_dem = 435 - expec_GOP
d = {'Expected Seats' : [expec_dem, expec_GOP], 'Party' : ["Dem","GOP"]}
expec = pd.DataFrame(data = d, index=None)
#output string
party_control = str(np.where(expec_GOP > expec_dem, "Republicans", "Democrats"))
num = np.where(expec_GOP > expec_dem, expec_GOP, expec_dem)
num_seats = str(num.round())
phrase = party_control + " are expected to control the house with a majority of " + num_seats.rstrip(".0")  + " seats"
#output majority plot
#output majority control chart
cd_scale = [0,50,100,150,175,200,218,235,260,285,335,385,435]
maj = alt.Chart(expec).mark_bar().encode(
    alt.X('sum(Expected Seats)',axis=alt.Axis(values=cd_scale),title="Expected Partisan Control"),
    alt.Color('Party', scale=alt.Scale(range=["blue","red"]))
).properties(title=phrase)
#format data for median seat chart
dist = dist.sort_values("metric",ascending=False).reset_index()
dist["X"] = hc["X"]
dist["Y"] = hc["Y"]
#output median Seat Chart
hg = alt.Chart(dist).mark_point().encode(
    alt.X('X',axis=None),
    alt.Y('Y',axis=None),
    #alt.Text('ST'),
    alt.Color('prob', scale=alt.Scale(range=["darkblue","blue","grey","red","darkred"]), legend=alt.Legend(title="Probability of GOP Representation")),
    tooltip = [alt.Tooltip('ST#',title='District'),
               alt.Tooltip('PVI',title='PVI'),
               alt.Tooltip('prob',title='Prob GOP'),
              ]
).configure_point(size=15,shape=hexagon,filled=True).configure_view(strokeWidth=0)

################### Display Code
# Display maps
sl.altair_chart(cart, use_container_width=True)
sl.write("""
*Note that this map is the distortion based on how the districts drawn, and is unaffected by the wave input
""")
sl.altair_chart(bubble, use_container_width=True)
sl.write("""
*Note that this graph is the distortion based on how the districts drawn, and is unaffected by the wave input
""")
sl.altair_chart(hg, use_container_width=True)
sl.altair_chart(maj, use_container_width=True)

sl.write("""
### Distortion Rating Methodology
###### Baseline distortion models are based on a state's pvi in comparison to the average expected value of its districts
###### Other methods, like 538's Median Seat and Efficiency Gap metrics don't account for probabilities
###### For instance, if Texas, which is R+5, drew 38 R+5 districts, it would preform as perfectly balances on almost every metrict
###### However, 38 R+5 districts are predicted to yield 30 Republican-held seats, misrepresenting the large Democratic population
###### This is automatically controled for partisan waves, as PVI assumes the election was 50-50 nationally
### Prediction Methodology
###### For predictive models, logistic regression was used to build out the probabilites for each wave
### 2022 Data
###### 46/50 states have been sourced from Dave's Redistricting, which draws in geographic data directly from state legislatures and applies APIs to report out Congressional districts
    - Two states did not have quality data in Dave's so KY and WV are drawn from 538 with a reverse engineered PVI metrics
    - MO and NH do not have complete maps, so the most likely outcome was selected
###### Data for NY and KS is reflected as the most recently passed maps, though they are likely to be redrawn by state courts
""")
