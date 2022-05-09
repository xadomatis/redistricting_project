#packages
import streamlit as sl
import pandas as pd
import numpy as np
import altair as alt
import os

#path = os.getcwd()
#path

# Terminal commands for streamlit
# cd desktop/dspp/solo_projects/redistricting_project/app_building
# streamlit run app.py

# Create the title and pretext
sl.title("Xav's Redistricting Hub")

sl.write("""
## Viewing Partisan Gerrymandering Through a Non-Geographic Lens
For decades, anti-Gerrymandering advocates have condemned districts like "Maryland's Crab" or "Ohio's By-The-Lake" districts as horribly gerrymandered distortions. However, highlighting these districts de-emphasizes several key states that misrepresent their constituencies with their Congressional maps.
Partisan distortion comes in all shapes and sizes, and can be accomplished with great subtlety. With this app, I seek to show partisan distortion fully abstracted from geography, based on history and probability.
To the left, input an election year and a partisan wave, and the plots below will show the baseline distortion and a predicted election outcome.
""")


# Generate important figures and spectrums
hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"
bluetored= ["darkblue","blue","grey","red","darkred"]
redtoblue = ["darkred","red","grey","blue","darkblue"]

# Read in csvs
cv = pd.read_csv("app_data/converter.csv")
st = pd.read_csv("app_data/state_data.csv")
sc = pd.read_csv("app_data/cartogram_coords.csv")
hc = pd.read_csv("app_data/hourglass_coords.csv")
dd = pd.read_csv("app_data/district_data.csv")
#preserve a copy for Machine learning


# Set sliders and waves
year_name = sl.sidebar.slider("Election Year", 2004, 2022, step=2)
wave_name = sl.sidebar.selectbox("Choose Wave for Predictive Plots", ("Neutral","Democratic","Republican"))
model_name = sl.sidebar.selectbox("Choose Model for Prediction", ("Binned Averages","Logit","Support Vector Machine"))


# merge state data with coordinates
st = st.merge(sc, how="left")
st["PVI"] = st["PVI"].replace("R+0","EVEN")

# limit datasets by year
st = st[st["year"] == year_name]
dist = dd[dd["year"] == year_name]

#create a function to change the probabilities based on a partisan wave


if model_name == "Binned Averages":
    cv = cv[["metric","pvi_range","bin_prob","bin_prob_red","bin_prob_blue"]]
elif model_name == "Logit":
    cv = cv[["metric","pvi_range","logit_all","logit_red","logit_blue"]]
elif model_name == "Support Vector Machine":
    cv = cv[["metric","pvi_range","svm_all","svm_red","svm_blue"]]
cv.columns = ["metric","pvi_range","all","red","blue"]


#create a function to change the probabilities based on a partisan wave

def call_wave(wave):
    '''Select a column that relates to the wave probability

    Args:
        A string referencing the wave

    Returns:
        A  list of probabilities
    '''
    #create an if/else to identify the possible models
    if wave == "Republican":
        col = cv[["red"]]
    elif wave == "Democratic":
        col = cv[["blue"]]
    else:
        col = cv[["all"]]
    index = cv[["metric"]]
    index = index.join(col)
    index.columns = ["metric","prob"]
    #return index
    return index

# Construct state cartogram
# Build the chart base
st_cart = alt.Chart(st).mark_point().encode(
    alt.X('X',axis=None),
    alt.Y('Y',axis=None),
    alt.Text('ST'),
    alt.Color('porp_loss', scale=alt.Scale(range=redtoblue), legend=alt.Legend(title="Porportion not Represented")),
    tooltip = [alt.Tooltip('porp_text',title='Porportion Lost'),
               alt.Tooltip('dist_text',title='District Deficit')
              ]
)
# Overlay labels
cart_labs = alt.Chart(st).mark_text(align='center').encode(
    alt.X('X'),
    alt.Y('Y'),
    alt.Text('ST')
).properties(title=f"Porportion Misrepresented by State in {year_name}")
# Display
cart = alt.layer(st_cart, cart_labs).configure_axis(grid=False).configure_view(strokeWidth=0).configure_point(size=250,shape=hexagon,filled=True)

# Construct the Bubble chart

#Create Chart
selection = alt.selection_single();
bubble = alt.Chart(st).mark_point(filled=True).encode(
    alt.X('metric', scale=alt.Scale(zero=False), title='Porportion Republican'),
    alt.Y('porp_loss', scale=alt.Scale(zero=False), title='Porportion of Population not Represented'),
    alt.Size('dist_loss', legend=alt.Legend(title="Amount of Seats Affected")),
    alt.Order('dist_loss', sort='descending'),
    alt.Color('porp_loss', scale=alt.Scale(range=redtoblue), legend=alt.Legend(title="Porportion not Represented")),
    tooltip = [alt.Tooltip('ST',title='State'),
               alt.Tooltip('porp_text',title='Porportion Lost'),
               alt.Tooltip('dist_text',title='District Deficit')
              ]
).add_selection(selection).properties(title='Severity of Partisan Misrepresentation')


# Build the competition chart
dist["Competitiveness"] = np.where(dist["metric"] > .53, "Likely GOP", "Competitive")
dist["Competitiveness"] = np.where(dist["metric"] < .47, "Likely Dem", dist["Competitiveness"])
dist["Competitiveness"] = np.where(dist["metric"] > .6, "Solidly GOP", dist["Competitiveness"])
dist["Competitiveness"] = np.where(dist["metric"] < .4, "Solidly Dem", dist["Competitiveness"])
category_names = ["Solidly GOP", "Likely GOP", "Competitive", "Likely Dem", "Solidly Dem"]
comp = alt.Chart(dist).mark_bar().encode(
    alt.X("Competitiveness", sort=category_names),
    alt.Color('Competitiveness', sort=category_names, scale=alt.Scale(range=redtoblue), legend=None),
    alt.Y('count()', scale=alt.Scale(domain=(0, 150)),axis=alt.Axis(title='Number of Districts'))
).properties(title="Seat Polarity Distribution")

# Prep predictive plots
conv = call_wave(wave_name)
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
phrase = f'The {model_name} model predicts that ' + party_control + " are expected to control the House with a majority of " + num_seats.rstrip(".0")  + " seats"
#output majority control chart
cd_scale = [0,50,100,150,175,200,218,235,260,285,335,385,435]
maj = alt.Chart(expec).mark_bar().encode(
    alt.X('sum(Expected Seats)',axis=alt.Axis(values=cd_scale),title="Expected Seats"),
    alt.Color('Party', scale=alt.Scale(range=["blue","red"]))
).properties(title="Predicted Mean Partisan Control of Congress")
dd = dist
act_GOP = dd["is_GOP"].sum()
act_dem = 435 - act_GOP
d = {'Actual Seats' : [act_dem, act_GOP], 'Party' : ["Dem","GOP"]}
actual = pd.DataFrame(data = d, index=None)
act = alt.Chart(actual).mark_bar().encode(
    alt.X('sum(Actual Seats)',axis=alt.Axis(values=cd_scale),title="Actual Seats"),
    alt.Color('Party', scale=alt.Scale(range=["blue","red"]))
).properties(title="Actual Partisan Control of Congress")
#format data for median seat chart
dist = dist.sort_values("metric",ascending=False).reset_index()
dist["X"] = hc["X"]
dist["Y"] = hc["Y"]
#output median Seat Chart
hg = alt.Chart(dist).mark_point().encode(
    alt.X('X',axis=None),
    alt.Y('Y',axis=None),
    #alt.Text('ST'),
    alt.Color('prob', scale=alt.Scale(range=bluetored), legend=alt.Legend(title="Probability of GOP Representation")),
    tooltip = [alt.Tooltip('ST#',title='District'),
               alt.Tooltip('PVI',title='PVI'),
               alt.Tooltip('prob',title='Prob GOP'),
              ]
).properties(title=f'Median Seat Predicted with a {wave_name} wave in {year_name}').configure_point(size=15,shape=hexagon,filled=True).configure_view(strokeWidth=0)

#prep ranking diagrams
st["PVI"] = st["PVI"].replace("R+0","EVEN")
st["dl_abs"] = st["dist_loss"].abs()
dis_dist = st.sort_values("dl_abs",ascending=False)
dis_dist.columns = ["State","PVI","Distortion"]
worst_dist = dis_dist[["State","PVI","dist_text"]].reset_index(drop=True).head(5)
best_dist = dis_dist[["State","PVI","dist_text"]].reset_index(drop=True).tail(5)
st["pl_abs"] = st["porp_loss"].abs()
dis_porp = st.sort_values("pl_abs",ascending=False)
dis_porp.columns = ["State","PVI","Distortion"]
worst_porp = dis_porp[["State","PVI","porp_text"]].reset_index(drop=True).head(5)
best_porp = dis_porp[["State","PVI","porp_text"]].reset_index(drop=True).tail(5)



################### Display Code ###################
# Display maps
sl.write("""
The model predicts:
""")
sl.write(phrase)
sl.altair_chart(maj, use_container_width=True)
if year_name != 2022:
    sl.altair_chart(act, use_container_width=True)
sl.altair_chart(hg, use_container_width=True)
sl.altair_chart(cart, use_container_width=True)
sl.write("""
*This map is unaffected by wave output
""")
sl.altair_chart(bubble, use_container_width=True)
sl.write("""
*This graph is unaffected by wave output
""")
sl.altair_chart(comp, use_container_width=True)
sl.write("""
*This graph is unaffected by wave output
""")
sl.write("""
#### Least Distorted States by Porportion
""")
sl.write(best_porp)
sl.write("""
#### Least Distorted States by District Allocation
""")
sl.write(best_dist)
sl.write("""
#### Most Distorted States by Porportion
""")
sl.write(worst_porp)
sl.write("""
#### Most Distorted States by District Allocation
""")
sl.write(worst_dist)

sl.write("""
## Information
### Distortion Rating Methodology
Baseline distortion models are based on a state's pvi in comparison to the average expected value of its districts.
Other methods, like 538's Median Seat and Efficiency Gap metrics don't account for probabilities. For instance, if Texas, which is R+5, drew 38 R+5 districts, it would preform as perfectly balanced on almost every metric. However, 38 R+5 districts are predicted to yield 30 Republican-held seats, misrepresenting the large Democratic population. Because this method uses expected value, which shows substantial differences between R+1 and R+5, it would not assess the example map as fair.
### Prediction Methodology
For predictive models, logistic regression was used to build out the probabilites for each wave.
### 2022 Data
##### Sources
46/50 states have been sourced from Dave's Redistricting, which draws in geographic data directly from state legislatures and applies APIs to report out Congressional districts.
    - KY and WV are drawn from 538 with a reverse engineered PVI metric, due to data gaps in Dave's API.
    - MO and NH do not have complete maps, so the most likely outcome was selected.
##### Quirks
Data for NY and KS is reflected as the most recently passed maps, though they are likely to be redrawn by state courts.
Data pulled from 538 has a lower accuracy; their PVI ratings are not limited to Presidential elections and may be biased by ±2 PVI points.
""")
