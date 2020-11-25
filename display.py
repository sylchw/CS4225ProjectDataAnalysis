import streamlit as st
import pandas as pd
import numpy as np
import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import scipy.stats

import pyodbc 

server = 'cs4225database.database.windows.net' 
database = 'sentimentResult' 
username = 'cs4225test' 
password = 'Cs4225gogogo' 
tianHangConn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
#cursor = cnxn.cursor()

server = 'cs4225projectdata.database.windows.net' 
database = 'cs4225projectdata' 
username = 'cs4225project' 
password = '1y2u3i4o5P!' 
sylvesterConn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)

def convertToTimestamp(string):
    return pd.Timestamp(datetime.datetime.strptime(string, "%Y-%m-%d"))

def getSentimentValue(positive, negative):
    return (positive/ (positive + negative))

################QUERIES#######################
casesQuery = '''SELECT NC.NewCases, NC.NewCasesSmoothed, NC.Date, CTY.CountryName FROM NewCases NC
    INNER JOIN Country CTY
    ON NC.CountryID = CTY.ID
    '''

deathsQuery = '''SELECT ND.NewDeaths, ND.NewDeathsSmoothed, ND.Date, CTY.CountryName FROM NewDeaths ND
INNER JOIN Country CTY
ON ND.CountryID = CTY.ID'''

testsQuery = '''SELECT NT.NewTests, NT.NewTestsSmoothed, NT.Date, CTY.CountryName FROM NewTests NT
INNER JOIN Country CTY
ON NT.CountryID = CTY.ID'''

bedsQuery = '''SELECT BPT.BedsPerThousand, BPT.Date, CTY.CountryName FROM BedsPerThousand BPT
INNER JOIN Country CTY
ON BPT.CountryID = CTY.ID'''

handwashQuery = '''SELECT HF.HandwashingShare, HF.Date, CTY.CountryName FROM HandwashingFacilities HF
INNER JOIN Country CTY
ON HF.CountryID = CTY.ID'''

stringencyQuery = '''SELECT ST.Stringency, ST.Date, CTY.CountryName FROM Stringency ST
INNER JOIN Country CTY
ON ST.CountryID = CTY.ID
'''


#@st.cache
def loadSentimentData():
  tianHang = pd.read_sql_query(
  '''SELECT * FROM dbo.sentimentCount''', tianHangConn)
  tianHang['Date'] = tianHang['_c0'].apply(convertToTimestamp)
  tianHang['Sentiment'] = np.vectorize(getSentimentValue)(tianHang['positive_Count'], tianHang['negative_Count'])
  return tianHang

#@st.cache
def loadCountryAnalysisData(query):
  sylvester = pd.read_sql_query(
  query, sylvesterConn)
  return sylvester

st.title('CS4225 Group 4 Analysis')

st.title('COVID-19 Metrics vs Sentiments of country tweets')

sentimentData = loadSentimentData()

#st.write(sentimentData)
#st.write(loadCountryAnalysisData())

selectedCountry = st.selectbox(
    'Select Country',
     sentimentData['_c2'].unique())

types = ['New Daily Covid-19 Cases', 'New Daily Covid-19 Deaths', 'New Daily Covid-19 Tests', 'Daily Hospital Beds per Thousand', 'Daily Basic Handwashing Facilities', 'Daily Government Response Stringency Index']
queries = [casesQuery, deathsQuery, testsQuery, bedsQuery, handwashQuery, stringencyQuery]
selectedQuery = ['NewCases', 'NewDeaths', 'NewTests', 'BedsPerThousand', 'HandwashingShare', 'Stringency']

selectedType = st.selectbox(
    'Select Graph Type',
     types)


plotByTypes = ['Date', 'index']
selectedPlot = st.selectbox(
    'Plot by',
     plotByTypes)

data_load_state = st.text('Loading data...')
#data = load_data(10000)

#filter by country
Country = selectedCountry

#filtered_Sylvester = sylvester[sylvester['CountryName'] == Country].copy()

countryAnalysisData = loadCountryAnalysisData(queries[types.index(selectedType)])
filteredCountryAnalysisData = countryAnalysisData[countryAnalysisData['CountryName'] == Country].copy()
filteredSentimentData = sentimentData[sentimentData['_c2'] == Country].copy()

plot = pd.merge(left=filteredCountryAnalysisData, right=filteredSentimentData, left_on='Date', right_on='Date')
plot = plot[['Date',selectedQuery[types.index(selectedType)], 'Sentiment']]
plot = plot.reset_index()

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])
# Add traces
fig.add_trace(go.Scatter(x=plot[selectedPlot], y=plot[selectedQuery[types.index(selectedType)]],
                    mode='lines+markers',
                    name=selectedType))

fig.add_trace(go.Scatter(x=plot[selectedPlot], y=plot['Sentiment'],
                    mode='lines+markers',
                    name='Sentiment'),
                    secondary_y=True,)
# Add figure title
fig.update_layout(
    title_text= selectedType
)
# Set x-axis title
fig.update_xaxes(title_text=selectedPlot)
# Set y-axes titles
fig.update_yaxes(title_text=selectedType, secondary_y=False)
fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)

st.write(fig)

r = np.corrcoef(plot[selectedQuery[types.index(selectedType)]], plot['Sentiment'])
correlationCoef = r[0, 1]
st.subheader("Correlation Coefficient: " + str(correlationCoef))

#Pearson's r
pearsons = scipy.stats.pearsonr(plot[selectedQuery[types.index(selectedType)]], plot['Sentiment'])

# Spearman's rho
spearman = scipy.stats.spearmanr(plot[selectedQuery[types.index(selectedType)]], plot['Sentiment'])

# Kendall's tau
kendall = scipy.stats.kendalltau(plot[selectedQuery[types.index(selectedType)]], plot['Sentiment'])

st.subheader("Pearson's r: " + str(pearsons))
st.subheader("Spearman's rho: " + str(spearman))
st.subheader("Kendall's tau: " + str(kendall))

data_load_state.text("Loading Done!")



#DATE_COLUMN = 'date/time'
##DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
##            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

#@st.cache
#def load_data(nrows):
#    data = pd.read_csv(DATA_URL, nrows=nrows)
#    lowercase = lambda x: str(x).lower()
#    data.rename(lowercase, axis='columns', inplace=True)
#    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#    return data


#if st.checkbox('Show raw data'):
#    st.subheader('Raw data')
#    st.write(data)

#st.subheader('Number of pickups by hour')
#hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
#st.bar_chart(hist_values)

## Some number in the range 0-23
#hour_to_filter = st.slider('hour', 0, 23, 17)
#filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

#st.subheader('Map of all pickups at %s:00' % hour_to_filter)
#st.map(filtered_data)