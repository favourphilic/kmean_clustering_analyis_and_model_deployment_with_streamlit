import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

 
st.header("Clustering Analysis and Model Building")

@st.cache_data
def getdata(path):
    df = pd.read_csv(path)
    df = df[df['HBUS']==1]
    df = df[df['INCOME'] < 500000]
    return df

df= getdata('SCFP2022.csv')
st.dataframe(df.head(2))
st.write(df.shape)
#df = df[df['HBUS']==1]
#df = df[df['INCOME'] < 500000]



st.sidebar.header("Options")
#Creating Side Bar Options
with st.sidebar:
     #activebusiness= st.multiselect('Active Business Group', df.HBUS.value_counts(normalize=True).index.to_list())
     add_variable = st.multiselect(
        "Variable with High Variance", options=df.var().sort_values().tail(10).index.to_list(),
          default =df.var().sort_values().tail(5).index.to_list()
          )
     #incomelevel = int(st.number_input('Enter and income Level limit'))
     
     
#mask2=(df['INCOME'] < incomelevel) #and  df['HBUS']== activebusiness) 
#mask = [mask1, mask2]
#df= df[mask2]
df= df[add_variable]

st.write("Creating Dataframe with high variance variable such as *ASSET*, *NETWORK*, *FIN*, *EQUITY*, *NFIM* ")
st.dataframe(df.head())
st.write(df.shape)

with st.sidebar:
     cluster= st.slider("Select the Number of Clusters to train your Model", min_value=2, max_value=8)

st.header("Model Building and Visualization")
finalModel =make_pipeline(StandardScaler(), KMeans(n_clusters=cluster, random_state=42))
finalModel.fit(df)
labels= finalModel.named_steps['kmeans'].labels_

xgb = df.groupby(labels).mean()

fig = px.bar(xgb, barmode='group')
st.plotly_chart(fig)

st.header("Using PCA to reduce our Variable to just 2 dimensions")

pca = PCA(n_components=2, random_state=42)
xpca = pca.fit_transform(df)
pcadf= pd.DataFrame(xpca, columns=["PC1","PC2"])

st.write("Create scatter plot of *PC2* vs *PC1* ")
fig = px.scatter(data_frame=pcadf, x='PC1', y='PC2', color=labels.astype(str))
st.plotly_chart(fig)

st.subheader("Finally, Lets Make Prediction")
st.markdown(" ### ")
asset = st.number_input('ASSET', key="a")
networth= st.number_input('NETWORTH', key="b")
fin = st.number_input('FIN', key="c")
equity = st.number_input('EQUITY', key="d")
nfin = st.number_input('NFIN', key="e")

test= np.array([asset, networth, fin, equity, nfin]).reshape(1, -1)
pred  = finalModel.predict(test.reshape(1, -1))[0]

st.button("Reset", type="primary")
if st.button('Get Prediction'):
    st.write(f"Belongs to the {pred} - cluster")


