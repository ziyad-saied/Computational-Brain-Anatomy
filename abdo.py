import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
from scipy import stats
import statistics
import math
import time
import hydralit_components as hc

st.set_page_config(page_title="Computational Brain Anatomy",page_icon=":tada:",layout="wide")

menu_data = [
        {'icon': "far fa-copy", 'label':"Left End"},
        {'id':'Copy','icon':"🐙",'label':"Copy"},
        {'icon': "far fa-chart-bar", 'label':"Chart"},#no tooltip message
        {'icon': "far fa-address-book", 'label':"Book"},
        {'id':' Crazy return value 💀','icon': "💀", 'label':"Calendar"},
        {'icon': "far fa-clone", 'label':"Component"},
        {'icon': "fas fa-tachometer-alt", 'label':"Dashboard",'ttip':"I'm the Dashboard tooltip!"}, #can add a tooltip message
        {'icon': "far fa-copy", 'label':"Right End"},
]
# we can override any part of the primary colors of the menu
#over_theme = {'txc_inactive': '#FFFFFF','menu_background':'red','txc_active':'yellow','option_active':'blue'}
over_theme = {'txc_inactive': '#FFFFFF'}
menu_id = hc.nav_bar(menu_definition=menu_data,home_name='Home',override_theme=over_theme)

    
# #get the id of the menu item clicked
# st.info(f"{menu_id=}")

st.title('Computational Brain Anatomy')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
rawF = pd.read_csv('E:/college/Final Year/graduation project/aa/RawF.txt')
DFF = pd.read_csv('E:/college/Final Year/graduation project/aa//DFF.txt')
@st.cache
def create_folder():
    import os
    dir = os.path.join("images")
    if not os.path.exists(dir):
        os.mkdir(dir)

def Data_preprocessing():
    create_folder()
    rawF = pd.read_csv('E:/college/Final Year/graduation project/aa/RawF.txt')
    DFF = pd.read_csv('E:/college/Final Year/graduation project/aa/DFF.txt')

    rawF_vector = rawF['Cell1']
    plt.plot(rawF_vector[40:80])
    plt.savefig('images/fig1.jpg', bbox_inches='tight')
    st.pyplot(plt)
    plt.close()

    m, n = rawF.shape
    rawF_matrix = rawF.iloc[:, 4:n]
    x = plt.imshow(rawF_matrix.iloc[0:480, :], aspect='auto', interpolation='none', origin='lower')
    plt.colorbar(x)
    plt.savefig('images/fig2.jpg', bbox_inches='tight')
    st.pyplot(plt)
    plt.close()

    rawF_rounded = round(rawF_vector / 10) * 10  # Round values to nearest multiple of 10
    baseline = statistics.mode(rawF_rounded)
    DFF_vector = (rawF_rounded - baseline) / baseline

    plt.plot(rawF_vector)
    plt.xlabel('Row')
    plt.ylabel('Raw F')
    plt.title('Raw fluorescence values of Cell 1')
    plt.savefig("images/fig3.jpg", bbox_inches='tight')
    st.pyplot(plt)
    plt.close()

    plt.plot(DFF_vector)
    plt.xlabel('Row')
    plt.ylabel('/DeltaF/F')
    plt.title('DF/F values of Cell 1')
    plt.savefig('images/fig4.jpg', bbox_inches='tight')
    st.pyplot(plt)
    plt.close()

    rawF_rounded = round(rawF_matrix / 10) * 10  # Round values to nearest multiple of 10
    baseline2 = rawF_rounded.mode(axis=0, numeric_only=True)
    baseline2 = baseline2.iloc[0]
    baseline2 = pd.DataFrame(baseline2)
    baseline2 = baseline2.T
    DFF_matrix = (rawF_rounded - baseline2.values) / baseline2.values
    DFF_matrix = pd.DataFrame(DFF_matrix)
    plt.imshow(rawF_matrix, extent=[0, 2500, 0, 2500])
    plt.xlabel("coulmn")
    plt.ylabel("Row")
    plt.colorbar(label='RawF Value', orientation="vertical")
    plt.savefig('images/fig5.jpg', bbox_inches='tight')
    st.pyplot(plt)
    plt.close()

    plt.imshow(DFF_matrix, extent=[0, 2500, 0, 2500])
    plt.xlabel("coulmn")
    plt.ylabel("Row")
    plt.colorbar(label='Column', orientation="vertical")
    plt.savefig('images/fig6.jpg', bbox_inches='tight')
    st.pyplot(plt)
    plt.close()

    cellOn = DFF.loc[(DFF['Cycle'] == 'ON') & (DFF['Orientation'] == 180)]
    cellOn = cellOn['Cell1']
    cellOff = DFF.loc[(DFF['Cycle'] == 'OFF') & (DFF['Orientation'] == 180)]
    cellOff = cellOff['Cell1']
    cellOn = pd.DataFrame(cellOn)
    cellOff = pd.DataFrame(cellOff)
    meanOn = np.mean(cellOn);
    meanOff = np.mean(cellOff);
    plt.plot(cellOn)
    plt.xlabel("Row")
    plt.ylabel("Raw F")
    plt.title('Raw fluorescence values of Cell 1a')
    plt.savefig('images/fig7.jpg', bbox_inches='tight')
    st.pyplot(plt)
    plt.close()

    plt.plot(cellOff)
    plt.xlabel("Row")
    plt.ylabel("Raw F")
    plt.title('Raw fluorescence values of Cell 1a')
    plt.savefig('images/fig8.jpg', bbox_inches='tight')
    st.pyplot(plt)
    plt.close()

    numTrials = 6
    results = DFF.groupby(["Orientation", "Cycle"]).mean()
    results = results.iloc[:, 2]
    results = pd.DataFrame(results)
    cellOn = DFF.loc[(DFF['Cycle'] == 'ON')]
    cellOff = DFF.loc[(DFF['Cycle'] == 'OFF')]
    resultsOn1 = cellOn.groupby("Orientation").std()
    resultsOn2 = cellOn.groupby("Orientation").mean()
    resultsOn1 = resultsOn1.iloc[:, 2]
    resultsOn2 = resultsOn2.iloc[:, 2]
    resultsOn1 = pd.DataFrame(resultsOn1)
    resultsOn2 = pd.DataFrame(resultsOn2)
    resultsOn = pd.concat([resultsOn2, resultsOn1], axis=1, ignore_index=True)
    resultsOn = resultsOn.rename(columns={0: 'mean', 1: 'std'}, inplace=False)
    resultsOff1 = cellOff.groupby("Orientation").std()
    resultsOff2 = cellOff.groupby("Orientation").mean()
    resultsOff1 = resultsOff1.iloc[:, 2]
    resultsOff2 = resultsOff2.iloc[:, 2]
    resultsOff1 = pd.DataFrame(resultsOff1)
    resultsOff2 = pd.DataFrame(resultsOff2)
    resultsOff = pd.concat([resultsOff2, resultsOff1], axis=1, ignore_index=True)
    resultsOff = resultsOff.rename(columns={0: 'mean', 1: 'std'}, inplace=False)
    meanCellOn = resultsOn.iloc[:, 0]
    meanCellOn = pd.DataFrame(meanCellOn)
    stdCellOn = resultsOn.iloc[:, 1]
    stdCellOn = pd.DataFrame(stdCellOn)
    stdErrorCellOn = stdCellOn / math.sqrt(numTrials)
    meanCellOff = resultsOff.iloc[:, 0]
    meanCellOff = pd.DataFrame(meanCellOff)
    stdCellOff = resultsOff.iloc[:, 1]
    stdCellOff = pd.DataFrame(stdCellOff)
    stdErrorCellOff = stdCellOff / math.sqrt(numTrials)
    orientation = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    orientations = pd.DataFrame(orientation)
    meanCellOfff = meanCellOff.iloc[0:1, 0:1]
    meanCellOfff = meanCellOfff['mean'].values.tolist()
    stdErrorCellOnn = stdErrorCellOn['std'].values.tolist()
    meanCellOnn = meanCellOn['mean'].values.tolist()
    line0 = plt.errorbar(orientation, meanCellOnn, yerr=stdErrorCellOnn, label="ON")
    line1 = plt.axhline(y=meanCellOfff, color='r', label="OFF")
    plt.xlabel("Orientation (deg)")
    plt.ylabel("/DeltaF/F")
    # plt.yline(meanCellOff[1],"r")
    plt.legend(handles=[line0, line1])
    plt.xlim(-30, 360)
    plt.savefig('images/fig9.jpg', bbox_inches='tight')
    st.pyplot(plt)
    plt.close()

    cellOn = DFF.loc[(DFF['Cycle'] == 'ON')]
    cellOff = DFF.loc[(DFF['Cycle'] == 'OFF')]
    resultsOn = cellOn.groupby(cellOn['Orientation']).mean()
    resultsOff = cellOff.groupby(cellOff['Orientation']).mean()
    meanOn = resultsOn.iloc[:, 2:]
    meanOff = resultsOff.iloc[:, 2:].mean()
    meanOff = pd.DataFrame(meanOff).T
    tuningCurves = meanOn.values - meanOff.values
    tuningCurves = pd.DataFrame(tuningCurves)
    tuningCurves[tuningCurves < 0] = 0
    orientation = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    line0, = plt.plot(orientation, tuningCurves.loc[:, 0], label='cell1')
    line1, = plt.plot(orientation, tuningCurves.loc[:, 1], label='cell2')
    line2, = plt.plot(orientation, tuningCurves.loc[:, 2], label='cell3')
    line3, = plt.plot(orientation, tuningCurves.loc[:, 3], label='cell4')
    line4, = plt.plot(orientation, tuningCurves.loc[:, 4], label='cell5')
    plt.xlabel("Orientation (deg)")
    plt.ylabel("ON response -  OFF response")
    plt.legend(handles=[line0, line1, line2, line3, line4])
    plt.title("Tuning curves")
    plt.savefig('images/fig10.jpg', bbox_inches='tight')
    st.pyplot(plt)
    plt.close()


x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)
st.text_input("Your name", key="name")

# You can access the value at any point with:
#st.session_state.name


if st.checkbox('Show Original Dataframe'):
    chart_data = pd.DataFrame(rawF)
    chart_data

if st.checkbox('Show Modified dataframe2'):
    chart_data = pd.DataFrame(DFF)
    chart_data

if st.checkbox('Image processing'):
    chart_data = Data_preprocessing()
    chart_data

# add_selectbox = st.sidebar.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone')
# )

# # Add a slider to the sidebar:
# add_slider = st.sidebar.slider(
#     'Select a range of values',
#     0.0, 100.0, (25.0, 75.0)
# )
# left_column, right_column = st.columns(2)
# # You can use a column just like st.sidebar:
# left_column.button('Press me!')

# # Or even better, call Streamlit functions inside a "with" block:
# with right_column:
#     chosen = st.radio(
#         'Sorting hat',
#         ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
#     st.write(f"You are in {chosen} house!")
# latest_iteration = st.empty()
# bar = st.progress(0)
