import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
import time
import hydralit_components as hc
#from multiapp import MultiApp

# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Computational Brain Anatomy", page_icon=":tada:", layout="wide")
# menu_data = [
#         {'id':'Copy','icon':"🐙",'label':"Image_processing"},
#         {'icon': "far fa-chart-bar", 'label':"Orignal_DATA"},#no tooltip message
#         {'icon': "far fa-address-book", 'label':"Book"},
#         {'id':' Crazy return value 💀','icon': "💀", 'label':"Calendar"},
#         {'icon': "far fa-clone", 'label':"Component"},
#         {'icon': "fas fa-tachometer-alt", 'label':"Dashboard",'ttip':"I'm the Dashboard tooltip!"}, #can add a tooltip message
# ]
# # we can override any part of the primary colors of the menu
# #over_theme = {'txc_inactive': '#FFFFFF','menu_background':'red','txc_active':'yellow','option_active':'blue'}
# over_theme = {'txc_inactive': '#FFFFFF'}
# menu_id = hc.nav_bar(menu_definition=menu_data,home_name='Home',override_theme=over_theme)


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_33asonmr.json")
img_contact_form = Image.open("images/yt_contact_form.png")
img_lottie_animation = Image.open("images/yt_lottie_animation.png")

# ---- HEADER SECTION ----
with st.container():
    st.subheader("Hiiiiiiiiiiii:wave:")
    st.title("Computational Brain Anatomy")
    st.write(
        "Analyze visual neuroscience data with ImageJ,MATLAB and Machine learning"
    )
    st.write("[Learn More >](https://www.hopkinsmedicine.org/health/conditions-and-diseases/anatomy-of-the-brain)")

# ---- WHAT I DO ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("What WE do")
        st.write("##")
        st.write(
            """
            On Our project We will :
            - Describe the features of images important for vision.
            - Define dendrites, soma, axon, synapses, and action potential. 
            - Identify the anatomy of the visual system. 
            - Describe the Nobel Prize winning experiments of Hubel and Wiesel.
            - Load and analyze visual neuroscience data using ImageJ.
            - Plot neuroscience data using MATLAB. 
            If this sounds interesting to you, so you don’t miss any content.
            """
        )
        st.write("[Contact Us >](https://gmail.com)")
    with right_column:
        st_lottie(lottie_coding, height=400, key="coding")


with st.container():
    st.write("---")
    st.header("Get In Touch With Ue!")
    st.write("##")

    contact_form = """
    <form action="https://formsubmit.co/abdohassan7001@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()
import collections
from numpy.core.defchararray import lower
import streamlit as st
import numpy as np
import pandas as pd
#from pages import utils


def app():
    st.markdown("## Data Upload")

    # Upload the dataset and save as csv
    st.markdown("### Upload a csv file for analysis.") 
    st.write("\n")

    # Code to read a single file 
    uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx','txt'])
    global data
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            data = pd.read_excel(uploaded_file)


    if st.button("Load Data"):
        
        # Raw data 
        st.dataframe(data)
        #data.to_csv('data/main_data.csv', index=False)

        # Collect the categorical and numerical columns 
        
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = list(set(list(data.columns)) - set(numeric_cols))
        
app()