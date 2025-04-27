# from test import test_unseen_image 
import streamlit as st
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from perform_gui import performance_graph

from testing_unseen import test_unseen_image



import streamlit as st

# Define pages
def predict():
    st.markdown(
    """
    <style>
    body {
        background-color: #ADD8E6; /* Light blue color */
    }
    </style>
    """,
    unsafe_allow_html=True
)
    st.title("Micro-Expression Recognition using Hierarchical Transformer Network")
    st.title("Prediction ❓")
    # st.set_page_config(page_title="Predict", page_icon="❓")
    onset_frame = st.file_uploader("Choose an onset frame")
    apex_frame = st.file_uploader("Choose an apex frame")
    a = st.button("Predict")

    if onset_frame and apex_frame:
        save_path_onset = os.path.join("E:\HTnet", onset_frame.name)  # Define save location
        save_path_apex = os.path.join("E:\HTnet", apex_frame.name)  # Define save location

    if a:
    # Declare prec as global before assigning any value to it
        # Assuming test_unseen_image is a function that processes the frames and returns the result
        model_config = {
        "image_size": 28,
        "patch_size": 7,
        "dim": 256,
        "heads": 3,
        "num_hierarchies": 3,
        "block_repeats": (2, 2, 10),
        "num_classes": 3  # Adjust according to the training setup
    }
        prec=test_unseen_image(save_path_onset,save_path_apex, weights_folder ="E:\HTnet\HTNet-master\ourmodel_threedatasets_weights", model_config= model_config)
                # b = st.button("predict")
        if prec == 0:
            pred = "Negative " 
        elif prec == 1:
            pred = "Positive (happiness)"
        else:
            pred = "Suprise (suprise)"
        var = f"Prediction : {pred}"

        st.markdown(f"""
<div style="display: flex; justify-content: center; align-items: center;">                    
<div style="border: 1px solid black; padding: 10px;display: inline-block; border-radius: 5px; background-color: #142b39; ">
    <p style = "color: white; font-size: 24px; font-weight: bold">{var}</p>
</div>
</div>
""", unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        # st.markdown( var )



        col1, col2 = st.columns(2, gap="small") 
        col1.image(onset_frame, caption="Onset frame image", width=400)
        col2.image(apex_frame, caption="Apex frame image", width=400)

        image1=r'E:\HTnet\HTNet-master\Optical_Flow_Image.png'
        st.image(image1, caption="optical flow image", width=400)
  
       



def performance():

    performance_graph()


def mcc():
    st.markdown("# MCC - Matthews Correlation Coefficient (MCC)")
    var = '''MCC considers all four elements of the confusion matrix: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
    This comprehensive nature makes MCC a balanced measure even when the class distributions are skewed.
    It provides a single summary statistic for multi-class classification.'''
    st.markdown(var)


    var =''' ## MCC for Htnet \n  
    \t Weighted Average MCC: 0.2679 \n
    \t Macro-Averaged MCC: 0.2155 \n
    \t Overall MCC: 0.6003'''

    st.markdown(var)



def ece():
    '''
Expected Calibration Error (ECE) measures the calibration of a model's predicted probabilities. In simple terms, it evaluates how well a model's predicted confidence levels align with the actual correctness of predictions. Unlike metrics like MCC that focus on classification performance, ECE focuses on the probabilistic output of the model.



ECE does not measure accuracy directly but evaluates how trustworthy the model's predicted probabilities are.'''

    st.markdown("# ECE - Expected Calibration Error ")
    st.markdown("Expected Calibration Error (ECE) measures the calibration of a model's predicted probabilities. In simple terms, it evaluates how well a model's predicted confidence levels align with the actual correctness of predictions. Unlike metrics like MCC that focus on classification performance, ECE focuses on the probabilistic output of the model.")
    
    var = '''ECE values range from 0 to 1.
0: Perfect calibration (confidence matches accuracy in all bins).
Higher values indicate poor calibration'''
    st.markdown(var)
   
    var =''' ## ECE for Htnet \n  
    \t ECE score for HTNet :  0.1942  \n'''

    st.markdown(var)

    
# Create sidebar for navigation


pages = {
    "Prediciton ❓": predict,
    "Performance Analysis 📈": performance,
    "MCC" : mcc,
    'ECE' : ece
    
}

# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Display the selected page
pages[selection]()