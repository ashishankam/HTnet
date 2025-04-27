import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# st.write("MCC - Matthews Correlation Coefficient (MCC)")
st.markdown("# MCC - Matthews Correlation Coefficient (MCC)")
var = '''MCC considers all four elements of the confusion matrix: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
This comprehensive nature makes MCC a balanced measure even when the class distributions are skewed.
It provides a single summary statistic for multi-class classification.'''
st.markdown(var)


var =''' ### MCC for Htnet \n  
\t  Weighted Average MCC: 0.2679 \n
\t Macro-Averaged MCC: 0.2155 \n
\t Overall MCC: 0.6003'''

st.markdown(var)



