import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Example dataframes for the Excel sheets (you can replace these with actual data)
data1 = {'Column1': [1, 2, 3], 'Column2': [4, 5, 6]}
data2 = {'Column1': [2, 3, 4], 'Column2': [5, 6, 7]}
data3 = {'Column1': [3, 4, 5], 'Column2': [6, 7, 8]}
data4 = {'Column1': [4, 5, 6], 'Column2': [7, 8, 9]}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)
df4 = pd.DataFrame(data4)

# Function to display graphs
def plot_graph(data):
    plt.figure(figsize=(6, 4))
    plt.plot(data['Column1'], data['Column2'], marker='o')
    plt.xlabel('Column1')
    plt.ylabel('Column2')
    plt.title('Graph')
    st.pyplot(plt)

# Streamlit UI
st.title('Radio Button to Display Graph and Excel')

# Radio buttons
option = st.radio('Choose an option', ['Option 1', 'Option 2', 'Option 3', 'Option 4'])

# Display content based on selected option
if option == 'Option 1':
    st.write("Displaying data for Option 1")
    st.write(df1)
    plot_graph(df1)

elif option == 'Option 2':
    st.write("Displaying data for Option 2")
    st.write(df2)
    plot_graph(df2)

elif option == 'Option 3':
    st.write("Displaying data for Option 3")
    st.write(df3)
    plot_graph(df3)

elif option == 'Option 4':
    st.write("Displaying data for Option 4")
    st.write(df4)
    plot_graph(df4)