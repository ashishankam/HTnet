import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
# Data for plotting (Option 1)
dimensions_1 = [112, 192, 256, 392]
uf1_full_1 = [0.7123, 0.7251, 0.7358, 0.68]
uf1_samm_1 = [0.7029, 0.7002, 0.8131, 0.72]
uf1_casme2_1 = [0.9023, 0.9205, 0.9532, 0.84]
data1 = {'number of dimensions': dimensions_1, 'full': uf1_full_1, "samm": uf1_samm_1, "casme2":uf1_casme2_1}
df1 = pd.DataFrame(data1)
# Function to plot the graph for Option 1
def plot_graph_option_1():
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions_1, uf1_full_1, marker='o', label='Full')
    plt.plot(dimensions_1, uf1_samm_1, marker='o', label='SAMM')
    plt.plot(dimensions_1, uf1_casme2_1, marker='o', label='CASME II')

    # Adding labels, title, and legend
    plt.xlabel('Dimensions', fontsize=12)
    plt.ylabel('UF1', fontsize=12)
    plt.title('UF1 vs Dimensions for Full, SAMM, and CASME II', fontsize=14)
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)  # Display the plot in Streamlit


# Data for plotting (Option 2)
dimensions_2 = [112, 192, 256, 392]
full_2 = [0.7673, 0.6928, 0.7111, 0.76]
samm_2 = [0.6607, 0.6825, 0.8124, 0.80]
casme_ii_2 = [0.88, 0.90, 0.9516, 0.92]
data2 = {'number of dimensions': dimensions_2, 'full': full_2, "samm": samm_2, "casme2":casme_ii_2}
df2 = pd.DataFrame(data2)
# Function to plot the graph for Option 2
def plot_graph_option_2():
    plt.figure(figsize=(10, 6))

    # Plot the three lines
    plt.plot(dimensions_2, full_2, label='Full', marker='o', linestyle='-', color='blue')
    plt.plot(dimensions_2, samm_2, label='SAMM', marker='o', linestyle='-', color='green')
    plt.plot(dimensions_2, casme_ii_2, label='CASME II', marker='o', linestyle='-', color='red')

    # Labeling the axes and the plot
    plt.xlabel('Dimensions', fontsize=12)
    plt.ylabel('UAR', fontsize=12)
    plt.title('UAR vs Dimensions for Full, SAMM, and CASME II', fontsize=14)

    # Add a legend to differentiate between the three lines
    plt.legend()

    # Show the plot
    plt.grid(True)
    st.pyplot(plt)  # Display the plot in Streamlit







# Data for plotting (Option 3)
heads_3 = [2, 3, 4, 5]  # Only heads 2-5
full_3 = [0.7056, 0.7682, 0.7168, 0.7012]  # Full values
samm_3 = [0.6612, 0.7385, 0.6743, 0.6580]  # SAMM values
casme_ii_3 = [0.8080, 0.8590, 0.8485, 0.8350]
data3 = {'number of heads': heads_3, 'full': full_3, "samm": samm_3, "casme2":casme_ii_3}

df3 = pd.DataFrame(data3)
# Function to plot the graph for Option 2
def plot_graph_option_3():
    plt.figure(figsize=(10, 6))

    # Plot the three lines
    plt.plot(heads_3, full_3, label='Full', marker='o', linestyle='-', color='blue')
    plt.plot(heads_3, samm_3, label='SAMM', marker='o', linestyle='-', color='green')
    plt.plot(heads_3, casme_ii_3, label='CASME II', marker='o', linestyle='-', color='red')

    # Labeling the axes and the plot
    plt.xlabel('heads', fontsize=12)
    plt.ylabel('UAR', fontsize=12)
    plt.title('UF1 vs head for Full, SAMM, and CASME II', fontsize=14)

    # Add a legend to differentiate between the three lines
    plt.legend()

    # Show the plot
    plt.grid(True)
    st.pyplot(plt)  # Display the plot in Streamlit






# Data for plotting (Option 4)
heads = [2, 3, 4, 5]  # Only heads 2-5
full = [0.6654, 0.8480, 0.6852, 0.5590]  # Full values
samm = [0.6349, 0.8150, 0.6477, 0.6280]  # SAMM values
casme_ii = [0.9580, 0.9480, 0.9420, 0.9280]
data4 = {'number of heads': heads, 'full': full, "samm": samm, "casme2":casme_ii}

df4 = pd.DataFrame(data4)
# Function to plot the graph for Option 2
def plot_graph_option_4():
    plt.figure(figsize=(10, 6))

    # Plot the three lines
    plt.plot(heads, full, label='Full', marker='o', linestyle='-', color='blue')
    plt.plot(heads, samm, label='SAMM', marker='o', linestyle='-', color='green')
    plt.plot(heads, casme_ii, label='CASME II', marker='o', linestyle='-', color='red')

    # Labeling the axes and the plot
    plt.xlabel('heads', fontsize=12)
    plt.ylabel('UAR', fontsize=12)
    plt.title('UAR vs Head for Full, SAMM, and CASME II', fontsize=14)

    # Add a legend to differentiate between the three lines
    plt.legend()

    # Show the plot
    plt.grid(True)
    st.pyplot(plt)  # Display the plot in Streamlit




def performance_graph():
    # Streamlit UI
    st.title('Performance Analysis 📈')

    # Radio buttons for selecting options
    option = st.radio('Choose an option', ['UF1 vs Dimensions for Full, SAMM, and CASME II', 
                                        'UAR vs Dimensions for Full, SAMM, and CASME II', 
                                        'UF1 vs Head for Full, SAMM, and CASME II', 
                                        'UAR vs Head for Full, SAMM, and CASME II'])

    # Display content based on selected option
    if option == 'UF1 vs Dimensions for Full, SAMM, and CASME II':
        st.write("Displaying graph")
        plot_graph_option_1() 
        st.write(df1)
        # Display graph for Option 1

    elif option == 'UAR vs Dimensions for Full, SAMM, and CASME II':
        st.write("Displaying graph")
        plot_graph_option_2() 
        st.write(df2)


    elif option == 'UF1 vs Head for Full, SAMM, and CASME II':
        st.write("Displaying graph")
        plot_graph_option_3() 
        st.write(df3)

    elif option == 'UAR vs Head for Full, SAMM, and CASME II':
        st.write("Displaying graph")
        plot_graph_option_4() 
        st.write(df4)