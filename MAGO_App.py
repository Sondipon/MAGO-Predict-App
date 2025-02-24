import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Determine the correct base path for accessing files
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS  # PyInstaller temp directory
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(base_path, "MAGO.csv")

# Check if CSV file exists
if not os.path.exists(file_path):
    st.error(f"CSV file not found at: {file_path}")
    st.stop()

# Load the dataset
data = pd.read_csv(file_path)

# Ensure required columns exist
required_columns = {'Structure', 'HW', 'TW', 'GO'}
if not required_columns.issubset(data.columns):
    st.error("CSV file must contain 'Structure', 'HW', 'TW', and 'GO' columns.")
    st.stop()

# Sidebar: Structure selection
st.sidebar.header("Select Structure")
structures = data['Structure'].unique()
selected_structure = st.sidebar.selectbox("Choose a structure:", structures)

# Filter data for selected structure
filtered_data = data[data['Structure'] == selected_structure]
X = filtered_data[['HW', 'TW']].values
y = filtered_data['GO'].values

# Sidebar: User input for HW & TW
st.sidebar.header("Enter HW & TW Conditions")
hw_input = st.sidebar.number_input("Headwater Level (HW)", min_value=float(filtered_data["HW"].min()), max_value=float(filtered_data["HW"].max()), step=0.1)
tw_input = st.sidebar.number_input("Tailwater Level (TW)", min_value=float(filtered_data["TW"].min()), max_value=float(filtered_data["TW"].max()), step=0.1)

# Prevent out-of-range inputs
if hw_input < filtered_data["HW"].min() or hw_input > filtered_data["HW"].max():
    st.error("HW input is out of range. Please enter a valid value.")
    st.stop()

if tw_input < filtered_data["TW"].min() or tw_input > filtered_data["TW"].max():
    st.error("TW input is out of range. Please enter a valid value.")
    st.stop()

# Predict GO using interpolation
try:
    predicted_go = griddata(X, y, [(hw_input, tw_input)], method='cubic')
    if np.isnan(predicted_go[0]):
        predicted_go = griddata(X, y, [(hw_input, tw_input)], method='linear')
except Exception:
    predicted_go = None

st.title("MAGO Curve Predictor")
st.subheader(f"Predicted Gate Opening for {selected_structure}")

if predicted_go is not None and not np.isnan(predicted_go[0]):
    st.write(f"**{predicted_go[0]:.2f} feet**")
else:
    st.write("Prediction unavailable for given inputs.")

# Batch Processing Section
st.sidebar.header("Batch Processing")
uploaded_file = st.sidebar.file_uploader("Upload CSV file with HW & TW", type=["csv"])

if uploaded_file is not None:
    try:
        input_data = pd.read_csv(uploaded_file)
        if {'HW', 'TW'}.issubset(input_data.columns):
            input_points = input_data[['HW', 'TW']].values
            input_data['GO'] = griddata(X, y, input_points, method='cubic')

            output_file = "output_HW_TW_GO.csv"
            st.download_button(
                label="Download Processed Data",
                data=input_data.to_csv(index=False),
                file_name=output_file,
                mime="text/csv"
            )
            st.success("Batch processing completed successfully!")
        else:
            st.error("Uploaded CSV must contain 'HW' and 'TW' columns.")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Plot interpolated surface
fig, ax = plt.subplots()
tw_values = np.linspace(filtered_data["TW"].min(), filtered_data["TW"].max(), 100)
hw_values = np.linspace(filtered_data["HW"].min(), filtered_data["HW"].max(), 100)
tw_grid, hw_grid = np.meshgrid(tw_values, hw_values)
go_grid = griddata(X, y, (hw_grid, tw_grid), method='cubic')

contour = ax.contourf(tw_grid, hw_grid, go_grid, cmap="viridis")
plt.colorbar(contour, label="Gate Opening (GO) feet")
ax.set_xlabel("Tailwater Level (TW) feet NAVD")
ax.set_ylabel("Headwater Level (HW) feet NAVD")
ax.set_title(f"Interpolated MAGO Curve for {selected_structure}")

# Mark user input
ax.scatter(tw_input, hw_input, color='red', label="User Input")
ax.legend()

st.pyplot(fig)
