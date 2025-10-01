# coherence_plotter
Author: Adam Kaczmarek, JDU in Czestochowa, Poland


Quantum Data Plotter
A Tkinter-based GUI application for visualizing quantum computing data from CSV files. The application allows users to load quantum data, select specific columns, and generate time series plots, histograms, and iteration-based plots with statistical analysis. The time-based plots aggregate data to show the first value per day.

Features
CSV Data Loading: Load and process quantum computing data from CSV files
Data Filtering: Automatically filters data to show the first value per day
Multiple Plot Types:
Time series plots (first value per day)
Iteration-based plots (with statistical lines)
Histograms with distribution fitting
Statistical Analysis:
Mean value calculation and visualization
Max/Min value marking with percentage differences
Normal distribution fitting
Data Export:
Export plotted data to CSV
Export plots to PDF or PNG formats
Enhanced Visualization: Larger, more readable fonts
Requirements
Python 3.6+
Required packages:
tkinter
pandas
matplotlib
scipy
numpy
Installation
Clone or download the repository
Install required packages:
bash


1
pip install pandas matplotlib scipy numpy
Run the application:
bash


1
python quantum_data_plotter.py
Usage
Getting Started
Launch the application
Click "Load CSV File" to load your quantum data
Select the column to analyze from the dropdown menu
Plot Types
Time Series Plot
Shows the evolution of selected parameter over time
Displays mean value as a red dashed line
Uses first value per day to reduce noise
Iteration Plot
Shows data evolution against iteration number (starting from 1)
Includes statistical lines:
Red dashed line: Mean value
Green dotted line: Maximum value (with percentage difference from mean)
Orange dotted line: Minimum value (with percentage difference from mean)
Automatically calculates percentage differences
Histogram
Creates density histogram of filtered data
Uses Rice formula for optimal bin calculation: n_bins = 2 * n^(1/3)
Fits normal distribution to the data
Displays fitted parameters (μ and σ)
Export Functions
Extract Plotted Data: Save current X and Y data to CSV
Export Plot: Save current plot as PDF or PNG (300 DPI)
Data Format
The application expects CSV files with:

A column named last_update_date with datetime values
Quantum data columns starting with:
T1_ (T1 relaxation times)
T2_ (T2 relaxation times)
readout_error_ (readout error rates)
Example structure:

csv


1
2
3
4
last_update_date,T1_0,T2_0,readout_error_0
2023-01-01 10:00:00,0.0001,0.00005,0.01
2023-01-01 11:00:00,0.00012,0.000045,0.009
2023-01-02 09:00:00,0.000095,0.000055,0.011

License
MIT License - see LICENSE file for details.

Contributing
Fork the repository
Create a feature branch
Make your changes
Add tests if applicable
Submit a pull request
Support
For issues and questions, please open an issue in the repository.
