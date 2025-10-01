"""
Quantum Data Plotter Application

A Tkinter-based GUI application for visualizing quantum computing data from CSV files.
The application allows users to load quantum data, select specific columns, and generate
time series plots, histograms, and iteration-based plots with statistical analysis.
The time-based plots now aggregate data to show the first value per day.

Classes:
    QuantumDataPlotterApp: Main application class handling GUI and data visualization.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
import numpy as np


class QuantumDataPlotterApp:
    """
    Main application class for quantum data visualization.
    
    This class creates a GUI interface that allows users to:
    - Load CSV files containing quantum data
    - Select from relevant quantum data columns (T1, T2, readout_error)
    - Plot time series data (first value per day)
    - Plot histograms with distribution fitting
    - Plot data vs iteration index (based on time series data - first per day)
    - Extract the data currently displayed on the plot
    - Export the current plot to PDF or PNG
    
    Attributes:
        root (tk.Tk): The main Tkinter root window
        dataframe (pd.DataFrame): Loaded CSV data as pandas DataFrame
        selected_column (tk.StringVar): Currently selected column for plotting
        fig (matplotlib.figure.Figure): Main figure for plots
        ax (matplotlib.axes.Axes): Axes for plotting
        canvas (FigureCanvasTkAgg): Tkinter canvas for matplotlib figure
        current_x_data (pd.Series or np.ndarray): Currently plotted X-axis data
        current_y_data (pd.Series or np.ndarray): Currently plotted Y-axis data
        current_plot_type (str): Type of plot currently displayed ("time_series", "histogram", or "iteration")
        
    Methods:
        load_file(): Load and process CSV file
        get_filtered_data(): Filter data to first value per date-day
        plot_vs_time(): Create time series plot (first per day)
        plot_histogram(): Create histogram with distribution fitting
        plot_vs_iteration(): Create iteration plot (first per day)
        on_column_change(): Handle column selection changes
        extract_current_data(): Get and save the currently plotted data
        export_plot(): Export the current plot to PDF or PNG
    """
    
    def __init__(self, root):
        """
        Initialize the Quantum Data Plotter application.
        
        Args:
            root (tk.Tk): The main Tkinter root window
            
        Initializes:
            - Main GUI layout with buttons, combobox, and plot canvas
            - Default column selection set to "T1_0"
            - Matplotlib figure and Tkinter canvas integration
            - Grid configuration for responsive layout
            - Attributes to store currently plotted data
        """
        self.root = root
        self.root.title("Quantum Data Plotter")

        self.dataframe = None
        self.selected_column = tk.StringVar()
        self.selected_column.set("T1_0")  # Default selection

        # Attributes to store currently plotted data
        self.current_x_data = None
        self.current_y_data = None
        self.current_plot_type = None # "time_series", "histogram", or "iteration"

        # Set larger font sizes for plots
        plt.rcParams.update({
            'font.size': 14,  # Default font size
            'axes.titlesize': 16,  # Title size
            'axes.labelsize': 14,  # X and Y label size
            'xtick.labelsize': 12,  # X tick labels size
            'ytick.labelsize': 12,  # Y tick labels size
            'legend.fontsize': 12,  # Legend font size
            'figure.titlesize': 18  # Figure title size
        })

        # --- GUI Layout ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # File loading
        load_button = ttk.Button(main_frame, text="Load CSV File", command=self.load_file)
        load_button.grid(row=0, column=0, padx=(0, 10), pady=(0, 10))

        # Column selection
        col_label = ttk.Label(main_frame, text="Select Column:")
        col_label.grid(row=0, column=1, sticky=tk.W, pady=(0, 10))
        
        self.column_combobox = ttk.Combobox(main_frame, textvariable=self.selected_column, state="disabled")
        self.column_combobox.grid(row=0, column=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.column_combobox.bind('<<ComboboxSelected>>', self.on_column_change)

        # Plot buttons
        plot_time_button = ttk.Button(main_frame, text="Plot vs Time", command=self.plot_vs_time)
        plot_time_button.grid(row=1, column=0, padx=(0, 5), pady=(0, 10))

        plot_hist_button = ttk.Button(main_frame, text="Plot Histogram", command=self.plot_histogram)
        plot_hist_button.grid(row=1, column=1, padx=(5, 0), pady=(0, 10))

        # New Plot vs Iteration button
        plot_iter_button = ttk.Button(main_frame, text="Plot vs Iteration", command=self.plot_vs_iteration)
        plot_iter_button.grid(row=1, column=2, padx=(5, 5), pady=(0, 10))

        # Extract data button
        extract_button = ttk.Button(main_frame, text="Extract Plotted Data", command=self.extract_current_data)
        extract_button.grid(row=1, column=3, padx=(5, 0), pady=(0, 10))

        # Export plot button
        export_button = ttk.Button(main_frame, text="Export Plot", command=self.export_plot)
        export_button.grid(row=1, column=4, padx=(5, 0), pady=(0, 10))

        # Matplotlib canvas for plotting
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=5, sticky=(tk.W, tk.E, tk.N, tk.S)) # Updated columnspan

        # Configure grid weights for resizing
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(4, weight=1) # Updated to match new button count
        main_frame.rowconfigure(2, weight=1)

    def load_file(self):
        """
        Load CSV file and populate column selection combobox.
        
        Opens a file dialog to select a CSV file, then:
        - Reads the CSV into a pandas DataFrame
        - Validates the presence of 'last_update_date' column
        - Converts dates to UTC timezone-aware datetime objects
        - Filters for quantum-related columns (T1_, T2_, readout_error_)
        - Enables the column selection combobox
        
        Raises:
            Exception: If file loading or processing fails, shows error message
        """
        filepath = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filepath:
            return  # User cancelled

        try:
            self.dataframe = pd.read_csv(filepath, index_col=False)
            print(f"Loaded data with shape: {self.dataframe.shape}")

            # Validate required column
            if 'last_update_date' not in self.dataframe.columns:
                messagebox.showerror("Error", "CSV must contain 'last_update_date' column.")
                self.dataframe = None
                return

            # Convert to datetime with UTC timezone
            self.dataframe['last_update_date'] = pd.to_datetime(self.dataframe['last_update_date'], utc=True)

            # Filter for quantum data columns
            relevant_cols = [col for col in self.dataframe.columns if 
                             col.startswith('T1_') or 
                             col.startswith('T2_') or 
                             col.startswith('readout_error_')]
            
            if not relevant_cols:
                 messagebox.showwarning("Warning", "No columns found starting with 'T1_', 'T2_', or 'readout_error_'.")
                 relevant_cols = list(self.dataframe.columns)  # Fallback to all columns

            self.column_combobox['values'] = relevant_cols
            self.column_combobox.config(state="readonly")
            
            messagebox.showinfo("Success", f"File loaded successfully!\nFound {len(relevant_cols)} relevant columns.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            self.dataframe = None
            self.column_combobox.config(state="disabled")

    def get_filtered_data(self):
        """
        Filter dataframe to get first value per date-day for selected column.
        
        Returns:
            tuple: (x_data, y_data) where:
                - x_ datetime values for the x-axis (representing the day)
                - y_ column values for the y-axis
            Returns (None, None) if data is unavailable or invalid
            
        Processing:
            - Creates date-day groups using UTC time
            - Takes first occurrence per day group
            - Removes NaN values from selected column
        """
        if self.dataframe is None or self.selected_column.get() not in self.dataframe.columns:
            return None, None

        col_name = self.selected_column.get()
        df = self.dataframe.copy()

        # Validate datetime type
        if not pd.api.types.is_datetime64_any_dtype(df['last_update_date']):
            print("Error: 'last_update_date' is not a datetime type after loading.")
            return None, None

        # Create daily groups and take first value per day
        df['date_day'] = df['last_update_date'].dt.floor('D') # Changed from 'H' to 'D'
        filtered_df = df.sort_values('last_update_date').groupby('date_day').first().reset_index()

        # Remove NaN values
        filtered_df = filtered_df.dropna(subset=[col_name])

        return filtered_df['last_update_date'], filtered_df[col_name] # Return the original datetime of the first entry per day

    def plot_vs_time(self):
        """
        Plot selected column values vs time (first value per date-day).
        
        Creates a line plot with markers showing the temporal evolution
        of the selected quantum parameter. Data is aggregated to show
        only the first value per day to reduce noise.
        
        Shows warning/error messages for:
            - No loaded data
            - Data filtering failures
            - Invalid column selections
        """
        if self.dataframe is None:
            messagebox.showwarning("Warning", "Please load a file first.")
            return

        x_data, y_data = self.get_filtered_data()
        if x_data is None or y_data is None:
            messagebox.showerror("Error", f"Could not filter data for column '{self.selected_column.get()}'. Check data types.")
            return

        if y_data.empty:
            messagebox.showwarning("Warning", f"No valid filtered data in column '{self.selected_column.get()}' to plot.")
            return

        self.ax.clear()
        self.ax.plot(x_data, y_data, marker='o', linestyle='-', markersize=4)
        
        # Calculate and plot mean line
        mean_val = y_data.mean()
        self.ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.4f}')
        
        self.ax.set_title(f'{self.selected_column.get()} vs Time (First per Day)')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel(self.selected_column.get())
        self.ax.tick_params(axis='x', rotation=45)
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

        # Store the plotted data
        self.current_x_data = x_data
        self.current_y_data = y_data
        self.current_plot_type = "time_series"
        print(f"Stored time series data for extraction. Shape: {len(x_data)}, {len(y_data)}")

    def plot_vs_iteration(self):
        """
        Plot selected column values vs iteration index (based on time series data - first per day).
        
        Creates a line plot with markers showing the evolution of the selected
        quantum parameter against an index representing each unique date-day
        occurrence (as determined by the get_filtered_data method).
        The X-axis is a simple integer index (0, 1, 2, ...).
        
        Shows warning/error messages for:
            - No loaded data
            - Data filtering failures
            - Invalid column selections
        """
        if self.dataframe is None:
            messagebox.showwarning("Warning", "Please load a file first.")
            return

        time_x_data, y_data = self.get_filtered_data()
        if time_x_data is None or y_data is None:
            messagebox.showerror("Error", f"Could not filter data for column '{self.selected_column.get()}'. Check data types.")
            return

        if y_data.empty:
            messagebox.showwarning("Warning", f"No valid filtered data in column '{self.selected_column.get()}' to plot.")
            return

        # Create the iteration index for the X-axis (starting from 1 instead of 0)
        x_iter = range(1, len(y_data) + 1) # Creates [1, 2, 3, ..., len(y_data)]

        self.ax.clear()
        self.ax.plot(x_iter, y_data, marker='o', linestyle='-', markersize=4)
        
        # Calculate statistics
        mean_val = y_data.mean()
        max_val = y_data.max()
        min_val = y_data.min()
        max_idx = y_data.idxmax() if hasattr(y_data, 'idxmax') else y_data.tolist().index(max_val)
        min_idx = y_data.idxmin() if hasattr(y_data, 'idxmin') else y_data.tolist().index(min_val)
        
        # Calculate percentage differences
        mean_diff_max = ((max_val - mean_val) / mean_val) * 100 if mean_val != 0 else float('inf')
        mean_diff_min = ((min_val - mean_val) / mean_val) * 100 if mean_val != 0 else float('inf')
        
        # Plot mean line
        self.ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.4f}')
        
        # Plot max and min lines
        self.ax.axhline(y=max_val, color='green', linestyle=':', linewidth=1.5, label=f'Max: {max_val:.4f} (+{mean_diff_max:.2f}%)')
        self.ax.axhline(y=min_val, color='orange', linestyle=':', linewidth=1.5, label=f'Min: {min_val:.4f} ({mean_diff_min:.2f}%)')
        
        self.ax.set_title('Relaxation time (' + self.selected_column.get() + ') vs Iteration')
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel(self.selected_column.get() + ' (s)')
        self.ax.legend()
        # Optionally, you could set x-ticks to show fewer indices if the plot gets too crowded
        # self.ax.set_xticks(x_iter[::max(1, len(x_iter)//10)]) # Show roughly 10 ticks
        self.fig.tight_layout()
        self.canvas.draw()

        # Store the plotted data (using the iteration index as x-axis)
        self.current_x_data = x_iter
        self.current_y_data = y_data
        self.current_plot_type = "iteration"
        print(f"Stored iteration plot data for extraction. Shape: {len(x_iter)}, {len(y_data)}")


    def plot_histogram(self):
        """
        Plot histogram of selected column with distribution fitting.
        
        Creates a density histogram of filtered data (first value per day)
        and attempts to fit a normal distribution to the data. Shows both
        the histogram and fitted probability density function. Displays
        the fitted mu (mean) and sigma (std) parameters in the title.
        
        Processing:
            - Uses filtered data (first value per day) for the selected column
            - Removes NaN values before plotting
            - Calculates number of bins using Rice formula: n_bins = 2 * n^(1/3)
            - Fits normal distribution using scipy.stats
            - Plots histogram and fitted PDF on same axes
            - Calculates and displays mu and sigma in the title
        
        Shows warnings for:
            - No loaded data
            - Empty or invalid data columns
            - Distribution fitting failures (handled gracefully)
        """
        if self.dataframe is None:
            messagebox.showwarning("Warning", "Please load a file first.")
            return

        # Use the filtered data (first value per day) instead of all data
        x_data, y_data = self.get_filtered_data()
        if x_data is None or y_data is None:
            messagebox.showerror("Error", f"Could not filter data for column '{self.selected_column.get()}'. Check data types.")
            return

        if y_data.empty:
             messagebox.showwarning("Warning", f"No valid filtered data in column '{self.selected_column.get()}' to plot.")
             return

        # Calculate number of bins using Rice formula: n_bins = 2 * n^(1/3)
        n = len(y_data)
        n_bins = int(2 * (n ** (1/3)))
        # Ensure at least 1 bin and reasonable maximum
        n_bins = max(1, min(n_bins, 100))

        self.ax.clear()

        # Plot histogram using the filtered data with  Rice formula bins currently not used
        counts, bins, patches = self.ax.hist(y_data, bins=16, density=True, alpha=0.6, 
                                            color='skyblue', edgecolor='black', 
                                            label='Filtered Data Histogram')

        # Fit normal distribution to the filtered data
        mu = None
        sigma = None
        try:
            dist_name = "norm"
            dist = getattr(stats, dist_name)
            params = dist.fit(y_data) # params = (mu, sigma)
            mu, sigma = params
            xmin, xmax = self.ax.get_xlim()
            x = np.linspace(xmin, xmax, 1000)
            fitted_pdf = dist.pdf(x, *params)
            self.ax.plot(x, fitted_pdf, label=f'Fitted {dist_name.title()} PDF', 
                        linewidth=2, color='red')
            self.ax.legend()
        except Exception as e:
            print(f"Warning: Could not fit distribution: {e}")
            # Optionally, clear mu/sigma if fitting fails
            # mu = None
            # sigma = None

        # Construct the title with fitted parameters if available
        if mu is not None and sigma is not None:
            title = f'Histogram of {self.selected_column.get()} \nFitted Normal Dist: μ={mu:.4f}, σ={sigma:.4f} '
        else:
            title = f'Histogram of {self.selected_column.get()}  (Fit Failed) '

        self.ax.set_title(title)
        self.ax.set_xlabel(self.selected_column.get())
        self.ax.set_ylabel('Density')
        self.fig.tight_layout()
        self.canvas.draw()

        # Store the plotted data for histogram
        # For histogram, x-axis is the bin centers, y-axis is the density values
        bin_centers = (bins[:-1] + bins[1:]) / 2
        self.current_x_data = bin_centers
        self.current_y_data = counts # Note: counts are the density values from the histogram
        self.current_plot_type = "histogram"
        print(f"Stored histogram data for extraction. Shape: {len(bin_centers)}, {len(counts)}")


    def on_column_change(self, event):
        """
        Handle column selection changes.
        
        Args:
            event: Tkinter event object (unused)
            
        Note:
            Currently this method is a placeholder and doesn't perform
            any automatic actions when column changes
        """
        # Optional: Could be used to update plots automatically when column changes
        pass

    def extract_current_data(self):
        """
        Extract and save the currently plotted X and Y data to a CSV file.
        
        This method retrieves the data stored by the last executed plot command
        (either plot_vs_time, plot_vs_iteration, or plot_histogram) and allows 
        the user to save it to a new CSV file.
        
        Behavior:
            - Checks if any plot has been generated yet
            - Opens a file dialog to choose the save location
            - Creates a DataFrame with the stored X and Y data
            - Adds a column indicating the plot type
            - Saves the DataFrame to the chosen CSV file
        
        Shows warnings for:
            - No data available (no plot generated yet)
            - User cancelling the save dialog
            - File saving errors
        """
        if self.current_x_data is None or self.current_y_data is None:
            messagebox.showwarning("Warning", "No data currently plotted. Generate a plot first before extracting data.")
            return

        # Ask user for file path to save the data
        filepath = filedialog.asksaveasfilename(
            title="Save Plotted Data As",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filepath:
            return # User cancelled

        try:
            # Create a DataFrame with the current plot data
            # Convert x_data to list if it's a range object (for iteration plot)
            x_data_to_save = list(self.current_x_data) if isinstance(self.current_x_data, range) else self.current_x_data
            export_df = pd.DataFrame({
                'plot_x_data': x_data_to_save,
                'plot_y_data': self.current_y_data
            })
            export_df['plot_type'] = self.current_plot_type # Add a column to indicate the plot type

            # Save the DataFrame to CSV
            export_df.to_csv(filepath, index=False)
            messagebox.showinfo("Success", f"Current plot data saved successfully to:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save \n{e}")
    
    def export_plot(self):
        """
        Export the current plot to PDF or PNG format.
        
        This method allows the user to save the currently displayed plot
        to a PDF or PNG file. The user selects the file format and location.
        
        Behavior:
            - Checks if any plot has been generated yet
            - Opens a file dialog to choose the save location and format
            - Saves the current figure to the chosen file
            - Supports both PDF and PNG formats
        
        Shows warnings for:
            - No plot available to export
            - User cancelling the save dialog
            - File saving errors
        """
        if self.current_x_data is None or self.current_y_data is None:
            messagebox.showwarning("Warning", "No plot currently displayed. Generate a plot first before exporting.")
            return

        # Ask user for file path to save the plot
        filepath = filedialog.asksaveasfilename(
            title="Export Plot As",
            defaultextension=".pdf",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        if not filepath:
            return # User cancelled

        try:
            # Save the current figure to the selected file
            self.fig.savefig(filepath, bbox_inches='tight', dpi=300)
            messagebox.showinfo("Success", f"Plot exported successfully to:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export plot:\n{e}")


if __name__ == "__main__":
    """
    Application entry point.
    
    Creates the Tkinter root window and instantiates the 
    QuantumDataPlotterApp class, then starts the main event loop.
    """
    root = tk.Tk()
    app = QuantumDataPlotterApp(root)
    root.mainloop()