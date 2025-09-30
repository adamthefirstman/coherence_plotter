import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
import numpy as np

class QuantumDataPlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Data Plotter")

        self.dataframe = None
        self.selected_column = tk.StringVar()
        self.selected_column.set("T1_0") # Default selection

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
        self.column_combobox.bind('<<ComboboxSelected>>', self.on_column_change) # Optional: Update plot on change

        # Plot buttons
        plot_time_button = ttk.Button(main_frame, text="Plot vs Time", command=self.plot_vs_time)
        plot_time_button.grid(row=1, column=0, padx=(0, 5), pady=(0, 10))

        plot_hist_button = ttk.Button(main_frame, text="Plot Histogram", command=self.plot_histogram)
        plot_hist_button.grid(row=1, column=1, padx=(5, 0), pady=(0, 10))

        # Matplotlib canvas for plotting
        self.fig, self.ax = plt.subplots(figsize=(10, 4)) # Smaller initial size
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights for resizing
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(2, weight=1) # Makes combobox expand
        main_frame.rowconfigure(2, weight=1)   # Makes canvas expand

    def load_file(self):
        """Loads the CSV file and populates the column selection combobox."""
        filepath = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filepath:
            return # User cancelled

        try:
            self.dataframe = pd.read_csv(filepath, index_col=False) # Ensure 'last_update_date' is a column
            print(f"Loaded data with shape: {self.dataframe.shape}")

            # Ensure 'last_update_date' exists and convert to datetime
            if 'last_update_date' not in self.dataframe.columns:
                messagebox.showerror("Error", "CSV must contain 'last_update_date' column.")
                self.dataframe = None
                return

            self.dataframe['last_update_date'] = pd.to_datetime(self.dataframe['last_update_date'])

            # Get columns that start with 'T1_', 'T2_', or 'readout_error_'
            relevant_cols = [col for col in self.dataframe.columns if 
                             col.startswith('T1_') or 
                             col.startswith('T2_') or 
                             col.startswith('readout_error_')]
            
            if not relevant_cols:
                 messagebox.showwarning("Warning", "No columns found starting with 'T1_', 'T2_', or 'readout_error_'.")
                 relevant_cols = list(self.dataframe.columns) # Fallback to all columns if none found

            self.column_combobox['values'] = relevant_cols
            self.column_combobox.config(state="readonly") # Enable and make readonly

            messagebox.showinfo("Success", f"File loaded successfully!\nFound {len(relevant_cols)} relevant columns.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            self.dataframe = None
            self.column_combobox.config(state="disabled")

    def get_filtered_data(self):
        """Filters the dataframe to get first value per date-hour for the selected column."""
        if self.dataframe is None or self.selected_column.get() not in self.dataframe.columns:
            return None, None

        col_name = self.selected_column.get()
        df = self.dataframe.copy()

        # Create a column for date-hour
        df['date_hour'] = df['last_update_date'].dt.floor('H') # Floor to hour

        # Group by date-hour and get the first row of each group
        filtered_df = df.groupby('date_hour').first().reset_index()

        # Remove rows where the selected column is NaN
        filtered_df = filtered_df.dropna(subset=[col_name])

        return filtered_df['last_update_date'], filtered_df[col_name]


    def plot_vs_time(self):
        """Plots the selected column value vs time (first per date-hour)."""
        if self.dataframe is None:
            messagebox.showwarning("Warning", "Please load a file first.")
            return

        x_data, y_data = self.get_filtered_data()
        if x_data is None or y_data is None:
            messagebox.showerror("Error", f"Column '{self.selected_column.get()}' not found or data is invalid.")
            return

        self.ax.clear()
        self.ax.plot(x_data, y_data, marker='o', linestyle='-', markersize=4)
        self.ax.set_title(f'{self.selected_column.get()} vs Time (First per Hour)')
        self.ax.set_xlabel('Date/Time')
        self.ax.set_ylabel(self.selected_column.get())
        self.ax.tick_params(axis='x', rotation=45)
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_histogram(self):
        """Plots a histogram of the selected column and fits a distribution."""
        if self.dataframe is None:
            messagebox.showwarning("Warning", "Please load a file first.")
            return

        # Use ALL data for the histogram, not just the first per hour
        col_name = self.selected_column.get()
        if col_name not in self.dataframe.columns:
            messagebox.showerror("Error", f"Column '{col_name}' not found.")
            return

        y_data = self.dataframe[col_name].dropna() # Remove NaNs for fitting

        if y_data.empty:
             messagebox.showwarning("Warning", f"No valid data in column '{col_name}' to plot.")
             return

        self.ax.clear()

        # Plot histogram
        counts, bins, patches = self.ax.hist(y_data, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Data Histogram')

        # Fit a distribution - example with Normal (Gaussian)
        # You can add logic to let the user choose the distribution
        try:
            dist_name = "norm" # Example: Normal distribution
            dist = getattr(stats, dist_name)
            params = dist.fit(y_data)
            # Generate x values for the fitted PDF
            xmin, xmax = self.ax.get_xlim()
            x = np.linspace(xmin, xmax, 1000)
            fitted_pdf = dist.pdf(x, *params)
            self.ax.plot(x, fitted_pdf, label=f'Fitted {dist_name.title()} PDF', linewidth=2, color='red')
            self.ax.legend()
        except Exception as e:
            print(f"Warning: Could not fit distribution: {e}")
            # Optionally, just show the histogram without the fit
            # self.ax.legend() # Only if you don't add the fit line

        self.ax.set_title(f'Histogram of {col_name}')
        self.ax.set_xlabel(col_name)
        self.ax.set_ylabel('Density')
        self.fig.tight_layout()
        self.canvas.draw()

    def on_column_change(self, event):
        """Optional: Could be used to update plots automatically when column changes."""
        # For now, we rely on the user clicking the plot buttons
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = QuantumDataPlotterApp(root)
    root.mainloop()