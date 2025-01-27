# your_package/gui.py
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import matplotlib.pyplot as plt
import numpy as np
from .data import OsirisGridFile
from .utils import integrate, transverse_average

class LAVA_tkinter_app:
    '''
    Main class for the LAVA application.
    
    Input:
        - root: the main window
            tkinter.Tk
    '''
    def __init__(self, root):
        
        # Main window
        self.root = root
        self.root.title("LAVA (LabAstro Visualization Assistant) - OSIRIS Data Grid Viewer")
        self.root.geometry("1000x600")
        
        # UI Elements
        self.frame_controls = tk.Frame(self.root)
        self.frame_controls.pack(padx=10, pady=10, fill=tk.X)
        
        # Add label controls
        self.frame_labels = tk.Frame(self.root)
        self.frame_labels.pack(padx=10, pady=5, fill=tk.X)
        
        # Title and labels entries
        self.title_var = tk.StringVar()
        self.xlabel_var = tk.StringVar()
        self.ylabel_var = tk.StringVar()
        
        # Title and labels entries (for the text boxes)
        tk.Label(self.frame_labels, text="Title:").pack(side=tk.LEFT)
        tk.Entry(self.frame_labels, textvariable=self.title_var, width=30).pack(side=tk.LEFT, padx=5)
        tk.Label(self.frame_labels, text="X Label:").pack(side=tk.LEFT)
        tk.Entry(self.frame_labels, textvariable=self.xlabel_var, width=20).pack(side=tk.LEFT, padx=5)
        tk.Label(self.frame_labels, text="Y Label:").pack(side=tk.LEFT)
        tk.Entry(self.frame_labels, textvariable=self.ylabel_var, width=20).pack(side=tk.LEFT, padx=5)
        
        # Set up trace for automatic updates
        self.title_var.trace_add("write", self.update_plot_labels)
        self.xlabel_var.trace_add("write", self.update_plot_labels)
        self.ylabel_var.trace_add("write", self.update_plot_labels)

        # Browse and save buttons
        self.browse_btn = tk.Button(self.frame_controls, text="Browse Files", command=self.load_file)
        self.browse_btn.pack(side=tk.LEFT)
        self.save_btn = tk.Button(self.frame_controls, text="Save Plot", command=self.save_plot)
        self.save_btn.pack(side=tk.LEFT, padx=10)
        
        # Plot type menu (will be updated based on dimensionality) - Choose the type of plot
        self.plot_type_var = tk.StringVar()
        self.plot_menu = tk.OptionMenu(self.frame_controls, self.plot_type_var, "Select Plot Type")
        self.plot_menu.pack(side=tk.LEFT, padx=10)
        self.plot_type_var.trace_add("write", lambda *_: self.plot_data())  # Auto-update
        
        # Plot area
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Data info
        self.data_info = None
        self.dims = 0
        self.current_ax = None

    def load_file(self):
        '''
        Load an OSIRIS output HDF5 file and plot the data.
        '''
        
        # Open file dialog
        filepath = filedialog.askopenfilename(filetypes=[("HDF5 Files", "*.h5")])
        if not filepath:
            return
        
        try:
            # Read the file
            gridfile = OsirisGridFile(filepath)
            # Dimensions based on the number of axis and not on the dimension of the simulation
            self.dims = len(gridfile.axis)
            # Set the type of the file - grid, particles, tracks, tracks-2
            self.type = gridfile.type
        
            if self.type == "grid":
                # Set default labels based on dimensionality
                if self.dims == 1:
                    self.xlabel_var.set(r"$%s$ [$%s$]" % (gridfile.axis[0]["long_name"], gridfile.axis[0]["units"]))                # Set default x-label
                    self.ylabel_var.set(r"$%s$ [$%s$]" % (gridfile.label, gridfile.units))                                          # Set default y-label
                    x = np.linspace(gridfile.grid[0], gridfile.grid[1], gridfile.nx)                                                # Create x-axis array for plotting               
                    data_arr = gridfile.data                                                                                        # Get data array   
                    self.data_info = (x, data_arr)                                                                                  # Store data for plotting
                elif self.dims == 2:
                    self.xlabel_var.set(r"$%s$ [$%s$]" % (gridfile.axis[0]["long_name"], gridfile.axis[0]["units"]))                # Set default x-label
                    self.ylabel_var.set(r"$%s$ [$%s$]" % (gridfile.axis[1]["long_name"], gridfile.axis[1]["units"]))                # Set default y-label
                    x = np.linspace(gridfile.grid[0][0], gridfile.grid[0][1], gridfile.nx[0])                                       # Create x-axis array for plotting
                    y = np.linspace(gridfile.grid[1][0], gridfile.grid[1][1], gridfile.nx[1])                                       # Create y-axis array for plotting
                    data_arr = gridfile.data                                                                                        # Get data array    
                    self.data_info = (x, y, data_arr)                                                                               # Store data for plotting
                elif self.dims == 3:
                    print("3D not supported yet")
                else:
                    raise ValueError("Unsupported dimensionality")
                
                self.title_var.set(r"$%s$ [$%s$]" %( gridfile.label, gridfile.units))                                               # Set default title  
                
                # Update plot menu based on dimensionality - for each dimensionality there are different options for plotting    
                self.update_plot_menu()                                                                     
                
                # Plot the data - this will call the plot_data method that calls the plot_1d or plot_2d method based on the dimensionality
                self.plot_data()
                
            elif self.type == "particles":
                print("Particles (RAW) not supported yet")
            elif self.type == "tracks":
                print("Tracks not supported yet")
            elif self.type == "tracks-2":
                print("Tracks-2 not supported yet")
                    
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_plot_labels(self, *args):
        """
        Update labels without replotting data
        This is useful when the user changes the labels manually.
        Works by updating the current axis labels and figure title and redrawing the canvas (tkinter).
        """
        if self.current_ax:
            self.current_ax.set_xlabel(self.xlabel_var.get())
            self.current_ax.set_ylabel(self.ylabel_var.get())
            self.figure.suptitle(self.title_var.get())
            self.canvas.draw()

    def plot_data(self):
        """
        Full plot recreation with current settings
        First, clear the figure and then plot the data based on the current settings.
        To finish, update the plot labels and draw the canvas.
        """
        self.figure.clear()
        if self.dims == 1:
            self.plot_1d()
        elif self.dims == 2:
            self.plot_2d()
        self.update_plot_labels()
        self.canvas.draw()

    def plot_1d(self):
        """
        Plots 1D data
        """
        x, data = self.data_info                                                    # Unpack data
        self.current_ax = self.figure.add_subplot(111)                              # Add subplot to the axis that will be plotted
        plot_type = self.plot_type_var.get()                                        # Get the plot type
        
        if "Line" in plot_type:                                                     # If the plot type is a line plot
            self.current_ax.clear()                                                 # Clear the current axis
            self.current_ax.plot(x, data)                                           # Plot the data
        elif "Scatter" in plot_type:                                                # If the plot type is a scatter plot
            self.current_ax.clear()                                                 # Clear the current axis
            self.current_ax.scatter(x, data)                                        # Scatter plot the data
        
        # Apply initial labels
        self.current_ax.set_xlabel(self.xlabel_var.get())                           # Set the x-label
        self.current_ax.set_ylabel(self.ylabel_var.get())                           # Set the y-label
        self.figure.suptitle(self.title_var.get())                                  # Set the title

    def plot_2d(self):
        """
        Plots 2D data
        """
        x, y, data = self.data_info                                                     # Unpack data
        self.current_ax = self.figure.add_subplot(111)                                  # Add subplot to the axis that will be plotted
        plot_type = self.plot_type_var.get()                                            # Get the plot type
        
        if "Quantity" in plot_type:                                                     # If the plot type is a quantity plot
            self.current_ax.clear()                                                     # Clear the current axis
            img = self.current_ax.imshow(data, extent=(x[0], x[-1], y[0], y[-1]),       # Plot the data
                                       origin='lower', aspect='auto')
            self.figure.colorbar(img)                                                   # Add colorbar
        elif "Integral" in plot_type:                                                   # If the plot type is an integral plot
            self.current_ax.clear()                                                     # Clear the current axis
            avg = integrate(transverse_average(data), x[-1]/len(x))                     # Get the integral of the transverse average
            self.current_ax.plot(x, avg)                                                # Plot the integral 
        elif "Transverse" in plot_type:                                                 # If the plot type is a transverse average plot
            self.current_ax.clear()                                                     # Clear the current axis 
            avg = transverse_average(data)                                              # Get the transverse average
            self.current_ax.plot(x, avg)                                                # Plot the transverse average
        
        # Apply initial labels
        self.current_ax.set_xlabel(self.xlabel_var.get())                               # Set the x-label
        self.current_ax.set_ylabel(self.ylabel_var.get())                               # Set the y-label
        self.figure.suptitle(self.title_var.get())                                      # Set the title
        
    def update_plot_menu(self):
        """
        This method updates the plot menu based on the dimensionality of the data.
        It is called when a new file is loaded.
        """
        menu = self.plot_menu["menu"]                                                                           # Get the menu
        menu.delete(0, "end")                                                                                   # Delete all the options
        
        if self.dims == 1:                                                  # If the dimensionality is 1
            options = ["Line Plot", "Scatter Plot"]                         # Set the options     
        elif self.dims == 2:                                                                    # If the dimensionality is 2
            options = ["Quantity Plot", "T. Average Integral", "Transverse Average"]            # Set the options
        
        for opt in options:                                                                         # For each option
            menu.add_command(label=opt, command=lambda v=opt: self.plot_type_var.set(v))            # Add the option to the menu
        self.plot_type_var.set(options[0])                                                          # Set the default option to the first one
        
    def save_plot(self):
        '''
        Save the current plot as a PNG or PDF file.
        '''
        filepath = filedialog.asksaveasfilename(filetypes=[("PNG Files", "*.png"), ("PDF Files", "*.pdf")])     
        if not filepath:
            return
        self.figure.savefig(filepath, dpi=800, bbox_inches="tight")
        
def LAVA_tkinter():
    root = tk.Tk()
    app = LAVA_tkinter_app(root)
    root.mainloop()

if __name__ == "__main__":
    LAVA_tkinter()