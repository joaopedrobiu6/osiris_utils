# your_package/gui.py
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from .data_readers import open1D, open2D, open3D, read_osiris_file  # Relative import
from .utils import integrate, transverse_average

class LAVA_app:
    def __init__(self, root):
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

        # Existing controls
        self.browse_btn = tk.Button(self.frame_controls, text="Browse Files", command=self.load_file)
        self.browse_btn.pack(side=tk.LEFT)
        
        self.pressure_var = tk.BooleanVar()
        self.pressure_check = tk.Checkbutton(self.frame_controls, text="Pressure", variable=self.pressure_var)
        self.pressure_check.pack(side=tk.LEFT, padx=10)
        
        self.save_btn = tk.Button(self.frame_controls, text="Save Plot", command=self.save_plot)
        self.save_btn.pack(side=tk.LEFT, padx=10)
        
        self.plot_type_var = tk.StringVar()
        self.plot_menu = tk.OptionMenu(self.frame_controls, self.plot_type_var, "Select Plot Type")
        self.plot_menu.pack(side=tk.LEFT, padx=10)
        self.plot_type_var.trace_add("write", lambda *_: self.plot_data())  # Auto-update
        
        # Plot area
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=20)
        self.slice_slider = None
        self.data_info = None
        self.dims = 0
        self.current_ax = None

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("HDF5 Files", "*.h5")])
        if not filepath:
            return
        
        try:
            pressure = self.pressure_var.get()
            _, _, data_ = read_osiris_file(filepath, pressure)
            self.dims = len(data_[:].shape)
            
            # Set default labels based on dimensionality
            if self.dims == 1:
                self.xlabel_var.set(r"x [c/$\omega_p$]")
                self.ylabel_var.set("Value")
                x, data_arr, _ = open1D(filepath, pressure)
                self.data_info = (x, data_arr)
            elif self.dims == 2:
                self.xlabel_var.set(r"x [c/$\omega_p$]")
                self.ylabel_var.set(r"y [c/$\omega_p$]")
                x, y, data_arr, _ = open2D(filepath, pressure)
                self.data_info = (x, y, data_arr)
            else:
                raise ValueError("Unsupported dimensionality")
            
            self.title_var.set("")  # Clear title on new file
            self.update_plot_menu()
            self.plot_data()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_plot_labels(self, *args):
        """Update labels without replotting data"""
        if self.current_ax:
            self.current_ax.set_xlabel(self.xlabel_var.get())
            self.current_ax.set_ylabel(self.ylabel_var.get())
            self.figure.suptitle(self.title_var.get())
            self.canvas.draw()

    def plot_data(self):
        """Full plot recreation with current settings"""
        self.figure.clear()
        if self.dims == 1:
            self.plot_1d()
        elif self.dims == 2:
            self.plot_2d()
        self.update_plot_labels()
        self.canvas.draw()

    def plot_1d(self):
        x, data = self.data_info
        self.current_ax = self.figure.add_subplot(111)
        plot_type = self.plot_type_var.get()
        
        if "Line" in plot_type:
            self.current_ax.clear()
            self.current_ax.plot(x, data)
        elif "Scatter" in plot_type:
            self.current_ax.clear()
            self.current_ax.scatter(x, data)
        
        # Apply initial labels
        self.current_ax.set_xlabel(self.xlabel_var.get())
        self.current_ax.set_ylabel(self.ylabel_var.get())
        self.figure.suptitle(self.title_var.get())

    def plot_2d(self):
        x, y, data = self.data_info
        self.current_ax = self.figure.add_subplot(111)
        plot_type = self.plot_type_var.get()
        
        if "Quantity" in plot_type:
            self.current_ax.clear()
            img = self.current_ax.imshow(data, extent=(x[0], x[-1], y[0], y[-1]), 
                                       origin='lower', aspect='auto')
            self.figure.colorbar(img)
        elif "Integral" in plot_type:
            self.current_ax.clear()
            avg = integrate(transverse_average(data), x[-1]/len(x))
            self.current_ax.plot(x, avg)
        elif "Transverse" in plot_type:
            self.current_ax.clear()
            avg = transverse_average(data)
            self.current_ax.plot(x, avg)
        
        # Apply initial labels
        self.current_ax.set_xlabel(self.xlabel_var.get())
        self.current_ax.set_ylabel(self.ylabel_var.get())
        self.figure.suptitle(self.title_var.get())
        
    def update_plot_menu(self):
        menu = self.plot_menu["menu"]
        menu.delete(0, "end")
        
        if self.dims == 1:
            options = ["Line Plot", "Scatter Plot"]
        elif self.dims == 2:
            options = ["Quantity Plot", "T. Average Integral", "Transverse Average"]
        
        for opt in options:
            menu.add_command(label=opt, command=lambda v=opt: self.plot_type_var.set(v))
        self.plot_type_var.set(options[0])
        
        
    def save_plot(self):
        filepath = filedialog.asksaveasfilename(filetypes=[("PNG Files", "*.png"), ("PDF Files", "*.pdf")])
        if not filepath:
            return
        self.figure.savefig(filepath, dpi=800, bbox_inches="tight")
        
    def add_title(self, title):
        self.figure.suptitle(title)
        self.canvas.draw()

def LAVA():
    root = tk.Tk()
    app = LAVA_app(root)
    root.mainloop()

if __name__ == "__main__":
    LAVA()