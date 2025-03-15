import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                              QFileDialog, QMessageBox, QComboBox, QHBoxLayout,
                              QVBoxLayout, QLabel, QLineEdit, QFrame, QDoubleSpinBox)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from .data import OsirisGridFile  # Update import as needed
from .utils import integrate, transverse_average  # Update import as needed

class LAVA_Qt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LAVA (LabAstro Visualization Assistant) - OSIRIS Data Grid Viewer')
        self.setGeometry(100, 100, 1000, 600)
        
        # Initialize data
        self.data_info = None
        self.dims = 0
        self.current_ax = None
        self.current_folder = None
        
        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        
        # Create UI elements
        self.create_controls()
        self.create_labels_section()
        self.create_plot_area()

    def create_controls(self):
        # Control buttons frame
        control_frame = QWidget()
        control_layout = QHBoxLayout(control_frame)
        
        # Buttons
        self.browse_btn = QPushButton('Browse Folder')
        self.browse_btn.clicked.connect(self.load_folder)
        self.save_btn = QPushButton('Save Plot')
        self.save_btn.clicked.connect(self.save_plot)
        
        # File selector
        self.file_selector = QComboBox()
        self.file_selector.setPlaceholderText('Select file...')
        self.file_selector.currentIndexChanged.connect(self.file_selection_changed)
        self.file_selector.view().setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.file_selector.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        
        # Plot type combo box
        self.plot_combo = QComboBox()
        self.plot_combo.addItem('Select Plot Type')
        self.plot_combo.currentTextChanged.connect(self.plot_data)
        
        control_layout.addWidget(self.browse_btn)
        control_layout.addWidget(self.save_btn)
        control_layout.addWidget(QLabel('Files:'))
        control_layout.addWidget(self.file_selector)
        control_layout.addWidget(QLabel('Plot Type:'))
        control_layout.addWidget(self.plot_combo)
        self.main_layout.addWidget(control_frame)

    def create_labels_section(self):
        # Labels frame
        labels_frame = QWidget()
        labels_layout = QHBoxLayout(labels_frame)
        
        # Title and labels
        self.title_edit = QLineEdit()
        self.xlabel_edit = QLineEdit()
        self.ylabel_edit = QLineEdit()
        
        # Connect text changes
        self.title_edit.textChanged.connect(self.update_plot_labels)
        self.xlabel_edit.textChanged.connect(self.update_plot_labels)
        self.ylabel_edit.textChanged.connect(self.update_plot_labels)
        
        
        labels_layout.addWidget(QLabel('Title:'))
        labels_layout.addWidget(self.title_edit)
        labels_layout.addWidget(QLabel('X Label:'))
        labels_layout.addWidget(self.xlabel_edit)
        labels_layout.addWidget(QLabel('Y Label:'))
        labels_layout.addWidget(self.ylabel_edit)
        
        # define the size of the labels frame
        self.main_layout.addWidget(labels_frame)

    
    def load_folder(self):
        folder_dialog = QFileDialog()
        folderpath = folder_dialog.getExistingDirectory(
            self, 'Select Folder with HDF5 Files'
        )
        
        if not folderpath:
            return
            
        try:
            self.current_folder = folderpath
            self.file_selector.clear()
            
            # Find all .h5 files
            h5_files = [f for f in os.listdir(folderpath) if f.endswith('.h5')]
            # all the files end with xxxxxx.h5 so we can use this to order them by the number
            def sort_key(filename):
                try:
                    # Split filename into parts and get the numeric portion
                    base = os.path.splitext(filename)[0]  # Remove .h5
                    numeric_part = base.split('-')[-1]    # Get last part after -
                    return int(numeric_part)
                except (IndexError, ValueError):
                    return 0  # Fallback for malformed filenames

            h5_files.sort(key=sort_key)
            
            if not h5_files:
                raise ValueError('No HDF5 files found in selected folder')
            
            self.file_selector.addItems(h5_files)
            self.file_selector.setCurrentIndex(0)
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def file_selection_changed(self, index):
        '''Handle file selection change in the combo box'''
        if index >= 0 and self.current_folder:
            filename = self.file_selector.itemText(index)
            self.process_file(filename)

    def process_file(self, filename):
        try:
            filepath = os.path.join(self.current_folder, filename)
            gridfile = OsirisGridFile(filepath)
            self.dims = len(gridfile.axis)
            self.type = gridfile.type
            
            if self.type == 'grid':
                if self.dims == 1:
                    x = np.arange(gridfile.grid[0], gridfile.grid[1], gridfile.dx)
                    self.xlabel_edit.setText(r'$%s$ [$%s$]' % (gridfile.axis[0]['long_name'], gridfile.axis[0]['units']))
                    self.ylabel_edit.setText(r'$%s$ [$%s$]' % (gridfile.label, gridfile.units))
                    self.data_info = (x, gridfile.data)
                elif self.dims == 2:
                    x = np.arange(gridfile.grid[0][0], gridfile.grid[0][1], gridfile.dx[0])
                    y = np.arange(gridfile.grid[1][0], gridfile.grid[1][1], gridfile.dx[1])
                    self.xlabel_edit.setText(r'$%s$ [$%s$]' % (gridfile.axis[0]['long_name'], gridfile.axis[0]['units']))
                    self.ylabel_edit.setText(r'$%s$ [$%s$]' % (gridfile.axis[1]['long_name'], gridfile.axis[1]['units']))
                    self.data_info = (x, y, gridfile.data)
                elif self.dims == 3:
                    raise ValueError('3D not supported yet')
                else:
                    raise ValueError('Unsupported dimensionality')
                
                self.title_edit.setText(r'$%s$ [$%s$]' %( gridfile.label, gridfile.units))
                self.update_plot_menu()
                self.plot_data()
                
            else:
                QMessageBox.information(self, 'Info', f'{self.type} data not supported yet')
                
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def create_plot_area(self):
        # Matplotlib figure and canvas
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.main_layout.addWidget(self.canvas)

    def update_plot_labels(self):
        if self.current_ax:
            self.current_ax.set_xlabel(self.xlabel_edit.text())
            self.current_ax.set_ylabel(self.ylabel_edit.text())
            self.figure.suptitle(self.title_edit.text())
            self.canvas.draw()

    def plot_data(self):
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
        plot_type = self.plot_combo.currentText()
        
        if 'Line' in plot_type:
            self.current_ax.plot(x, data)
        elif 'Scatter' in plot_type:
            self.current_ax.scatter(x, data)
            
        self.current_ax.set_xlabel(self.xlabel_edit.text())
        self.current_ax.set_ylabel(self.ylabel_edit.text())
        self.figure.suptitle(self.title_edit.text())

    def plot_2d(self):
        x, y, data = self.data_info
        self.current_ax = self.figure.add_subplot(111)
        plot_type = self.plot_combo.currentText()
        
        if 'Quantity' in plot_type:
            img = self.current_ax.imshow(data.T, extent=(x[0], x[-1], y[0], y[-1]), origin='lower', aspect='auto')
            self.figure.colorbar(img)
        elif 'Integral' in plot_type:
            avg = integrate(transverse_average(data), x[-1]/len(x))
            self.current_ax.plot(x, avg)
        elif 'Transverse' in plot_type:
            avg = transverse_average(data)
            self.current_ax.plot(x, avg)
        elif 'Phase' in plot_type:
            img = self.current_ax.imshow(np.abs(-data.T), extent=(x[0], x[-1], y[0], y[-1]), origin='lower', aspect='auto', norm=LogNorm())
            self.figure.colorbar(img)
        
        self.current_ax.set_xlabel(self.xlabel_edit.text())
        self.current_ax.set_ylabel(self.ylabel_edit.text())
        self.figure.suptitle(self.title_edit.text())

    def update_plot_menu(self):
        
        # Save current plot type before clearing
        current_plot_type = self.plot_combo.currentText()
        self.plot_combo.clear()
        
        # Determine items based on dimensions
        if self.dims == 1:
            items = ['Line Plot', 'Scatter Plot']
        elif self.dims == 2:
            items = ['Quantity Plot', 'T. Average Integral', 'Transverse Average', 'Phase Space']
        else:
            items = []
        
        self.plot_combo.addItems(items)
    
        # Restore previous selection if possible
        if current_plot_type in items:
            self.plot_combo.setCurrentText(current_plot_type)
        else:
            self.plot_combo.setCurrentIndex(0 if items else -1)

    def save_plot(self):
        file_dialog = QFileDialog()
        filepath, _ = file_dialog.getSaveFileName(
            self, 'Save Plot', '', 'PNG Files (*.png);;PDF Files (*.pdf)'
        )
        
        if filepath:
            self.figure.savefig(filepath, dpi=800, bbox_inches='tight')

def LAVA():
    app = QApplication(sys.argv)
    window = LAVA_Qt()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    LAVA()