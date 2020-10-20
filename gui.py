"""
Tristan Anderson
tja1015@wildats.unh.edu
tris31299@gmail.com

Proceed Formally.
"""
from slifercal import sliferCal
import gc # garbage
import tkinter as tk
import tkinter.scrolledtext as scrolledtext
from tkinter import filedialog
from statistics import mode
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import datetime, pandas, os
"""
I have almost no practice with GUIs. All this stuff is self taught
and found in the library's documentation.
"""

class sliferCalGUI(tk.Tk):                # Class
    """
        Pretend this is main()
    """
    def __init__(self, *args, **kwargs):            # Method
        
        tk.Tk.__init__(self, *args, **kwargs)
        window = tk.Frame(self)

        window.pack(side="top", fill="both", expand=True)

        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0,weight=1)

        self.frames = {}                            # Attribute            

        for F in (File_Selector, Omni_View, Key_Word, Calibrate):
            frame = F(window, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(cont=File_Selector)

    def show_frame(self, **kwargs):
        cont = kwargs.pop('cont', False)

        instance = kwargs.pop('instance', sliferCal())


        frame = self.frames[cont]

        frame.fetch_instance(instance)
        
        frame.tkraise()


class File_Selector(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.guiTitle = tk.Label(self, text="Slifer-Cal GUI")
        self.guiTitle.grid(row=1, column=1)

        self.fileframe = tk.LabelFrame(self, text="Data File Selector")
        self.fileframe.grid(row=2, column=1)

        self.options = tk.LabelFrame(self, text="Utilities")
        self.options.grid(row=3, column=1)


    def fetch_instance(self, instance):
        self.instance = instance
        self.populate_toggleables()
        self.spawn_utilities()

    def populate_toggleables(self):
        self.get_data_frame = tk.LabelFrame(self.fileframe, text="Raw Data")
        self.get_data_frame.grid(row=1, column=1)
        self.get_data_button = tk.Button(self.get_data_frame, text="Select DAQ File", command=self.datafileDialog)
        self.get_data_button.grid(row=1, column=1)


    def datafileDialog(self):
        ftyps = (("DAQ File", "*.csv"),("all files","*.*"))
        self.dffilename = filedialog.askopenfilename(initialdir =  "$HOME", title = "Select A File", filetypes = ftyps)
        self.dflabel = tk.Label(self.get_data_frame, text = "")
        self.dflabel.grid(column = 1, row = 1)
        self.dflabel.configure(text = self.dffilename)
        if self.instance.update_datafile(datafile_location=self.dffilename, gui=True):
            print("Datafile Loaded")
            if not self.instance.logbook_status():
                self.get_logbook_frame = tk.LabelFrame(self.fileframe, text="Logbook File")
                self.get_logbook_frame.grid(row=2, column=1)
                self.get_logbook_button = tk.Button(self.get_logbook_frame, text="Select Logbook File", command=self.logbookfileDialog)
                self.get_logbook_button.grid(row=1, column=1)
        else:
            print("**ERROR: Data file was not loaded into slifercal.")


    def logbookfileDialog(self):
        ftyps = (("DAQ File", "*.csv"),("all files","*.*"))
        self.logbookfilename = filedialog.askopenfilename(initialdir =  "$HOME", title = "Select A File", filetypes = ftyps)
        self.lblabel = tk.Label(self.get_logbook_frame, text = "")
        self.lblabel.grid(column = 1, row = 1)
        self.lblabel.configure(text = self.logbookfilename)
        if self.instance.load_logbook(newpath=self.logbookfilename, gui=True):
            print("Logbook updated by user.")
        else:
            print("**Warning: Logbook update failed.")
    
    def spawn_utilities(self):
        self.omniview_button_frame = tk.LabelFrame(self.options, text="Date-Based Data Viewer")
        self.omniview_button_frame.grid(column=1)
        self.omniview_button = tk.Button(self.omniview_button_frame, text="Omni-View", command=self.omni_continue)
        self.omniview_button.grid(column=1)

        self.keyword_button_frame = tk.LabelFrame(self.options, text='Keyword Data Viewer')
        self.keyword_button_frame.grid(column=1)
        self.keyword_button = tk.Button(self.keyword_button_frame, text="Keyword Lookup")
        self.keyword_button.grid(column=1)

        self.calibrate_button_frame = tk.LabelFrame(self.options, text="Stable Region Finder")
        self.calibrate_button_frame.grid(column=1)
        self.calibrate_button = tk.Button(self.calibrate_button_frame, text="Thermometry Calibration")
        self.calibrate_button.grid(column=1)


    def omni_continue(self):
        self.controller.show_frame(cont=Omni_View, instance=self.instance)






class Omni_View(File_Selector):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        
        self.controller = controller

    def fetch_instance(self, instance):
        self.instance = instance
        self.populate_toggleables()


    def populate_toggleables(self):
        self.toggleables_frame = tk.LabelFrame(self, text="Options")
        self.toggleables_frame.grid(column=2, row=1)
        self.title = tk.Label(self, text="Omni-View")
        self.title.grid(column=2, row=1)

        #self.channel_frame = tk.Frame(self, text="Y-axis Selection")
        #self.channel_frame.grid(column=1, row =1)
        print(self.instance.get_columns())

        self.generate_channels()

    def generate_channels(self):
        cols = elf.instance.get_columns()
        self.channel_frame = tk.Frame(self, text="Y-Axis")




class Key_Word(File_Selector):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        
        self.controller = controller

class Calibrate(File_Selector):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        
        self.controller = controller

        


root = sliferCalGUI()
root.mainloop()
