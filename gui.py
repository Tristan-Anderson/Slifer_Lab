"""
Tristan Anderson
tja1015@wildats.unh.edu


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
        self.omniview_button = tk.Button(self.omniview_button_frame, text="Omni-View", command= lambda: self.controller.show_frame(cont=Omni_View, instance=self.instance))
        self.omniview_button.grid(column=1)

        self.keyword_button_frame = tk.LabelFrame(self.options, text='Keyword Data Viewer')
        self.keyword_button_frame.grid(column=1)
        self.keyword_button = tk.Button(self.keyword_button_frame, text="Keyword Lookup")
        self.keyword_button.grid(column=1)

        self.calibrate_button_frame = tk.LabelFrame(self.options, text="Stable Region Finder")
        self.calibrate_button_frame.grid(column=1)
        self.calibrate_button = tk.Button(self.calibrate_button_frame, text="Thermometry Stable Region Finder")
        self.calibrate_button.grid(column=1)


class Omni_View(File_Selector):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.title = tk.Label(self, text="Omni-View")
        self.title.grid(column=1, row=1)

        self.toggleables_frame = tk.LabelFrame(self, text="Options")
        self.toggleables_frame.grid(column=1, row=2)

        self.ychannel_frame = tk.LabelFrame(self.toggleables_frame, text="Y-Axis Selection")
        self.ychannel_frame.grid(column=1, row=1)

        self.x_frame = tk.LabelFrame(self.toggleables_frame, text="X-Axis Selector")
        self.x_frame.grid(column=1, row=2)

        self.timeframe = tk.LabelFrame(self.toggleables_frame, text="Select Timewindow to Graph")
        self.timeframe.grid(column=1,row=3)
        
    def fetch_instance(self, instance):
        self.instance = instance
        self.populate_toggleables()

    def populate_toggleables(self):
        self.generate_ychannels()
        self.generate_ylabelentry()

        self.generate_xpulldown()

        self.generate_timeselection()


    def get_timespans(self, maxtime,mintime):
        y = []
        m = []
        d =[]
        h = []
        M = []
        S = []

        timedelta = maxtime-mintime
        try:
            for i in range(0,timedelta.years+1):
                step = (mintime+datetime.timedelta(years=i)).year
                if step not in y:
                    y.append(step)
        except:
            y = [x for x in range(mintime.year, maxtime.year+1)]


        try:
            for i in range(0,timedelta.months+1):
                step = (mintime+datetime.timedelta(months=i)).month
                if step not in m:
                    m.append(step)
        except:
            m = [x for x in range(mintime.month, maxtime.month+1)]


        try:
            for i in range(0,timedelta.days+1):
                step = (mintime+datetime.timedelta(days=i)).day
                if step not in d:
                    d.append(step)
        except:
            d = [x for x in range(mintime.day, maxtime.day+1)]

        try:
            for i in range(0,timedelta.hours+1):
                step = (mintime+datetime.timedelta(hours=i)).hour
                if step not in h:
                    h.append(step)
        except:
            h = [x for x in range(0, 24)]


        try:
            for i in range(0,timedelta.minutes+1):
                step = (mintime+datetime.timedelta(minutes=i)).minute
                if step not in M:
                    M.append(step)
        except:
            M = [x for x in range(0,60)]


        try:
            for i in range(0,timedelta.seconds+1):
                step = (mintime+datetime.timedelta(seconds=i)).second
                if step not in s:
                    s.append(step)
        except:
            s = [x*15 for x in range(0,5)]


        return y,m,d,h,M,s

    def generate_timeselection(self):
        maxtime,mintime = self.instance.gettimerange()

        y,m,d,h,M,s = self.get_timespans(maxtime,mintime)

        self.minyear,self.minmonth,self.minday,
        self.minhour,self.minminute,
        self.minsecond = tk.StringVar(value=min(y)), tk.StringVar(value=min(m)), 
            tk.StringVar(value=min(d)), tk.StringVar(value=min(h)), tk.StringVar(value=min(M)), 
                tk.StringVar(value=min(s))

        self.maxyear,self.maxmonth,self.maxday,
        self.maxhour,self.maxminute,
        self.maxsecond = tk.StringVar(value=max(y)), tk.StringVar(value=max(m)), 
            tk.StringVar(value=max(d)), tk.StringVar(value=max(h)), tk.StringVar(value=max(M)), 
                tk.StringVar(value=max(s))
        
        ###   Creating Timerange
        

        self.minlabelframe = tk.LabelFrame(self.timeframe,text="Approximate Start Date")
        self.minlabelframe.grid(column=1, row=1)
        
        


        self.maxlabelframe = tk.LabelFrame(self.timeframe, text="Approximate End Date")
        self.maxlabelframe.grid(column=1, row=1)

    def generate_xpulldown(self):
        self.xaxisvalue = tk.StringVar(value="Time")
        self.xpulldown = tk.OptionMenu(self.x_frame, self.xaxisvalue, *self.plottable_ys)
        self.xpulldown.grid(column=1,row=1)
        
        self.xentryframe = tk.Frame(self.x_frame)
        self.xentryframe.grid(column=1, row=2)
        self.xlabel = tk.StringVar(value="Time")

        self.xaxislabel = tk.Label(self.xentryframe, text="X-Axis Label")
        self.xlabelentry = tk.Entry(self.xentryframe, textvariable=self.xlabel)
        self.xaxislabel.grid(column=1, row=1)
        self.xlabelentry.grid(column=2,row=1)

    def generate_ychannels(self):
        self.yaxisselection_subframe = tk.Frame(self.ychannel_frame)
        self.yaxisselection_subframe.grid(column=1,row=1)
        plottable_keywords = ["(Ohms)", "Magnet", "waves", "(K)", "level", "Torr", "Time"]
        self.buttons, self.checkbutton = {},{}
        ycols = self.instance.get_columns()
        ycols_sorted = sorted(ycols, key=lambda word: (".R" not in word, word))
        
        self.plottable_ys = []
        for index,value in enumerate(ycols_sorted):
            if any(y in value for y in plottable_keywords):
                self.plottable_ys.append(value)
        
        for index,v in enumerate(self.plottable_ys):
            self.checkbutton[value] = tk.StringVar(value=0)
            self.buttons[v] = tk.Checkbutton(self.yaxisselection_subframe, text=v, variable=self.checkbutton[value], onvalue='1',offvalue='0')
            self.buttons[v].grid(column=index%3, row=int(index/3))
        #self.button_array =
        #for index,value in enumerate(ycols_sorted):

    def generate_ylabelentry(self):
        self.ylabelsubframe = tk.Frame(self.ychannel_frame)
        self.ylabelsubframe.grid(column=1,row=2)
        self.ylabel = tk.StringVar(value='Ohms')
        self.tk_YLabel = tk.Label(self.ylabelsubframe, text="Y-Axis Label")
        self.ylabelentry = tk.Entry(self.ylabelsubframe, textvariable=self.ylabel)
        self.tk_YLabel.grid(column=1,row=1)
        self.ylabelentry.grid(column=2, row=1)


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
