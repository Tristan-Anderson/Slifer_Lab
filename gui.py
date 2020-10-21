"""
Tristan Anderson
tja1015@wildats.unh.edu


Proceed Formally.
"""
from slifercal import sliferCal
import tkinter as tk
import tkinter.scrolledtext as scrolledtext
from tkinter import filedialog
from statistics import mode
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import datetime, pandas, os, gc, matplotlib

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

        self.frame1 = tk.Frame(self)
        self.frame1.grid(column=1,row=2)

        self.toggleables_frame = tk.LabelFrame(self.frame1, text="Options")
        self.toggleables_frame.grid(column=1, row=1)

        self.ychannel_frame = tk.LabelFrame(self.toggleables_frame, text="Y-Axis Selection")
        self.ychannel_frame.grid(column=1, row=1)

        self.x_frame = tk.LabelFrame(self.toggleables_frame, text="X-Axis Selector")
        self.x_frame.grid(column=1, row=2)

        self.timeframe = tk.LabelFrame(self.toggleables_frame, text="Select Time Window to Graph")
        self.timeframe.grid(column=1,row=3)

        self.finalframe = tk.LabelFrame(self.toggleables_frame, text="Finalize")
        self.finalframe.grid(column=1,row=4)

    def fetch_instance(self, instance):
        self.instance = instance
        self.populate_toggleables()

    def execute(self):
        year=int(self.minyear.get())
        month=int(self.minmonth.get())
        day=int(self.minday.get())
        hour=int(self.minhour.get())
        minute=int(self.minminute.get())
        second=int(self.minsecond.get())
        
        u_start= datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

        year=int(self.maxyear.get())
        month=int(self.maxmonth.get())
        day=int(self.maxday.get())
        hour=int(self.maxhour.get())
        minute=int(self.maxminute.get())
        second=int(self.maxsecond.get())

        u_end = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

        thermistors = self.get_selected_thermistors()
        xaxis = self.xaxisvalue.get()
        comments = True if self.comments.get()=='1' else False

        self.figure = self.instance.omniview_gui(u_start,u_end, thermistors, xaxis, comments=comments, save_fig=False, gui=True)
        self.graph = tk.LabelFrame(self, text="Graph")
        self.graph.grid(column=2, row=2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
    
    def populate_toggleables(self):
        self.generate_ychannels()
        self.generate_ylabelentry()

        self.generate_xpulldown()

        self.generate_timeselection()

        
        self.startbutton = tk.Button(self.finalframe, text="Start", command=self.execute)
        self.startbutton.grid(column=1,row=1)

        self.comments = tk.StringVar(value='1')
        self.commentsbox = tk.Checkbutton(self.finalframe, text="Comments", variable=self.comments, onvalue='1', offvalue='0')
        self.commentsbox.grid(column=2,row=1)

        self.backbutton = tk.Button(self, text="Back", command=lambda: self.controller.show_frame(cont=File_Selector))
        self.backbutton.grid(column=1, row=3)

    def get_selected_thermistors(self):
        thermistors = []
        for key in self.checkbuttons:
            if self.checkbuttons[key].get() == '1':
                thermistors.append(key)
        return thermistors

    def generate_timeselection(self):
        y,m,d,h,M,s = self.instance.get_timespans()

        self.minyear,self.minmonth,self.minday,\
        self.minhour,self.minminute,\
        self.minsecond = tk.StringVar(value=min(y)), tk.StringVar(value=min(m)),\
            tk.StringVar(value=min(d)), tk.StringVar(value=min(h)), tk.StringVar(value=min(M)),\
            tk.StringVar(value=min(s))

        self.maxyear,self.maxmonth,self.maxday,\
        self.maxhour,self.maxminute,\
        self.maxsecond = tk.StringVar(value=max(y)), tk.StringVar(value=max(m)),\
            tk.StringVar(value=max(d)), tk.StringVar(value=max(h)), tk.StringVar(value=max(M)),\
            tk.StringVar(value=max(s))
     
        self.minlabelframe = tk.LabelFrame(self.timeframe,text="Approximate Start Date")
        self.minlabelframe.grid(column=1, row=1, padx=30)

        self.mindateframe = tk.LabelFrame(self.minlabelframe, text="Date")
        self.mindateframe.grid(column=1, row=1)

        self.mintimeframe = tk.LabelFrame(self.minlabelframe, text="Time")
        self.mintimeframe.grid(column=2,row=1)

        self.miny = tk.Label(self.mindateframe, text="Year:")
        self.miny.grid(column=0, row=1)
        self.minM = tk.Label(self.mindateframe, text="Month:")
        self.minM.grid(column=0, row=2)
        self.mind = tk.Label(self.mindateframe, text="Day:")
        self.mind.grid(column=0, row=3)
        self.minyearpulldown = tk.OptionMenu(self.mindateframe, self.minyear, *y)
        self.minyearpulldown.grid(column=1, row=1)
        self.minmonthpulldown = tk.OptionMenu(self.mindateframe, self.minmonth, *m)
        self.minmonthpulldown.grid(column=1, row=2)
        self.mindaypulldown = tk.OptionMenu(self.mindateframe, self.minday, *d)
        self.mindaypulldown.grid(column=1,row=3)

        self.minh = tk.Label(self.mintimeframe, text='Hour:')
        self.minh.grid(column=0, row=1)
        self.minhourpulldown = tk.OptionMenu(self.mintimeframe, self.minhour, *h)
        self.minhourpulldown.grid(column=1, row=1)
        self.minm = tk.Label(self.mintimeframe, text="Minute:")
        self.minm.grid(column=0, row=2)
        self.minminutepulldown = tk.OptionMenu(self.mintimeframe, self.minminute, *M)
        self.minminutepulldown.grid(column=1, row=2)
        self.mins = tk.Label(self.mintimeframe, text="Second:")
        self.mins.grid(column=0, row=3)
        self.minsecondpulldown = tk.OptionMenu(self.mintimeframe, self.minsecond, *s)
        self.minsecondpulldown.grid(column=1,row=3)
        

        self.maxlabelframe = tk.LabelFrame(self.timeframe,text="Approximate End Date")
        self.maxlabelframe.grid(column=2, row=1, padx=30)

        self.maxdateframe = tk.LabelFrame(self.maxlabelframe, text="Date")
        self.maxdateframe.grid(column=1, row=1)

        self.maxtimeframe = tk.LabelFrame(self.maxlabelframe, text="Time")
        self.maxtimeframe.grid(column=2,row=1)

        self.maxy = tk.Label(self.maxdateframe, text="Year:")
        self.maxy.grid(column=0, row=1)
        self.maxM = tk.Label(self.maxdateframe, text="Month:")
        self.maxM.grid(column=0, row=2)
        self.maxd = tk.Label(self.maxdateframe, text="Day:")
        self.maxd.grid(column=0, row=3)
        self.maxyearpulldown = tk.OptionMenu(self.maxdateframe, self.maxyear, *y)
        self.maxyearpulldown.grid(column=1, row=1)
        self.maxmonthpulldown = tk.OptionMenu(self.maxdateframe, self.maxmonth, *m)
        self.maxmonthpulldown.grid(column=1, row=2)
        self.maxdaypulldown = tk.OptionMenu(self.maxdateframe, self.maxday, *d)
        self.maxdaypulldown.grid(column=1,row=3)

        self.maxh = tk.Label(self.maxtimeframe, text='Hour:')
        self.maxh.grid(column=0, row=1)
        self.maxhourpulldown = tk.OptionMenu(self.maxtimeframe, self.maxhour, *h)
        self.maxhourpulldown.grid(column=1, row=1)
        self.maxm = tk.Label(self.maxtimeframe, text="maxute:")
        self.maxm.grid(column=0, row=2)
        self.maxminutepulldown = tk.OptionMenu(self.maxtimeframe, self.maxminute, *M)
        self.maxminutepulldown.grid(column=1, row=2)
        self.maxs = tk.Label(self.maxtimeframe, text="Second:")
        self.maxs.grid(column=0, row=3)
        self.maxsecondpulldown = tk.OptionMenu(self.maxtimeframe, self.maxsecond, *s)
        self.maxsecondpulldown.grid(column=1,row=3)

    def generate_xpulldown(self):
        self.xaxisvalue = tk.StringVar(value="Time")
        self.xpulldown = tk.OptionMenu(self.x_frame, self.xaxisvalue, *self.plottable_ys)
        self.xpulldown.grid(column=1,row=1)
        
        """self.xentryframe = tk.Frame(self.x_frame)
                                self.xentryframe.grid(column=1, row=2)
                                self.xlabel = tk.StringVar(value="Time")
                        
                                self.xaxislabel = tk.Label(self.xentryframe, text="X-Axis Label")
                                self.xlabelentry = tk.Entry(self.xentryframe, textvariable=self.xlabel)
                                self.xaxislabel.grid(column=1, row=1)
                                self.xlabelentry.grid(column=2,row=1)"""

    def generate_ychannels(self):
        self.yaxisselection_subframe = tk.Frame(self.ychannel_frame)
        self.yaxisselection_subframe.grid(column=1,row=1)
        plottable_keywords = ["(Ohms)", "Magnet", "waves", "(K)", "level", "Torr", "Time", "Polarization"]
        self.buttons, self.checkbuttons = {},{}
        ycols = self.instance.get_columns()
        ycols_sorted = sorted(ycols, key=lambda word: (".R" not in word, word))
        
        self.plottable_ys = []
        for index,value in enumerate(ycols_sorted):
            if any(y in value for y in plottable_keywords):
                self.plottable_ys.append(value)
        
        for index,v in enumerate(self.plottable_ys):
            self.checkbuttons[v] = tk.StringVar(value='0')
            self.buttons[v] = tk.Checkbutton(self.yaxisselection_subframe, text=v, variable=self.checkbuttons[v], onvalue='1',offvalue='0')
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
