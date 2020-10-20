import time, pandas, numpy, copy, datetime, os, traceback, dateutil.parser, multiprocessing, re
from multiprocessing import Pool
from pandas.plotting import register_matplotlib_converters
from thermistor_profile import thermistor_profile as tp
from collections import namedtuple
timeRangeOverlap = namedtuple("timeRangeOverlap", ["start", "end"])
import _pickle as pickle
import matplotlib
from matplotlib import pyplot as plt
import gc


global fig_x_dim, fig_y_dim, dpi
fig_x_dim, fig_y_dim, dpi = 8,4.5, 300


class sliferCal(object):
    def __init__(self, processes=0, datafile_location=None, logbook_datafile_location=None, data_record_location='data_record.csv', gui=False):
        """
            Upon initalization we aquire the following:
                - Number of processing threads (Used for analysis)
                - The variable name in which the instance is stored from the traceback
            We recieve from calling, or default to the following:
                - Datafile location: default data.csv, is required for you to do anything with
                    this program, so it will terminate if you haven't given it a good one.
                - Logbook location: default None, but will search for 'logbook.csv'
                    not required for 

        """
        print("Slifer Cal initalizing...")
        plt.style.use('fast')
        self.data_record_location=data_record_location        
        register_matplotlib_converters() # Calling the calibrate method without this here told me to put this here.
        self.trslat = {0: "RT", 1: "LN2", 2: "LHe"}
        self.datafile_location = datafile_location
        self.logbook_datafile_location = logbook_datafile_location
        (filename,line_number,function_name,text)=traceback.extract_stack()[-2] # NAME of THIS CLASS (for pickling later if needed - but probably not.)
        def_name = text[:text.find('=')].strip()
        self.name = def_name
        if processes == 0:
            self.processes = int(8*multiprocessing.cpu_count()/10)
            print(self.processes, "Processing threads available")
        else:
            self.processes = processes
        print("Initalization succesful.")
    
    def get_columns(self):
        trash = self.df.columns.tolist()
        while ("" in  trash):
            trash.remove("")
        return trash

    def get_thermometry_data(self):
        ################################################
        """
            Takes the dataframe and drops all columns
            except the timestamp, and resistance
                        thermometry data
                                                     """
        ################################################
        indexes_that_are_to_be_deleted = []
        rows_to_be_deleted = []
        column_names = list(self.df)
        column = ["Time"]

        for name in self.df:
            if '.R' in name and ('M' in name or '.F' in name) and "Ohms" in name: # Feature that allows user to throw in raw .csv of entire cooldown
                    column.append(name)                         # Probably no longer necessary for Slifer Lab, but may be useful to
        self.df = self.df[column]

        for index in self.df:
            if index != "Time":
                self.df[index] = self.df[index].astype(float)   

        self.df["Time"] = pandas.to_datetime(self.df["Time"])

        print("Data has been reduced to thermometry data.")


    def complete_keyword(self, timeit, keywords, thermistors, rangeshift=1, range_length=None):
        """
            Ready-to-go prebuilt method.

                - Grabs experimental data and cleans it.
                - Grabs logbook.
                - Grabs the list of keywords, and finds keyword hits.
                - Plots the data for the keyword hits for each thermistor.
                - Prints time for execution in seconds.
        """
        if timeit:
            readings = time.time()
        self.load_experimental_data()
        if timeit:
            readingf=time.time()
            cleans = time.time()
        self.get_thermometry_data()
        if timeit:
            cleanf=time.time()
            analysiss = time.time()
        self.range_election(rangeshift=rangeshift, range_length=range_length)
        if timeit:
            analysisf = time.time()
            plottings = time.time()
        self.plot_keyword_hits(keywords, thermistors)
        if timeit:
            plottingf=time.time()
        print("Reading", readingf-readings, "Analysis", analysisf-cleans, "Plotting", plottingf-plottings)


    def __debug_attribute(self, obj):
        ############################################
        """
           Prints any attribute you feed it to a
                          file.
                                                 """
        ############################################
        import pprint
        with open("debug.txt", 'w') as fout:
            pprint.pprint(obj, fout)
        print("Printed object to file: debug.txt")


    def find_stable_regions(self, rangeshift=1):
        ##############################################
        """
            It analyses data, then saves the data
                    in kernels for graphing. 
                                                   """
        ##############################################
        self.load_experimental_data()
        self.get_thermometry_data()
        self.range_election(rangeshift=rangeshift)
    

    def load_persistence_data(self, file_location=None):
        ###################################################
        """
            Unpickeler for data saved with intent of 
                            persistence.
                                                        """
        ###################################################
        # Loads persistence data
        if file_location == None:
            
            # If no persistence path was provided, search for one 

            print("No file path provided. Finding most recent pickle...")
            list_of_files = []
            desired_file = ""
            big_date = 0

            for file in os.listdir(os.getcwd()):
                if file.endswith(".pk") and "keeper" in file:
                    list_of_files.append(file)
                    date = int("".join(list(file)[-17:-3]))
                    if date > big_date:
                        big_date = int(date)
            big_date = str(big_date)

            for file in list_of_files:
                if big_date in file:
                    self.kd_name = file
            print("Pickle found.  Reading file...")
            
            try:
                with open(self.kd_name, 'rb') as fin:
                    self.keeper_data = pickle.load(fin)
            except AttributeError:
                print("No parsed datafile. Trying to search for datafile one last time.")
                self.load_experimental_data()
            
            print("File Read")


        elif file_location != None:
            # If persistence path was provided
            # Load it
            self.kd_name = file_location
            print("Reading file")
            with open(file_location, 'rb') as fin:
                self.keeper_data = pickle.load(fin)
            print("File read")


    def logbook_status(self):
        #Check for empty file
        try:
            if len(self.logbook_df) == 0:
                return False
        except AttributeError:
            self.load_logbook()
            if len(self.logbook_df) == 0:
                return False
        return True


    def load_logbook(self, newpath=None, gui=False):
        if newpath is not None:
            self.logbook_datafile_location = newpath
        if self.logbook_datafile_location == None:
            print("No loogbook path was used to initalize the instance!\nAssuming logbook is \"logbook_data.csv\" \nSearching local directory:")
            self.logbook_datafile_location = "logbook.csv"
        try:
            with open(self.logbook_datafile_location,'r') as f:
                self.logbook_df = pandas.read_csv(f, sep='\t')
            self.logbook_df["Time"] = pandas.to_datetime(self.logbook_df["Time"]) 
            print("File found. Comments File loaded.")
            if gui:
                return True
        except:
            print("**WARNING: Logbook was not found. Empty DF created.")
            self.logbook_df = pandas.DataFrame(columns=["Time", "Comment"])
            if gui:
                return False


    def __nearest(self, test_val, iterable): 
        # In an interable data-structure, find the nearest to the 
        # value presented.
        return min(iterable, key=lambda x: abs(x - test_val))


    def keyword(self, keywords, thermistors=None, persistance=True, kelvin=False):
        self.load_experimental_data()
        self.get_thermometry_data()
        self.__plot_keyword_hits(keywords, thermistors=thermistors, persistance=persistance, kelvin=kelvin)


    def keyword_nearest(self, test_val, iterable, tag):
        # Based on the __nearest() method, 
        # but returns the critical range information for our kernels.
        print("Looking for the nearest date to", test_val, "from logbook index", tag, "in raw-data file")
        print(type(iterable))
        nearest_time = min(iterable, key=lambda x: abs(x - test_val))
        df_index = self.df.index[self.df["Time"] == nearest_time][0]
        return [tag, nearest_time, df_index]


    def range_election(self, rangeshift=1, range_length=None):
        #############################################################
        """
                            This method slices
           data in self.df, skips the slice if there are any zero 
           vals the the slice. Then takes the files average, std,
           and assigns it a "Temperature range" based on ballpark
                       estimates found in the function:
                      what_temperature_range_are_we_in()

              IF THERMOMETRY GROUPINGS ARE NO LONGER ACCURATE
                      EDIT THE FUNCTION STATED ABOVE.
                                                                  """
        #############################################################
        if range_length == None:
            print("No range length given. Making range length one hour long.")
            self.__time_steps_suck()
            self.range_end = int(3600/self.average_timestep)
        else:
            self.range_end = range_length
        self.keeper_data = {}
        with Pool(processes=self.processes) as pool:
            result_objects = [pool.apply_async(self.range_election_metric, args=(column_name, rangeshift)) for column_name in self.df]
            pool.close()
            pool.join()
        results = [r.get() for r in result_objects if r.get() != False]
        for dictionary in results:
            self.keeper_data.update(dictionary)
        self.time_for_range_election_pickle = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        with open('keeper_data_original_'+self.time_for_range_election_pickle+'.pk', 'wb') as handle:
            pickle.dump(self.keeper_data, handle)
        print("Dictionary pickled as :", 'keeper_data_original_'+self.time_for_range_election_pickle+'.pk')
        self.kd_name = 'keeper_data_original_'+self.time_for_range_election_pickle+'.pk'


    def range_election_metric(self,column_name, rangeshift):
        #####################################################
        """
            Parallelizable function that is called once
            per column of data that searches through that
            column for a pattern. In this case, it looks
            for regions of data that that are 'flat' by
                    their standard deviation
                                                          """
        #####################################################
        temp_RT_dict = {}
        temp_LN2_dict = {}
        temp_LHe_dict = {}
        keeper_data = {}

        if column_name == "Time":
            print("Starting data analysis")
            return False
        elif column_name != "Time":
            print("Analyzing",column_name)
            nth_datarange = []
            range_begin = 0
            range_end = self.range_end
            what_shift_is_this = 0
            nth_range = 0
            while range_end < len(self.df[column_name]): # Data Slice
                if what_shift_is_this % int(len(self.df[column_name])/4) == 0 and what_shift_is_this != 0:
                    print(
                        what_shift_is_this, "of",
                        int((len(self.df[column_name])-self.range_end)/rangeshift),
                        "Ranges averaged.")
                data_slice = self.df[column_name][range_begin:range_end]
                if not data_slice.isnull().values.any(): # If there are no zeros in the range, average the range
                    avg = numpy.average(data_slice) # If the range is within what we are looking for 
                    std = numpy.std(data_slice)
                    nth_datarange=[std, avg, range_begin,range_end] # Save some stuff about it
                    temperature_range = what_temperature_range_are_we_in(avg,column_name) # Make another index for the following dataset
                    if temperature_range == 0:
                        temp_RT_dict[nth_range] = nth_datarange
                    elif temperature_range == 1:
                        temp_LN2_dict[nth_range] = nth_datarange
                    elif temperature_range == 2:
                        temp_LHe_dict[nth_range] = nth_datarange
                nth_range += 1
                range_end += rangeshift # Slice another range
                range_begin += rangeshift
                what_shift_is_this += 1
            keeper_data[column_name] = {
                "RT":pandas.DataFrame.from_dict(temp_RT_dict, orient='index', columns=["STD", "AVG", "RANGE START", "RANGE END"]),
                "LN2":pandas.DataFrame.from_dict(temp_LN2_dict, orient='index', columns=["STD", "AVG", "RANGE START", "RANGE END"]),
                "LHe":pandas.DataFrame.from_dict(temp_LHe_dict, orient='index', columns=["STD", "AVG", "RANGE START", "RANGE END"])} # Save data on thermistor; continue
            return keeper_data


    def update_datafile(self, datafile_location='', delimeter='\t', gui=False):
        # Read the data into the instance. 
        # This will be useful for the gui.
        self.datafile_location = datafile_location
        self.load_experimental_data(delimeter=delimeter)
        if gui:
            try:
                self.df
            except AttributeError:
                return False
            return True
        #self.get_thermometry_data()


    def __ensuretime(self):
        if type(self.df.loc[1, "Time"]) == numpy.float64: 
            # If david did not convert time from 1904/12/31 20:00:00, 
            # then do the conversion and put it into datetime.
            self.df["Time"] = self.df["Time"].apply(self.__time_since_1904)
        else:
            self.df["Time"] = pandas.to_datetime(self.df["Time"])


    def get_timespans(self):
        times = self.df["Time"]
        maxtime, mintime = max(times), min(times)

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
            s = [x*15 for x in range(0,4)]


        return y,m,d,h,M,s


    def load_experimental_data(self, delimeter='\n', new=True):
        # Reads in the raw data, and tries to find comments that go with it.
        self.thermistor_names = []
        if self.datafile_location == None:
            print(
                "No datafile was used to initalize the instance!\
                \nAssuming filename is \"data.csv\"")
            self.datafile_location = "data.csv"

        datafile = [] 
        with open(self.datafile_location,'r') as f:
            if new:
                for index, line in enumerate(f):
                    if index == 0:
                        header = line.split('\t')[:-1]
                        continue
                    l = line.split('\t')
                    datafile.append(l[:len(header)])
                    
                self.df = pandas.DataFrame(datafile, columns=header)
            else:
                self.df = pandas.read_csv(f, delimiter=delimeter)

       
        print("File loaded.")

        self.__ensuretime()

        try:
            self.logbook_df = self.df['Time', "Comment"]
        except KeyError as e:
            try:
                self.df["Comment"]
            except KeyError as c:
                print("No comments were provided in the datafile. Searching elsewhere...")
                self.load_logbook()
        except TypeError as e:
            self.load_logbook()
        for column in self.df:
            if column not in ["Time", "Comment"]:
                self.thermistor_names.append(column)

        self.logbook_coinsides()


    def logbook_coinsides(self):
        # Do the logbook timestamps coinside with the raw data's timestamps?
        try:
            lb_time = self.logbook_df["Time"]
            lb_min = min(lb_time)
            lb_max =  max(lb_time)
            lbrange = timeRangeOverlap(start=lb_min, end=lb_max)
        except KeyError as e:
            print(e)
            print("**Advisory: Unable to determine if the logbook belongs to the raw datatset.")
            return False

        try:
            df_time = self.df["Time"]
            df_min = min(df_time)
            df_max = max(df_time)
            df_range = lbrange = timeRangeOverlap(start=df_min, end=df_max)
        except KeyError as e:
            print(e)
            print("**Advisory: Unable to determine if the logbook belongs to the raw datatset.")
            return False

        # https://stackoverflow.com/questions/9044084/efficient-date-range-overlap-calculation-in-python
        latest_start = max(lbrange.start, df_range.start)
        earliest_end = min(lbrange.end, df_range.end)
        delta = earliest_end - latest_start

        if delta >= datetime.timedelta(seconds=5):
            print("Logbook belongs with dataset, with a total overlap time of",delta)
                                    

    def find_top_n_ranges(self, n=10):
        ###############################################################
        """
           This is the self.keeper_data parser that sorts through 
           self.keeper_data and creates a new dictionary containing 
                     the n most stable regions of data.
                                                                    """
        ###############################################################
        print("Do you have the right data?")
        try:
            self.keeper_data
        except AttributeError:
            print("Nope. You dont.")
            self.find_stable_regions()
            print("Now you have the right data.")
        
        print("Searching for n-best...")
        self.load_persistence_data()
        self.load_experimental_data()
        self.n_best = {}

        for thermistor in self.keeper_data:
            calibration_list = {}
            for temperature in self.keeper_data[thermistor]:
                try:
                    calibration_list[temperature]= self.keeper_data[thermistor][temperature].nsmallest(n, "STD", keep="first")
                except:
                    print("No Calibration point present for", thermistor, "in", temperature, " range.")
                    continue
                print("Located flattest", temperature, "datapoint for thermistor:", thermistor)
            self.n_best[thermistor] = calibration_list


    def find_keyword_hits(self, keywords, thermistors=None):

        #############################################################
        """ 
            Finds what indices of the logbook_df contain keywords
            then map indices of hits to data-set. Packs those into
            a kernel for graphing.
            - Remove kernels that have overlapping points within 
                45 minutes of the center of each point
                (This is an optimization problem that may 
                never come to fruition)

                "Dense"
                    Pros :
                        -Most ammount of information, least
                            ammount of graphs
                    Cons :
                        -No longer have ability to scrutanize each
                            instance


                                                                   """
        ##############################################################
        
        self.load_experimental_data()
        self.keyword_hits = {}
        df_nearest_indices = []
        logbook_indices = [] # The indices of the logbook_df that contain keywords
        
        print("Finding Keywords in comments....")
        for index, row in self.logbook_df.iterrows():
            if any(x in str(row["Comment"]) for x in keywords): # If true; we found a keywords
                logbook_indices.append(index)
        print("Keywords Found")
        
        print(
              "Asynchronously Parallelizing", len(logbook_indices),
              "Queries over", len(self.df["Time"]), 
              "rows.\nExpecting 10k rows/s. Estimated time:", 
              (len(logbook_indices)/self.processes)*len(self.df["Time"])/(10000), "seconds.\n\n")
        time.sleep(0.5)

        if thermistors is not None:
            self.thermistor_names = thermistors

        start = time.time()
        with Pool(processes=self.processes) as pool: # ~20 Seconds per Query at 3.05 GHz clock-speed.
            result_objects = [pool.apply_async(
                              self.keyword_nearest, 
                              args=(self.logbook_df.loc[logbook_index, "Time"],
                              self.df["Time"], logbook_index)) for logbook_index in logbook_indices]
            pool.close()
            pool.join()
        results = [r.get() for r in result_objects if r.get() != False]
        end = time.time()
        
        print(
            "\n\n",len(logbook_indices), "Queries completed in", end-start, "seconds.\n",
            "Estimated", (len(logbook_indices)/self.processes)*len(self.df["Time"])/(10000),
            "seconds. Prediction within", ((end-start)/((len(logbook_indices)/self.processes)*len(self.df["Time"])/(10000)) - 1)*100, "percent of measured value.")
        
        for thermistor in self.thermistor_names: # Creating Kernels here.
            kernel_dicts = {} # [std, avg, range_begin,range_end]
            for result in results: # [logbook_index, nearest_df_time, data_file_index]
                kernel_dicts[result[0]] = [1, 1, result[2], result[2]]
            self.keyword_hits[thermistor] = {"KEYWORD":pandas.DataFrame.from_dict(kernel_dicts, orient='index', columns=["STD", "AVG", "RANGE START", "RANGE END"])}


    def time_since_1904(self,sec): 
        # LabVIEW conveniently used seconds from "1 January, 1904" as time-stamp.
        self.begining_of_time = datetime.datetime(1903, 12, 31)+datetime.timedelta(seconds=72000) # Ellie noted a -4 hour time difference from UTC.
        return self.begining_of_time + datetime.timedelta(seconds=sec) # Assume accuracy no finer than a minute.

    
    def __time_steps_suck(self):
        ######################################
        """
           This will find the average
           time step in between the first
           10 data-points, and use that 
           to search hour(ish) long slices
                   of the data.
                                           """
        ######################################
        df_times = []
        diff_times = []
        #self.df.loc[1, "Time"]
        if type(self.df.loc[1, "Time"]) == str:
            for i in range(1,10):
                ent = dateutil.parser.parse(self.df.loc[i+1, "Time"])-dateutil.parser.parse(self.df.loc[1, "Time"])
                df_times.append(ent.total_seconds())
            self.average_timestep = numpy.mean(df_times)
        elif type(self.df["Time"][1]) == numpy.float64:
            for i in range(1, 10):
                df_times.append(self.df.loc[i+1, "Time"]-self.df.loc[1, "Time"])
            self.average_timestep = numpy.mean(df_times)
        elif type(self.df["Time"][1]) == pandas._libs.tslibs.timestamps.Timestamp:
            for i in range(1, 10):
                df_times.append((self.df.loc[i+1, "Time"]-self.df.loc[1, "Time"]).total_seconds())
            self.average_timestep = numpy.mean(df_times)


    def make_some_graphs(self):
        self.load_persistence_data()
        self.load_experimental_data()
        for thermistor in self.keeper_data:
            for temperature in self.keeper_data[thermistor]:
                for cut, row in self.keeper_data[thermistor][temperature].iterrows():
                    self.plotting_module(thermistor, temperature, cut, row, avg_bars=True, keywords=["waves", "mm", "microwaves", "vna"])


    def return_df(self):
        self.load_experimental_data()
        return self.df


    def omniview_gui(self, user_start, user_end, thermistors, xaxis, comments=False, save_fig=False, dpi_val=150):
        #       
        #       An in-memory way of viewing data from a particular timerange
        #       
        fig_x_basic_info = (0.25/16)*fig_x_dim
        fig_y_basic_info = (8.1/9)*fig_y_dim
        fig_x_end_range_data = (13/16)*fig_x_dim
        fig_x_start_range_data = (1.75/16)*fig_x_dim
        fig_y_range_data = (4.3/9)*fig_y_dim
        fig_x_logbook_comment = (0.25/16)*fig_x_dim
        fig_y_logbook_comment = (4.25/9)*fig_y_dim
        fig_x_timestamp = (0.25/16)*fig_x_dim
        fig_y_anchor_timestamp = (4.1/9)*fig_y_dim
        fig_y_step_timestamp = (.15/18)*fig_y_dim
        fig_x_comment_start = (1.2/16)*fig_x_dim

        try:
            self.df
        except AttributeError:
            self.load_experimental_data()

        to_drop = []
        for thermistor in self.df:
            if thermistor != "Time" and thermistor not in thermistors:
                to_drop.append(thermistor)
        self.df.drop(labels=to_drop, axis=1, inplace=True)
        surviving_columns = self.df.columns.to_list()
        start_date = min(self.df['Time'])
        end_date = max(self.df['Time'])
        start_index = self.df[self.df['Time']==start_date].index.to_list()[0]
        end_index = self.df[self.df['Time']==end_date].index.to_list()[0]
        max_datapoints = 3000

        if user_start < user_end:
            fig = plt.figure(figsize=(fig_x_dim,fig_y_dim), dpi=dpi_val)
            if comments:
                print("Comments have been turned on")
                canvas = fig.add_subplot(111)
                graph = fig.add_subplot(211)
                footnotes = fig.add_subplot(212)
                footnotes.axis('off')
                canvas.axis('off')
            else:
                graph = fig.add_subplot(111)
            

            # __nearest(self, test_val, iterable)
            print("Locating nearest raw data-frame start index from user provided time")
            data_start_index = self.df[self.df['Time'] == self.__nearest(user_start, self.df['Time'])].index.to_list()[0]
            print("Start index located.", data_start_index)
            print("Locating nearest raw data-frame end index from user provided time")
            data_end_index = self.df[self.df['Time'] == self.__nearest(user_end, self.df['Time'])].index.to_list()[0]
            print("End index located.", data_end_index)
            delta = data_end_index-data_start_index
            index_modulus = (delta*(len(surviving_columns)-1))/max_datapoints

            
            if index_modulus <= 1:
                df_yslice = self.df.iloc[data_start_index:data_end_index:1]
            else:
                df_yslice = self.df.iloc[data_start_index:data_end_index:round(index_modulus)]
            ycut = df_yslice.drop("Time", axis=1, inplace=True)


            df_xslice = self.df.loc[df_yslice.index.tolist(),"Time"]#,xaxis] # GENERALIZE it.
            
            k = len(df_xslice)
            
            if comments:
                print("Querrying Logbook start")
                logbook_start = self.__nearest(user_start, self.logbook_df["Time"])
                print("Querrying Logbook end")
                logbook_end = self.__nearest(user_end, self.logbook_df["Time"])
                logbook_start_index = self.logbook_df[self.logbook_df["Time"] == logbook_start].index[0]
                logbook_end_index = self.logbook_df[self.logbook_df["Time"] == logbook_end].index[0]
                logbook_slice = self.logbook_df[logbook_start_index:logbook_end_index]
                print("Commenting Graph.")
                canvas, graph = self.__commenter(canvas, graph, logbook_slice, 
                                             df_xslice, rng_ss=data_start_index, 
                                             rng_ee=data_end_index, avg=90, dpi_val=dpi_val
                                             )


            print("Sliced", k, "datapoints from", delta, "Total datapoints", "between", user_start, "and", user_end)
            graph.title.set_text("Data between "+user_start.strftime("%m/%d/%Y, %H:%M:%S")+" and "+user_end.strftime("%m/%d/%Y, %H:%M:%S"))
            graph.set_xlabel("Time")
            
            for column in df_yslice:
                graph.plot(df_xslice,df_yslice[column], label=column)

            graph.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y/%m/%d %H:%M'))
            graph.xaxis_date()
            graph.legend(loc='best')


            if save_fig == True:
                if comments:
                    plt.savefig(user_start.strftime("%m_%d_%Y_%H_%M_%S")+"_to_"+user_end.strftime("%m_%d_%Y_%H_%M_%S_")+"wc")
                else:
                    plt.savefig(user_start.strftime("%m_%d_%Y_%H_%M_%S")+"_to_"+user_end.strftime("%m_%d_%Y_%H_%M_%S_"))
            else:
                plt.show()
        else:
            print("Bad Date selection.")

        plt.close('all')
        plt.clf()
        gc.collect()


    def omniview_in_terminal(self,thermistors=[]):

        #       DEPRICATED, but still useful if devloping.
        #       Terminal in-memory way of viewing data from a particular timerange
        #
        try:
            self.df
        except AttributeError:
            self.load_experimental_data()
        if len(thermistors) == 0:
            thermistors = self.df.columns.to_list()
        to_drop = []
        for thermistor in self.df:
            if thermistor != "Time" and thermistor not in thermistors:
                to_drop.append(thermistor)
        self.df.drop(labels=to_drop, axis=1, inplace=True)

        surviving_columns = self.df.columns.to_list()

        start_date = min(self.df['Time'])
        end_date = max(self.df['Time'])
        start_index = self.df[self.df['Time']==start_date].index.to_list()[0]
        end_index = self.df[self.df['Time']==end_date].index.to_list()[0]
        max_datapoints = 3000
        
        while True:
            print("Start Date:",start_date, "\n","End Date:", end_date)
            print("Select A start Date in the form: \"yyyy-mm-dd HH:MM:SS\", which is zero-padded day/month in 24-hour time")
            user_start = datetime.datetime.strptime(input("Your Date: "), "%Y-%m-%d %H:%M:%S")
            print("Select An end Date in the form: \"yyyy-mm-dd HH:MM:SS\"")
            user_end = datetime.datetime.strptime(input("Your Date: "), "%Y-%m-%d %H:%M:%S")

            if (user_start < end_date and user_start < user_end) and (user_start >= start_date and user_end <= end_date):
                fig = plt.figure(figsize=(16,9), dpi=300)
                canvas = fig.add_subplot(111)

                # __nearest(self, test_val, iterable)
                print("Locating nearest raw data-frame start index from user provided time")
                data_start_index = self.df[self.df['Time'] == self.__nearest(user_start, self.df['Time'])].index.to_list()[0]
                print("Start index located.", data_start_index)
                print("Locating nearest raw data-frame end index from user provided time")
                data_end_index = self.df[self.df['Time'] == self.__nearest(user_end, self.df['Time'])].index.to_list()[0]
                print("End index located.", data_end_index)
                delta = data_end_index-data_start_index
                index_modulus = (delta*(len(surviving_columns)-1))/max_datapoints
                if index_modulus <= 1:
                    data_slice = self.df.iloc[data_start_index:data_end_index:1]
                else:
                    data_slice = self.df.iloc[data_start_index:data_end_index:round(index_modulus)]
                print(data_slice)
                k = len(data_slice["Time"])
                time_slice = data_slice["Time"]
                print(time_slice)
                data_slice.drop("Time", axis=1, inplace=True)
                print("Sliced", k, "datapoints from", delta, "Total datapoints", "between", user_start, "and", user_end)
                
                plt.title("Data between "+user_start.strftime("%m/%d/%Y, %H:%M:%S")+" and "+user_end.strftime("%m/%d/%Y, %H:%M:%S"))
                canvas.set_xlabel("Time")
                canvas.set_ylabel("Resistance/Temperature")
                for column in data_slice:
                    canvas.plot(time_slice,data_slice[column], label=column)
                canvas.legend(loc='best')
                
                plt.savefig(user_start.strftime("%m_%d_%Y_%H_%M_%S")+"_to_"+user_end.strftime("%m_%d_%Y_%H_%M_%S_"))
                

                plt.close('all')
                plt.clf()
                gc.collect()
            else:
                print('Bad input')
                return False


    def plot_top_n_ranges(self, n=10, comments=True):
        self.find_top_n_ranges(n=n)
        for thermistor in self.n_best:
            for temperature in self.n_best[thermistor]:
                for cut, row in self.n_best[thermistor][temperature].iterrows():
                    self.plotting_module(
                                        thermistor, temperature, cut, row,
                                        avg_bars=True)     

    
    def __plot_keyword_hits(self, keywords, thermistors=None, persistance=True, kelvin=False):
        try:
            self.keyword_hits
        except AttributeError:
            try:
                print("Attempting to load keyword graph kernels from previous class instance.")
                with open("keyword_persistence.pk", 'rb') as f:
                    self.keyword_hits = pickle.load(f)
                print("Previous keyword kernels found. Graphing will begin momentarily.")
                if persistance != True:
                    print("Persistance flag not True. Forcing program to search for keywords.")
            except:
                print("No keyword kernels found. Searching for keyword hits")
                self.find_keyword_hits(keywords, thermistors=thermistors)
                with open("keyword_persistence.pk", 'wb') as f:
                    pickle.dump(self.keyword_hits, f)
        
        for thermistor in self.keyword_hits:
            for temperature in self.keyword_hits[thermistor]:
                for cut, row in self.keyword_hits[thermistor][temperature].iterrows():
                    self.plotting_module(thermistor, temperature, cut, row, keywords=keywords, wing_width=1000, avg_bars=True, kelvin=kelvin)

            
    def plotting_module(self, thermistor, temperature, cut, kernel, avg_bars=None, keywords=[], dpi_val=150, wing_width=1000, kelvin=False, extra=''):
        #################################################################################
        """
           This takes some basic information in the form of its arguments, and with a 
           kernel in the form of:
                        [Average, Standard Deviation, Range Start, Range End] 
            And generates a graph with that information. Arguments: dpi_val and
            wing_width effect that of the graph. Everything should take care of itself,

                                Please proceed formally.

            Valid combinations of optional arguments are:
                - avg_bars = True, keywords=None, comments=True
                - avg_bars = None, keywords=[List of keywords], comments = True
                                                                                      """
        #################################################################################
        #fig_x_dim = 8
        #fig_y_dim = 4.5
        fig_x_basic_info = (0.25/16)*fig_x_dim
        fig_y_basic_info = (8.1/9)*fig_y_dim
        fig_x_end_range_data = (13/16)*fig_x_dim
        fig_x_start_range_data = (1.75/16)*fig_x_dim
        fig_y_range_data = (4.3/9)*fig_y_dim
        fig_x_logbook_comment = (0.25/16)*fig_x_dim
        fig_y_logbook_comment = (4.25/9)*fig_y_dim
        fig_x_timestamp = (0.25/16)*fig_x_dim
        fig_y_anchor_timestamp = (4.1/9)*fig_y_dim
        fig_y_step_timestamp = (.15/18)*fig_y_dim
        fig_x_comment_start = (1.2/16)*fig_x_dim

        if kernel[1] > 0:
            fig = plt.figure(figsize=(fig_x_dim,fig_y_dim), dpi=dpi_val)
            canvas = fig.add_subplot(111)
            std = kernel[0]
            avg = kernel[1]
            rng = [kernel[2],kernel[3]]
            nth_range = cut
            (rng_start, rng_end) = (rng[0], rng[1])
            (rng_ss,rng_ee) = (rng_start, rng_end)
            d_points = rng[1]-rng[0]

            while rng_start > 0: # Provides wings about to the region to the left
                if abs(rng_ss - rng_start) <= wing_width and rng_start > 0:
                    rng_start -= 1
                else:
                    break
            while rng_end > 0: # Provides wings about the region to the right.
                if abs(rng_ee - rng_end) <= wing_width and rng_end < len(self.df["Time"]):
                    rng_end += 1
                else:
                    rng_end -= 1
                    break
            (df_xslice, df_yslice) = (self.df.loc[rng_start:rng_end, "Time"], self.df.loc[rng_start:rng_end,thermistor])
            if kelvin:
                try:
                    self.coefficents_df
                except AttributeError:
                    self.load_coefficents()
                df_yslice = self.convert_df_yslice(thermistor, df_yslice)

            if kernel[1] == 1: # So we get meaningful results.
                avg = numpy.mean(df_yslice)
                std = numpy.std(df_yslice)
                rng_ee += 1

            ### Annotations ###

            # Upper figure Annotations: basic information
            canvas.annotate(
                "Average: "+str(avg)+"\n"+"Standard Deviation: "+str(std)+\
                '\nRange: '+str(rng)+'\nRange length: '+str(d_points)+\
                '\nColumn Length: '+str(len(self.df[thermistor])),
                xy=(fig_x_basic_info*dpi_val,fig_y_basic_info*dpi_val), 
                xycoords='figure pixels')

            # Far left graph x-axis: Date of the FIRST datapoint.
            canvas.annotate(
                self.df["Time"][rng_start],
                xy=(fig_x_start_range_data*dpi_val,fig_y_range_data*dpi_val),
                xycoords='figure pixels')

            # Far right graph x-axis: Date of LAST datapoint.
            canvas.annotate(
                self.df["Time"][rng_end],
                xy=(fig_x_end_range_data*dpi_val,fig_y_range_data*dpi_val),
                xycoords='figure pixels')

        
            try:
                # Load logbook if module has been called prematurely.
                self.logbook_df
            except AttributeError:
                try:
                    self.__load_logbook()
                except FileNotFoundError:
                    try:
                        self.logbook_df = self.df['Time', "Comment"]
                    except AttributeError:
                        print("No comments have been provided in either the logbook_data.csv, or datafile.")
                        exit()

            canvas.annotate(
                "Logbook comments:",
                xy=(fig_x_logbook_comment*dpi_val,fig_y_logbook_comment*dpi_val),
                xycoords='figure pixels')

            # Generator of figures components.
            (range_start, range_end) = (min(df_xslice), max(df_xslice))
            graph = fig.add_subplot(211)
            footnotes = fig.add_subplot(212)
            footnotes.axis('off')
            canvas.axis('off')
            
            ycut = self.df.loc[rng_start:rng_end, thermistor]
            if kelvin:
                ycut = self.convert_df_yslice(thermistor, ycut)
            if len(ycut) > 1:
                ### All of the Data ###
                graph.plot(self.df.loc[rng_start:rng_end, "Time"], ycut, color="blue", label="Data")
            else:
                return False

            if avg_bars is not None:
                ### Average Dashed Line ###
                graph.plot(
                    (df_xslice[rng_ss],df_xslice[rng_ee-1]),
                    (avg,avg),'g', dashes=[30, 30], label="Average Value of selected Range")
                
                ### Red Lines ###
                graph.plot(
                    (df_xslice[rng_ss],df_xslice[rng_ss]),
                    (avg-max(ycut)*0.05,avg+max(ycut)*0.05),
                    'r')
                graph.annotate(
                    str(df_xslice[rng_ss]),
                    xy=(df_xslice[rng_ss], avg+max(ycut)*0.053),
                    xycoords='data', color='red') # The range-of-interest start time
                graph.plot(
                    (df_xslice[rng_ee-1],df_xslice[rng_ee-1]),
                    (avg-max(ycut)*0.05,avg+max(ycut)*0.05),
                    'r')
                graph.annotate(
                    str(df_xslice[rng_ee-1]),
                    xy=(df_xslice[rng_ee-1], avg+max(ycut)*0.053),
                    xycoords='data', color='red') # The range of interest end time
            
            logbook_start = self.__nearest(range_start, self.logbook_df["Time"])
            logbook_end = self.__nearest(range_end, self.logbook_df["Time"])
            logbook_start_index = self.logbook_df[self.logbook_df["Time"] == logbook_start].index[0]
            logbook_end_index = self.logbook_df[self.logbook_df["Time"] == logbook_end].index[0]
            logbook_slice = self.logbook_df[logbook_start_index:logbook_end_index]
            
            canvas, graph = self.__commenter(canvas, graph, logbook_slice, 
                                             df_xslice, rng_ss, keywords, 
                                             rng_ee, avg, dpi_val=dpi_val
                                             )

            graph.set_xlim(left=self.df.loc[rng_start, "Time"], right=self.df.loc[rng_end, "Time"])
            graph.set_title(thermistor+"_"+temperature+"_in_range_"+str(nth_range)+"_"+extra)
            graph.set_xlabel("Time")
            if kelvin:
                graph.set_ylabel("Kelvin")
            else:
                graph.set_ylabel("Resistance")
            graph.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y/%m/%d %H:%M'))
            graph.xaxis_date()
            graph.legend(loc='best')

            plt.savefig(thermistor+"_"+temperature+"_in_range_"+str(nth_range)+extra+".png")
            print("Generated: ", thermistor+"_"+temperature+"_in_range_"+str(nth_range)+extra+".png")
            plt.close('all')
            plt.clf()
            gc.collect() # You will run out of memory if you do not do this.
            return True


    def __commenter(self, canvas, graph, logbook_slice, df_xslice, rng_ss=0, keywords=[], rng_ee=0, avg=0, dpi_val=300):
        # THERE IS CURRENTLY A BUG WHERE IF COMMENTS ARE SET TO TRUE, IN THE GUI DATE-GRAPHER THIS THING WILL DROP TIMESTAMPS
        # ON THE COMMENTS OF THE FIGURES. I HAVE YET TO FIND OUT WHAT IS CAUSING THAT, BUT FOR NOW THE PROGRAM WORKS. 
        #
        #fig_x_dim = 32
        #fig_y_dim = 18
        fig_x_basic_info = (0.25/16)*fig_x_dim
        fig_y_basic_info = (8.1/9)*fig_y_dim
        fig_x_end_range_data = (13/16)*fig_x_dim
        fig_x_start_range_data = (1.75/16)*fig_x_dim
        fig_y_range_data = (4.3/9)*fig_y_dim
        fig_x_logbook_comment = (0.25/16)*fig_x_dim
        fig_y_logbook_comment = (4.25/9)*fig_y_dim
        fig_x_timestamp = (0.25/16)*fig_x_dim
        fig_y_anchor_timestamp = (4.1/9)*fig_y_dim
        fig_y_step_timestamp = (.15/18)*fig_y_dim
        fig_x_comment_start = (1.2/16)*fig_x_dim
        avg_comments = []
        poi = True
        v = 0
        n = 0
        was = False
        shift = False
        shift_2 = False
        trip = True
        for index, row in logbook_slice.iterrows():
            modified_comment, y = self.graph_comment_formater(row["Comment"])
            v += y
            timestamp = row["Time"]
            have_i_printed = False
            old_v = v
            old_n = n
            if v > 54:
                if not shift:
                    fig_x_comment_start += 10.7
                    fig_x_timestamp += 10.7
                    shift = True
                elif not shift_2 and v > 108:
                    fig_x_comment_start += 10.7
                    fig_x_timestamp += 10.7
                    shift_2 = True
                
                if v > 108:
                    v -= 108
                    n -= 108
                else:
                    v -= 55
                    n -= 55
            try:
                if df_xslice[rng_ss] <= timestamp and timestamp <= df_xslice[rng_ee-1]:
                    canvas.annotate(
                        timestamp, 
                        xy=(fig_x_timestamp*dpi_val,(fig_y_anchor_timestamp-fig_y_step_timestamp*n)*dpi_val), 
                        xycoords='figure pixels', color="green") 
                    avg_comments.append(n)
                else:
                    canvas.annotate(
                        timestamp, 
                        xy=(fig_x_timestamp*dpi_val,(fig_y_anchor_timestamp-fig_y_step_timestamp*n)*dpi_val), 
                        xycoords='figure pixels')
            except KeyError:
                if trip:
                    print("WARNING: Slicing miss-match. IGNORE if using Date Grapher with comments.")
                    trip = False
            n = old_n
            n += 1
            n += y
            if any(x in str(row["Comment"]) for x in keywords):
                if min(df_xslice) <= row["Time"] and row["Time"] <= max(df_xslice): 
                    canvas.annotate(
                        modified_comment, 
                        xy=(fig_x_comment_start*dpi_val,(fig_y_anchor_timestamp-fig_y_step_timestamp*v)*dpi_val),
                        xycoords='figure pixels', color='goldenrod')
                    x_loc = int(self.logbook_df[self.logbook_df["Comment"] == row["Comment"]].index[0])
                    logbook_hit_date = self.logbook_df.loc[x_loc, "Time"]
                    graph.plot(
                        logbook_hit_date,
                        avg, 'ro',
                        color="goldenrod", ms=10, label=("Keyword Hit") if poi else None)
                    poi = False
                    have_i_printed = True
            if v in avg_comments:
                for index in avg_comments:
                    if v == index:
                        canvas.annotate(
                            modified_comment, 
                            xy=(fig_x_comment_start*dpi_val,(fig_y_anchor_timestamp-fig_y_step_timestamp*v)*dpi_val),
                            xycoords='figure pixels', color='green')
                    have_i_printed = True
            elif not have_i_printed:
                canvas.annotate(
                    modified_comment, 
                    xy=(fig_x_comment_start*dpi_val,(fig_y_anchor_timestamp-fig_y_step_timestamp*v)*dpi_val),
                    xycoords='figure pixels')
                have_i_printed = True
            v = old_v
            v += 1
            if shift_2 and v > 200:
                print("WARNING: Out of lab-book comment space on figure. Consider selecting a narrower plotting range if labbook comment insight is critical.")
                return canvas,graph

        return canvas, graph


    def convert_df_yslice(self, thermistor, data):
        thermistors_in_file = ["CCS.F1","CCS.F2","CCS.F3","CCS.F4","AB.F5", "AB.F6", "CCS.F7", "CCS.F8","CCS.F9","CCS.F10", "CCS.F11", "CCS.S1", "CCS.S2", "CCS.S3"]
        if thermistor in thermistors_in_file:
            (a,b,c) = (self.coefficents_df.loc['a', thermistor], self.coefficents_df.loc['b', thermistor], self.coefficents_df.loc['c', thermistor])
            data2 = data.apply(convert_to_k, args=(a,b,c))
        else:
            data2 = data.apply(convert_to_k_spect, args=(), thermistor=thermistor)
        return data2


    def graph_comment_formater(self, comment):
        # MAX COLUMN LENGTH 55
        ls = list(str(comment))
        n = 0
        testval=110
        for element in range(0, len(ls)):
            if element % testval == 0 and element != 0:
                if re.search('[a-zA-Z]', ls[element]):
                    ls.insert(element-1, '-')
                ls.insert(element, '\n')
                n += 1
        str_to_return = "".join(ls)
        return str_to_return, n

                
    def return_dfs(self):
        self.load_persistence_data()
        self.load_experimental_data()
        self.get_thermometry_data()
        thermistors = {}
        for name in self.df.columns.values:
            if name != "Time":
                thermistors[name] = tp(name, self.kd_name, self.keeper_data[name])
        for key in thermistors:
            if key != "Time":
                thermistors[key].calibrate_curve()


    def load_coefficents(self):
        with open("curve_coefficent_data.csv", 'r') as f:
            self.coefficents_df = pandas.read_csv(f, index_col='Name')


    def __load_persistence_data_record(self):
        with open(self.data_record_location, 'r') as f:
            self.data_record = pandas.read_csv(f, delimiter='\t')
        self.data_record["Time"] = pandas.to_datetime(self.data_record["Time"])

    
    def nearest_spike(self, test_val, iterable, updown, tag):
        # Based on the __nearest() method, this does that, 
        # but returns the critical range information for our kernels.
        print("Looking for the nearest date to", test_val, "from data record", tag, "in raw-data file")
        nearest_time = min(iterable, key=lambda x: abs(x - test_val))
        df_index = self.df.index[self.df["Time"] == nearest_time][0]
        return [tag, nearest_time, df_index, updown]


    def plot_magnet_spikes(self, thermistors=None, keywords=[], kelvin=False):
        self.magnet_spikes = {}
        self.__load_persistence_data_record()
        self.load_experimental_data()
        self.get_thermometry_data()
        prv_state = 0
        self.mag_spike_indexes = []
        for index, row in self.data_record.iterrows():
            time = row["Time"]
            state = row["State"]
            if prv_state == 0 and state == 1:
                self.mag_spike_indexes.append((index, "up"))
            elif prv_state == 1 and state == 0:
                self.mag_spike_indexes.append((index, "down"))
            prv_state = state
        with Pool(processes=self.processes) as pool:
            result_objects = [pool.apply_async(
                              self.nearest_spike, 
                              args=(self.data_record.loc[dr_index[0], "Time"],
                              self.df["Time"], dr_index[1], dr_index[0])) for dr_index in self.mag_spike_indexes]
            pool.close()
            pool.join()
        results = [r.get() for r in result_objects if r.get() != False]

        if thermistors is not None:
            self.thermistor_names = thermistors

        for thermistor in self.thermistor_names: # Creating Kernels here.
            kernel_dicts = {} # [std, avg, range_begin, range_end]
            for result in results: # [logbook_index, nearest_df_time, data_file_index]
                kernel_dicts[result[0]] = [1, 1, result[2], result[2], "_"+result[3]]
            self.magnet_spikes[thermistor] = {"MAGNET_SPIKE":pandas.DataFrame.from_dict(kernel_dicts, orient='index', columns=["STD", "AVG", "RANGE START", "RANGE END", "UPDOWN"])}
        
        for thermistor in self.magnet_spikes:
            for cut, row in self.magnet_spikes[thermistor]["MAGNET_SPIKE"].iterrows():
                self.plotting_module(thermistor, "MAGNET_SPIKE", cut, row, keywords=keywords, wing_width=1000, avg_bars=True, kelvin=kelvin, extra=self.magnet_spikes[thermistor]["MAGNET_SPIKE"].loc[cut,"UPDOWN"])
                
        
def convert_to_k(r,a,b,c):
    #
    return a+b*numpy.exp(c*(1000/r))

def convert_to_k_spect(r,**kwargs):
    t = kwargs["thermistor"]
    coeffs = {
              "CCX.T1": [1.09853, -1.262496, 0.610678, -0.26231, 0.103527, -0.0381089, 0.013162, -0.004359, 0.001512],
              "CX.T2":[1.09853, -1.262496, 0.610678, -0.26231, 0.103527, -0.0381089, 0.013162, -0.004359, 0.001512],
              "CCCS.T3":[-0.1562396321606, 27.64546747296, -188.2549809283, 1044.765194077, -2679.688300274, 3485.87992613, -1215.52472759]
             }
    val = 0
    n=0
    if t == "CCCS.T3":
        for coeff in coeffs[t]:
            try:
                val += coeff*(1000/r)**n
            except:
                print("Divide by zero error")
                return 0
            n += 1
        return val
    else:
        ZU = 4.57773645241
        ZL = 2.79190447712
        Z = numpy.log(r)
        k = ((Z-ZL)-(ZU-Z))/(ZU-ZL)
        n = 0
        val = coeffs[t][n]
        n += 1
        for coeff in coeffs[t]:
            val += coeff*numpy.cos(n*numpy.arccos(k))
        val = coeffs[t][0] 
        return val

def what_temperature_range_are_we_in(average,column_name):
        if "C" in column_name:
            if average<1200:
                return 0
            elif average<2800:
                return 1
            elif average>2800:
                return 2 
        elif "A" in column_name:
            if average<155:
                return 0
            elif average<800:
                return 1
            elif average>800:
                return 2
