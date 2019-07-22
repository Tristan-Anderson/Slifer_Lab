import time, pandas, numpy, copy, datetime, os, traceback, dateutil.parser, multiprocessing
from multiprocessing import Pool
from pandas.plotting import register_matplotlib_converters
from thermistor_profile import thermistor_profile as tp
import _pickle as pickle
import matplotlib
from matplotlib import pyplot as plt
import gc

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

class slifercal(object):
    def __init__(self, processes=0, datafile_location=None, logbook_datafile_location=None, record_location=None):
        register_matplotlib_converters() # Calling the calibrate method without this here told me to put this here.
        self.trslat = {0: "RT", 1: "LN2", 2: "LHe"}
        self.datafile_location = datafile_location
        self.logbook_datafile_location= logbook_datafile_location
        self.data_record_datafile_location = record_location
        (filename,line_number,function_name,text)=traceback.extract_stack()[-2] # NAME of THIS CLASS (for pickling later if needed - but probably not.)
        def_name = text[:text.find('=')].strip()
        self.name = def_name
        if processes == 0:
            self.processes = int(8*multiprocessing.cpu_count()/10)
        else:
            self.processes = processes

    def __cleandf(self):
        ################################################
        """
           Basic data cleaning geared for thermometry
                         lab data
                                                     """
        ################################################
        indexes_that_are_to_be_deleted = []
        rows_to_be_deleted = []
        for title in self.df:
            indexes_that_are_to_be_deleted.append(self.df[(self.df[title]=="#VALUE!")].index.tolist())
            indexes_that_are_to_be_deleted.append(self.df[(self.df[title]=="Err:502")].index.tolist())
        for l in indexes_that_are_to_be_deleted:
            for entry in l:
                if entry not in rows_to_be_deleted:
                    rows_to_be_deleted.append(entry)
        self.df = self.df.drop(rows_to_be_deleted, axis=0)
        print("File Cleaned.")
        for index in self.df:
            if index != "Time":
                self.df[index] = self.df[index].astype(float)   
        column_names = list(self.df)
        column =[]
        for name in column_names:
            if '.R' in name and ('.M' in name or '.F' in name): # Feature that allows user to throw in raw .csv of entire cooldown
                    column.append(name)                         # Probably no longer necessary for Slifer Lab, but may be useful to
            for name in column_names:                           # People who dont read documentation    
                if name != 'Time' and name not in column:
                    self.df.drop(columns=name)

    def complete_keyword(self, timeit, keywords, rangeshift=1, range_length=None):
        if timeit:
            readings = time.time()
        self.__read_data()
        if timeit:
            readingf=time.time()
            cleans = time.time()
        self.__cleandf()
        if timeit:
            cleanf=time.time()
            analysiss = time.time()
        self.__range_election(rangeshift=rangeshift, range_length=range_length)
        if timeit:
            analysisf = time.time()
            plottings = time.time()
        self.plot_keyword_hits(keywords, thermistors=None)
        if timeit:
            plottingf=time.time()
        print("Reading", readingf-readings, "Analysis", analysisf-cleans, "Plotting", plottingf-plottings)

    def keyword(self, keywords, thermistors=None):
        self.__read_data()
        self.__cleandf()
        self.load_data()
        self.plot_keyword_hits(keywords, thermistors=thermistors)
    
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
        self.__read_data()
        self.__cleandf()
        self.__range_election(rangeshift=rangeshift)

    def load_data(self, file_location=None):
        ###################################################
        """
           This can be used in many modes, but it really
           was made with the intent of persistance. The
             user would load their pickled keeper_data 
             in a live version of python3 and call the
           plot_calibration_candidates method to make a
           bunch of graphs based on the data loaded into
                   the instance from this method.
                                                        """
        ###################################################

        if file_location == None:
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
                print("No parsed datafile. Exiting.")
                exit()
            print("File Read")
        elif file_location != None:
            self.kd_name = file_location
            print("Reading file")
            with open(file_location, 'rb') as fin:
                self.keeper_data = pickle.load(fin)
            print("File read")

    def __load_data_record(self):
        if self.data_record_datafile_location == None:
            print("No loogbook was used to initalize the instance!\nAssuming filename is \"data_record.csv\"")
            self.data_record_datafile_location = "data_record.csv"
        with open(self.data_record_datafile_location,'r') as f:
            self.record_df = pandas.read_csv(f, sep='\t')
        self.record_df["Time"] = pandas.to_datetime(self.record_df['Time'])
        print("Data Record loaded.")

    def __load_logbook(self):
        if self.logbook_datafile_location == None:
            print("No loogbook path was used to initalize the instance!\nAssuming logbook is \"logbook_data.csv\" \nSearching local directory:")
            self.logbook_datafile_location = "logbook_data.csv"
        with open(self.logbook_datafile_location,'r') as f:
            self.logbook_df = pandas.read_csv(f, sep='\t')
        self.logbook_df["Time"] = pandas.to_datetime(self.logbook_df["Time"]) 
        print("File found. Comments File loaded.")

    def __nearest(self, test_val, iterable): 
        # In an interable data-structure, find the nearest to the 
        # value presented.
        return min(iterable, key=lambda x: abs(x - test_val))

    def keyword_nearest(self, test_val, iterable, tag):
        # Based on the __nearest() method, this does that, 
        # but returns the critical range information for our kernels.
        print("Looking for the nearest date to", test_val, "from logbook index", tag, "in raw-data file")
        nearest_time = min(iterable, key=lambda x: abs(x - test_val))
        df_index = self.df.index[self.df["Time"] == nearest_time][0]
        return [tag, nearest_time, df_index]

    def __range_election(self, rangeshift=1, range_length=None):
        #############################################################
        """
                            This method slices
           data in self.df, skips the slice if there are any zero 
           vals the the slice. Then takes the files average, std,
           and assigns it a "Temperature range" based on ballpark
                       estimates found in the function:
                      what_temperature_range_are_we_in()
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

    def __read_data(self):
        # Reads in the raw data, and find comments that goes with it.
        self.thermistor_names = []
        if self.datafile_location == None:
            print(
                "No datafile was used to initalize the instance!\
                \nAssuming filename is \"data.csv\"")
            self.datafile_location = "data.csv"
        with open(self.datafile_location,'r') as f:
            self.df = pandas.read_csv(f)
        print("File loaded.")
        if type(self.df["Time"][1]) == numpy.float64: # If david did not convert time from 1904/12/31 20:00:00, then do the conversion and put it into datetime.
            self.df["Time"] = self.df["Time"].apply(self.__time_since_1904)
        try:
            self.logbook_df = self.df['Time', "Comment"]
        except KeyError as e:
            try:
                self.df["Comment"]
            except KeyError as c:
                print("No comments were provided in the datafile. Searching elsewhere...")
                self.__load_logbook()
        for column in self.df:
            if column not in ["Time", "Comment"]:
                self.thermistor_names.append(column)

    def find_top_n_ranges(self, n=10):
        ###############################################################
        """
           This is the self.keeper_data parser that sorts through 
           self.keeper_data and creates a new dictionary containing 
                     the n most stable regions of data.
                                                                    """
        ###############################################################
        print("Searching for n-best...")
        self.load_data()
        self.__read_data()
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
                                                                   """
        ##############################################################
        
        self.__read_data()
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
        time.sleep(1)

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

    def __time_since_1904(self,sec): 
        # LabVIEW conveniently used seconds from "1 January, 1904" as time-stamp.
        self.begining_of_time = datetime.datetime(1903, 12, 31)+datetime.timedelta(seconds=72000) # I saw a -4 hour time difference.
        return self.begining_of_time + datetime.timedelta(seconds=sec) # This returns a "Ballpark" time. Its probably not accruate to the second, but it is definately accurate to the hour.
    
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
        if type(self.df["Time"][1]) == str:
            for i in range(1,10):
                ent = dateutil.parser.parse(self.df["Time"][i+1])-dateutil.parser.parse(self.df["Time"][i])
                df_times.append(ent.total_seconds())
            self.average_timestep = numpy.mean(df_times)
        elif type(self.df["Time"][1]) == numpy.float64:
            for i in range(1, 10):
                df_times.append(self.df["Time"][i+1]-self.df["Time"][i])
            self.average_timestep = numpy.mean(df_times)
        elif type(self.df["Time"][1]) == pandas._libs.tslibs.timestamps.Timestamp:
            for i in range(1, 10):
                df_times.append((self.df["Time"][i+1]-self.df["Time"][i]).total_seconds())
            self.average_timestep = numpy.mean(df_times)

    def make_some_graphs(self):
        self.load_data()
        self.__read_data()
        for thermistor in self.keeper_data:
            for temperature in self.keeper_data[thermistor]:
                for cut, row in self.keeper_data[thermistor][temperature].iterrows():
                    self.plotting_module(thermistor, temperature, cut, row, keywords=["waves", "mm", "microwaves", "vna"])
    
    def plot_top_n_ranges(self, n=10, comments=True):
        self.find_top_n_ranges(n=n)
        for thermistor in self.n_best:
            for temperature in self.n_best[thermistor]:
                for cut, row in self.n_best[thermistor][temperature].iterrows():
                    self.plotting_module(
                                        thermistor, temperature, cut, row,
                                        avg_bars=True, comments=True)     
    
    def plot_keyword_hits(self, keywords, thermistors=None):
        self.find_keyword_hits(keywords, thermistors=thermistors)
        for thermistor in self.keyword_hits:
            for temperature in self.keyword_hits[thermistor]:
                for cut, row in self.keyword_hits[thermistor][temperature].iterrows():
                    self.plotting_module(thermistor, temperature, cut, row, keywords=keywords, comments=True, wing_width=1000, avg_bars=True)
            

    def plotting_module(self, thermistor, temperature, cut, kernel, avg_bars=None, keywords=None, comments=None, dpi_val=150, wing_width=1000):
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
        fig_x_dim = 32
        fig_y_dim = 18
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
                    break
            (df_xslice, df_yslice) = (self.df.loc[rng_start:rng_end, "Time"], self.df.loc[rng_start:rng_end,thermistor])
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

            if comments is not None:
                try:
                	# Load logbook if module has been called prematurely.
                    self.logbook_df
                except:
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

                ### Title and Labels ###
                graph.set_title(thermistor+"_"+temperature+"_in_range_"+str(nth_range))
                graph.set_xlabel("Time")
                graph.set_ylabel("Resistance")
                graph.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y/%m/%d %H:%M'))

                
                ycut = self.df.loc[rng_start:rng_end, thermistor]

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
                        (avg-max(self.df.loc[rng_start:rng_end, thermistor])*0.05,avg+max(self.df.loc[rng_start:rng_end, thermistor])*0.05),
                        'r')
                    graph.annotate(
                        str(df_xslice[rng_ss]),
                        xy=(df_xslice[rng_ss], avg+max(self.df.loc[rng_start:rng_end, thermistor])*0.053),
                        xycoords='data', color='red') # The range-of-interest start time
                    graph.plot(
                        (df_xslice[rng_ee-1],df_xslice[rng_ee-1]),
                        (avg-max(self.df.loc[rng_start:rng_end, thermistor])*0.05,avg+max(self.df.loc[rng_start:rng_end, thermistor])*0.05),
                        'r')
                    graph.annotate(
                        str(df_xslice[rng_ee-1]),
                        xy=(df_xslice[rng_ee-1], avg+max(self.df.loc[rng_start:rng_end, thermistor])*0.053),
                        xycoords='data', color='red') # The range of interest end time
                
                logbook_start = self.__nearest(range_start, self.logbook_df["Time"])
                logbook_end = self.__nearest(range_end, self.logbook_df["Time"])
                logbook_start_index = self.logbook_df[self.logbook_df["Time"] == logbook_start].index[0]
                logbook_end_index = self.logbook_df[self.logbook_df["Time"] == logbook_end].index[0]
                logbook_slice = self.logbook_df[logbook_start_index:logbook_end_index]
                
                avg_comments = []
                poi = True
                v = 0
                n = 0
                for index, row in logbook_slice.iterrows():
                	row["comment"], y = self.graph_comment_formater(row["Comment"])
                	
                	v += y
                    timestamp = row["Time"]
                    have_i_printed = False
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
                    n += 1
                    if any(x in str(row["Comment"]) for x in keywords):
                        if min(df_xslice) <= row["Time"] and row["Time"] <= max(df_xslice): 
                            canvas.annotate(
                                row["Comment"], 
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
                    if v in avg_comments and not have_i_printed:
                        for index in avg_comments:
                            if v == index:
                                canvas.annotate(
                                    row["Comment"], 
                                    xy=(fig_x_comment_start*dpi_val,(fig_y_anchor_timestamp-fig_y_step_timestamp*v)*dpi_val),
                                    xycoords='figure pixels', color='green')
                            have_i_printed = True
                    elif not have_i_printed:
                        canvas.annotate(
                            row["Comment"], 
                            xy=(fig_x_comment_start*dpi_val,(fig_y_anchor_timestamp-fig_y_step_timestamp*v)*dpi_val),
                            xycoords='figure pixels')
                        have_i_printed = True
                    v += 1

            graph.set_xlim(left=self.df.loc[rng_start, "Time"], right=self.df.loc[rng_end, "Time"])
            graph.set_title(thermistor+"_"+temperature+"_in_range_"+str(nth_range))
            graph.set_xlabel("Time")
            graph.set_ylabel("Resistance")
            graph.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y/%m/%d %H:%M'))
            graph.xaxis_date()
            graph.legend(loc='best')

            plt.savefig(thermistor+"_"+temperature+"_in_range_"+str(nth_range)+".png")
            print("Generated: ", thermistor+"_"+temperature+"_in_range_"+str(nth_range)+".png")
            plt.close('all')
            plt.clf()
            gc.collect() # You will run out of memory if you do not do this.
            return True

	def graph_comment_formater(self, comment):
		# MAX COLUMN LENGTh 55
		ls = list(comment)
		n = 1
		for element in range(0, len(ls)):
			if element % 35 == 0 and element != 0:
				if re.search('[a-zA-Z]', ls[element]):
					ls.insert(element-1, '-')
				ls.insert(element, '\n')
				n += 1
				print("Linebreak")
		str_to_return = "".join(ls)
		return str_to_return, n
                
    def return_dfs(self):
        self.load_data()
        self.__read_data()
        self.__cleandf()
        thermistors = {}
        for name in self.df.columns.values:
            if name != "Time":
                thermistors[name] = tp(name, self.kd_name, self.keeper_data[name])
        for key in thermistors:
            if key != "Time":
                thermistors[key].calibrate_curve()
