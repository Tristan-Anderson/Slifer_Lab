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
    """
    This class is meant to be used in Python's Programming / Interactive mode on a command line.

    ON INITALIZATION:
        The objective of this class was to create a persistance thermistor calibration program using pickle as the I/O. The user can initalize the class with
            a datafile location (datafile_location) but providing a datafile location is not necessary to create an instance of this class. All the class does when 
            initalized is take whatever location provided, and store it as a self variable. Nothing more.
        
        The flexibility in the initalization allows for a user to specify or load a previously parsed version of data that was analyzied in the methods of the class
        described below:
    METHODS:
        load_previous_data(data_location=None):
            This method searches for a "keeper_data_original_YYYY_MM_DD_HHMMSS.pk" where Y, M, D, H, M, S is the year, month, day, hour, minute, and second respectively
                specifying the file's time of generation. By default it will search for the most recent "keeper_data_original" file. One does not have to provide a file
                name or location so long as the name of the file remains unchanged.
        find_calibration_points(range_shift=1):
            This will read the datafile provided during the initalization. If no file was provided, it will prompt of the location one. It will take several minutes to calculate the
                calibration points. The program should update you wehre it is in the process.
            This method works by looking at one column of data at a time, and by taking a slice of data out of the column. The program then calculates the average value within that
                slice of data, and calculates its standard deviation. This information, including the range in which the slice originated in is saved, and then is written to a
                dictionary. This dictionary of standard deviations, averages, and ranges is then further analyzed and each column has 3 range-averages selected from the dictioanry 
                (CALIBRATION POINTS) with the smallest standard deviation. It then writes the calibration points dictionary, as well as the dictioanry with averages and ranges to a
                binary pickle file.
            PARAMETERS:
                range_shift modifies how the slices of data are selected from the datafile. The default is 1, which shifts the range by 1 in the "averaging" section above

    """
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
            self.processes = int(2*multiprocessing.cpu_count()/3)
        else:
            self.processes = processes

    def cal_suite(self, rangeshift=1, n_best=10, range_length=None, dpi_val=150, logbook=True):
        self.__read_data()
        self.__cleandf()
        self.__range_election(rangeshift=rangeshift, range_length=range_length)
        self.plot_calibration_candidates(n_best=n_best, plot_logbook=logbook, data_record=True, dpi_val=dpi_val, plotwidth=1500)

    def __cleandf(self):
        #############################################
        """
           Basic data cleaning geared for lab data
                                                  """
        #############################################
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

    def complete(self, timeit, rangeshift=1, n_best=3, range_length=None, dpi_val=200, logbook=True):
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
        self.plot_calibration_candidates(n_best=n_best, plot_logbook=logbook, data_record=True, dpi_val=dpi_val, plotwidth=1500)
        if timeit:
            plottingf=time.time()
            update_cals = time.time()
        thermistors = {}
        for name in self.df.columns.values:
            if name != "Time":
                thermistors[name] = tp(name, self.kd_name, self.keeper_data[name])
        for key in thermistors:
            if key != "Time":
                thermistors[key].auto_update_calpoint()
        if timeit:
            update_calf = time.time()
            print("\n\nReading:", readingf-readings, "\nCleaning:", cleanf-cleans, "\nAnalyis:", analysisf-analysiss, "\nPlotting:",plottingf-plottings,"\nCals:",update_calf-update_cals)

    def __debug_attribute(self, obj):
        import pprint
        with open("debug.txt", 'w') as fout:
            pprint.pprint(obj, fout)
        print("Printed object to file: debug.txt")

    def find_stable_regions(self, rangeshift=1):
        self.__read_data()
        self.__cleandf()
        self.__range_election(rangeshift=rangeshift)

    def __keeper_data_cleaner(self, do_i_print=True):
        #######################################
        """
            This is a less robust version of 
                self.__save_top_n_ranges()
                                            """
        #######################################
        print("Parsing data...")
        self.thermistor_calibration_points = {}
        for thermistor in self.keeper_data:
            for temprange in self.keeper_data[thermistor]:
                min_std = 100000000000 # God help you if your ranges are this noisy.
                for row in self.keeper_data[thermistor][temprange].itertuples(name=None):
                    std = row[1]
                    if min_std > std:
                        min_std = std
                        slice_number = nth_range
                try:
                    calibration_list[temperature] = [
                        min_std, 
                        self.keeper_data[thermistor][temperature][slice_numbers][1], 
                        self.keeper_data[thermistor][temperature][slice_numbers][2]]
                except:
                    if do_i_print:
                        print("No Calibration point present for", thermistor, "in", temperature, " range.")
                        continue
                if do_i_print:        
                    print("Located flattest", temperature, 
                          "datapoint for thermistor:", thermistor)
            self.thermistor_calibration_points[thermistor] = calibration_list

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
            print("No loogbook was used to initalize the instance!\nAssuming filename is \"logbook_data.csv\"")
            self.logbook_datafile_location = "logbook_data.csv"
        with open(self.logbook_datafile_location,'r') as f:
            self.logbook_df = pandas.read_csv(f, sep='\t')
        self.logbook_df["Time"] = pandas.to_datetime(self.logbook_df["Time"]) 
        print("Data Record File loaded.")


    def __nearest(self, test_val, iterable): # In an interable data-structure, find the nearest to the value presented.
        return min(iterable, key=lambda x: abs(x - test_val))

    def __range_election(self, rangeshift=1, range_length=None):
        #############################################################
        """
           This is the 'brawns' of the program. This method slices
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
        if self.datafile_location == None:
            print(
                "No datafile was used to initalize the instance!\
                \nAssuming filename is \"data.csv\"")
            self.datafile_location = "data.csv"
        with open(self.datafile_location,'r') as f:
            self.df = pandas.read_csv(f)
        print("File loaded.")

    def __save_top_n_ranges(self, n=10):
        ########################################
        """
           This is the self.keeper_data parser 
           that sorts through self.keeper_data
           and creates a new dictionary con-
           taining the n most stable regions 
                           of data.
                                              """
        #########################################
        print("Parsing parsed data...")
        self.thermistor_calibration_points = {}
        for thermistor in self.keeper_data:
            calibration_list = {}
            for temperature in self.keeper_data[thermistor]:
                try:
                    calibration_list[temperature]= self.keeper_data[thermistor][temperature].nsmallest(n, "STD", keep="first")
                except:
                    print("No Calibration point present for", thermistor, "in", temperature, " range.")
                    continue
                print("Located flattest", temperature, "datapoint for thermistor:", thermistor)
            self.thermistor_calibration_points[thermistor] = calibration_list

    def __time_since_1904(self,sec): # David for some reason used seconds from "1 January, 1904" for some reason as timestamp.
        self.begining_of_time = datetime.datetime(1903, 12, 31)+datetime.timedelta(seconds=72000) # I saw a -4 hour time difference.
        return self.begining_of_time + datetime.timedelta(seconds=sec) # This returns a "Ballpark" time. Its probably not accruate to the second, but it is definately accurate to the hour.

    def __time_steps_suck(self):
        ###################################
        """
           This will find the average
           timestep in between the first
           10 datapoints, and use that 
           to search "Hour long slices" 
                   of the data.
                                        """
        ###################################
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
                
    def plot_calibration_candidates(self, n_best=10, dpi_val=150, plotwidth=1000, plot_logbook=False, data_record=True):
        #################################################################################
        """
           This method can be used after load_data. If load_data was not called before
               the execution of this method, this method will assume that you have
           "data.csv" in the same directory in which you are executing this file from.
           This method generates the n-best stable regions based on the metric defined
           in the class instance in which self.keeper_data was generated. No detection
                                will be implemented in the future.
                                                                                      """
        #################################################################################
        # STRUCTURE: thermistor_calibration_points = [Thermistor][Temperature][0:2] = [Minimum STD, average, [slice_start, slice_end]]
        k = []
        try:
            k = self.thermistor_calibration_points
            k = self.df
            k = self.logbook_entries
        except AttributeError:
            del k
            self.__read_data()
            self.__save_top_n_ranges(n=n_best)
        ### If you want to plot the logbook. Load the logbook. ###
        if plot_logbook:
            self.__load_logbook()
        ### If you want to plot the data_record, load the data record. ### 
        if data_record:
            self.__load_data_record()
        if type(self.df["Time"][1]) == numpy.float64: # If david did not convert time from 1904/12/31 20:00:00, then do the conversion and put it into datetime.
            self.df["Time"] = self.df["Time"].apply(self.__time_since_1904)

        ### Figure Formatting ###
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
        fig_x_comment_start = (2/16)*fig_x_dim

        ### Begin plotting ###
        for thermistor in self.thermistor_calibration_points:
            for temperature in self.thermistor_calibration_points[thermistor]:
                for packet in self.thermistor_calibration_points[thermistor][temperature].itertuples(name=None):
                    avg = packet[2]
                    if avg > 0:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                        std = packet[1]
                        rng = [packet[3],packet[4]]
                        nth_range = packet[0]
                        (rng_start, rng_end) = (rng[0], rng[1])
                        (rng_ss,rng_ee) = (rng_start, rng_end)
                        d_points = rng[1]-rng[0]
                        while rng_start > 0:
                            if abs(rng_ss - rng_start) <= plotwidth and rng_start > 0:
                                rng_start -= 1
                            else:
                                break
                        while rng_end > 0:
                            if abs(rng_ee - rng_end) <= plotwidth and rng_end < len(self.df["Time"]):
                                rng_end += 1
                            else:
                                break
                        (df_xslice,df_yslice) = (self.df["Time"][rng_start:rng_end], self.df[thermistor][rng_start:rng_end])
                        
                        if plot_logbook:
                            ### Data selection ###
                            (range_start, range_end) = (min(df_xslice), max(df_xslice))
                            mag = False
                            milimeter_waves = False

                            logbook_start = self.__nearest(range_start, self.logbook_df["Time"])
                            logbook_end = self.__nearest(range_end, self.logbook_df["Time"])
                            logbook_start_index = self.logbook_df[self.logbook_df["Time"] == logbook_start].index[0]
                            logbook_end_index = self.logbook_df[self.logbook_df["Time"] == logbook_end].index[0]
                            logbook_slice = self.logbook_df[logbook_start_index:logbook_end_index]
                            
                            datarecord_start = self.__nearest(range_start, self.record_df["Time"])
                            datarecord_end = self.__nearest(range_end, self.record_df["Time"])
                            data_record_start_index = self.record_df[self.record_df["Time"] == datarecord_start].index[0]
                            data_record_end_index = self.record_df[self.record_df["Time"] == datarecord_end].index[0]
                            data_record_slice = self.record_df[data_record_start_index:data_record_end_index]
                            data_record_slice = data_record_slice[data_record_slice["State"].isin([1])]
                            data_record_slice = data_record_slice[data_record_slice["Time"].isin(pandas.date_range(start=range_start,end=range_end))]

                            ### Figure Generation ###
                            fig = plt.figure(figsize=(fig_x_dim,fig_y_dim), dpi=dpi_val)
                            big_fig = fig.add_subplot(111) # Canvas
                            graph = fig.add_subplot(211) # Top-Subplot where data goes
                            footnotes = fig.add_subplot(212) # Bottom-Subplot where logbook comments go.
                            footnotes.axis('off')
                            big_fig.axis('off')



                            ### Determines What Entries in the Logbook needs to be colored, then colors them ###
                            v = 0
                            avg_comments = []
                            magnet_comments = []
                            for timestamp in logbook_slice["Time"]:
                                if df_xslice[rng_ss] <= timestamp and timestamp <= df_xslice[rng_ee-1]:
                                    big_fig.annotate(
                                        timestamp, 
                                        xy=(fig_x_timestamp*dpi_val,(fig_y_anchor_timestamp-fig_y_step_timestamp*v)*dpi_val), 
                                        xycoords='figure pixels', color="green") 
                                    avg_comments.append(v)
                                else:
                                    big_fig.annotate(
                                        timestamp, 
                                        xy=(fig_x_timestamp*dpi_val,(fig_y_anchor_timestamp-fig_y_step_timestamp*v)*dpi_val), 
                                        xycoords='figure pixels')
                                v += 1
                            
                            v = 0
                            poi = True
                            for comment in logbook_slice["Comment"]: 
                                have_i_printed = False
                                if "waves" in str(comment) or "mm" in str(comment) or "UCA" in str(comment):
                                    waves = True
                                    big_fig.annotate(
                                        comment, 
                                        xy=(fig_x_comment_start*dpi_val,(fig_y_anchor_timestamp-fig_y_step_timestamp*v)*dpi_val),
                                        xycoords='figure pixels', color='goldenrod')
                                    x_loc = int(self.logbook_df[self.logbook_df["Comment"] == comment].index[0])
                                    milimeter_waves = True
                                    #graph.plot(
                                        #(self.logbook_df.loc[x_loc, "Time"],self.logbook_df.loc[x_loc, "Time"]),
                                        #(avg+max(self.df[thermistor][rng_start:rng_end])*0.01, avg-max(self.df[thermistor][rng_start:rng_end])*0.01),\
                                        #color="goldenrod", label=("Where Waves were mentioned") if poi else None)
                                    graph.plot(
                                        self.logbook_df.loc[x_loc, "Time"],
                                        avg+max(self.df[thermistor][rng_start:rng_end])*0.02, 'ro',
                                        color="goldenrod", ms=10, label=("Where Waves were mentioned") if poi else None)
                                    poi = False
                                    have_i_printed = True
                                if v in avg_comments and not have_i_printed:
                                    for index in avg_comments:
                                        if v == index:
                                            big_fig.annotate(
                                                comment, 
                                                xy=(fig_x_comment_start*dpi_val,(fig_y_anchor_timestamp-fig_y_step_timestamp*v)*dpi_val),
                                                xycoords='figure pixels', color='green')
                                        have_i_printed = True
                                elif not have_i_printed:
                                    big_fig.annotate(
                                        comment, 
                                        xy=(fig_x_comment_start*dpi_val,(fig_y_anchor_timestamp-fig_y_step_timestamp*v)*dpi_val),
                                        xycoords='figure pixels')
                                    have_i_printed = True
                                v += 1
                            del v

                            ### Title and Labels ###
                            graph.set_title(thermistor+"_"+temperature+"_in_range_"+str(nth_range))
                            graph.set_xlabel("Time")
                            graph.set_ylabel("Resistance")
                            graph.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y/%m/%d %H:%M'))

                            ### All of the Data ###
                            graph.plot(self.df["Time"][rng_start:rng_end],self.df[thermistor][rng_start:rng_end], color="blue", label="Thermometry Data")
                            ### Plots a line @ (1+0.05)*avg of the graph when the magnet is on ###
                            graph.plot(data_record_slice["Time"],data_record_slice["State"]*avg*1.05,color='k',label="When the magnet is on")

                            ### Average Dashed Line ###
                            graph.plot(
                                (df_xslice[rng_ss],df_xslice[rng_ee-1]),
                                (avg,avg),'g', dashes=[30, 30], label="Average Value of selected Range")
                            
                            ### Red Lines ###
                            graph.plot(
                                (df_xslice[rng_ss],df_xslice[rng_ss]),
                                (avg-max(self.df[thermistor][rng_start:rng_end])*0.05,avg+max(self.df[thermistor][rng_start:rng_end])*0.05),
                                'r')
                            graph.annotate(
                                str(df_xslice[rng_ss]),
                                xy=(df_xslice[rng_ss], avg+max(self.df[thermistor][rng_start:rng_end])*0.053),
                                xycoords='data', color='red') # The range-of-interest start time
                            graph.plot(
                                (df_xslice[rng_ee-1],df_xslice[rng_ee-1]),
                                (avg-max(self.df[thermistor][rng_start:rng_end])*0.05,avg+max(self.df[thermistor][rng_start:rng_end])*0.05),
                                'r')
                            graph.annotate(
                                str(df_xslice[rng_ee-1]),
                                xy=(df_xslice[rng_ee-1], avg+max(self.df[thermistor][rng_start:rng_end])*0.053),
                                xycoords='data', color='red') # The range of interest end time

                            ### Annotations ###
                            big_fig.annotate(
                                "Average: "+str(avg)+"\n"+"Standard Deviation: "+str(std)+\
                                '\nRange: '+str(rng)+'\nRange length: '+str(d_points)+\
                                '\nColumn Length: '+str(len(self.df[thermistor])),
                                xy=(fig_x_basic_info*dpi_val,fig_y_basic_info*dpi_val), 
                                xycoords='figure pixels')
                            big_fig.annotate(
                                self.df["Time"][rng_start],
                                xy=(fig_x_start_range_data*dpi_val,fig_y_range_data*dpi_val),
                                xycoords='figure pixels')
                            big_fig.annotate(
                                self.df["Time"][rng_end-1],
                                xy=(fig_x_end_range_data*dpi_val,fig_y_range_data*dpi_val),
                                xycoords='figure pixels')
                            big_fig.annotate(
                                "Logbook comments:",
                                xy=(fig_x_logbook_comment*dpi_val,fig_y_logbook_comment*dpi_val),
                                xycoords='figure pixels')

                            ### Save Plot ###
                            graph.legend(loc='best')
                            plt.savefig(thermistor+"_"+temperature+"_in_range_"+str(nth_range)+".png")
                            print("Generated: ", thermistor+"_"+temperature+"_in_range_"+str(nth_range)+".png")
                            plt.close('all')
                            plt.clf()
                    gc.collect() # You will run out of memory if you do not do this.

    def plotting(self, rangeshift=1, nbest=20, range_length=None, dpi_val=5, logbook=True):
        self.load_data()
        self.plot_calibration_candidates(n_best=nbest, dpi_val=dpi_val, plot_logbook=logbook)

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


#class graphing(object)
#   def __init__(self, dataset, thermistor, comments=None, keywords=None)
#       Decoupple Logbook data, and keywords from from plotting