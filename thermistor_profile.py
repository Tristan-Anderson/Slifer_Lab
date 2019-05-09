import pandas, traceback, datetime, time, numpy, os, glob, matplotlib, scipy, pprint
from scipy import optimize, stats
from matplotlib import pyplot as plt
import _pickle as pickle

"""
USE CAUTION WITH AB.F6. THE PREVIOUS THERMISTOR BROKE SO
THE CURRENT CALIBRATION IS NOT VALID. AB.F6 IS R6.
"""

def function(r,a,b,c):
    return a+b*numpy.exp(c*(1000/r))

class thermistor_profile(object):
    ###############################################################
    """
       This class was created as a tool to better handle and
          manipulate thermometry calibrations. It takes a
       mise-en plas approach requiring a single file to operate.
          The file it requires is a simple .csv that stores
         calibration coefficents, and initial points that this
            program uses to find the coefficents of the ntc 
                            thermistor curve 
                                                                """
    ###############################################################
    def __init__(self, name, parsed_path, parsed_slice, profile=None, changelog="thermometry_changelog.csv"):
        self.parsed_slice = parsed_slice
        self.parsed_path = parsed_path
        self.droppit = ['a', 'b', 'c']
        self.profile_path = profile
        self.changelog_path = changelog
        self.name = name
        self.pointlabels = []
        self.datapoints = 0

    def __debug_attribute(self, obj):
        with open("debug.txt", 'w') as fout:
            pprint.pprint(obj, fout)
        print("Printed object to file: debug.txt")

    def __load_coefficents(self, do_print=True):
        if self.profile_path == None:
            if do_print:
                print(
                    "No coefficent name or path was given during initalization! \
                    \nAssuming coefficent csv name to be: \"curve_coefficent_data.csv\"")
            self.profile_path = "curve_coefficent_data.csv"
        with open(self.profile_path, 'r') as f:
            self.profile = pandas.read_csv(f, sep=',')
        self.profile = self.profile.set_index("Name")

    def __load_changelog(self, do_print=True):
        if self.changelog_path == None:
            if do_print:
                print(
                    "No logbook name or path was given during initalization! \
                    \nDefaulting to: \"thermometry_changelog.csv\"")
            self.changelog_path = "thermometry_changelog.csv"
        with open(self.changelog_path, 'r') as f:
           self.changelog = pandas.read_csv(f, sep='\t')

    def write_coefficents(self):
        with open("curve_coefficent_data.csv", 'w') as f:
            self.profile.to_csv(f, sep=',')

    def write_changelog(self, message):
        with open(self.changelog_path, 'a') as f:
            f.write(str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3])+'\t'+str(self.name)+'\t'+str(message))

    def __available_thermistor_temperatures(self):
        if len(self.datapoints) == 3:
            return numpy.array([290, 77.36, 4.2])
        elif len(self.datapoints) == 4:
            return numpy.array([290, 175.5, 77.36, 4.2])
        else:
            raise Ydata_Not_Found(self.name)

    def __logbook_entry(self, message):
        self.__load_changelog(do_print=False)
        entry = pandas.DataFrame({"Timestamp":datetime.datetime.now().strftime("%Y/%m/%d/ %H:%M:%S"),
            "Name":self.name, "Comment":message}, columns=["Timestamp", "Name", "Comment"], index=[1,2,3])
        self.changelog = self.changelog.append(entry, ignore_index=True)
        self.write_changelog()

    def __execute_instructions(self, instructions):
        self.__load_coefficents(do_print=False)  
        for instruction in instructions:
            temp = instruction[0]
            cut = int(instruction[1])
            before = self.profile.loc[temp, self.name]
            if self.parsed_slice[temp].loc[int(cut), "AVG"] == 0:
                print("Average in range", cut,"from", self.parsed_path, "is 0... Skipping.")
                continue
            if self.parsed_slice[temp].loc[int(cut), "STD"] == 0:
                print("Standard Deviation is ZERO in range",cut,"from")
                continue
            try:
                after = self.parsed_slice[temp].loc[int(cut), "AVG"]
            except KeyError:
                print("Bad data from graph: "+self.name+"_"+temp+"_in_range_"+str(cut)+".png", '\n', cut, "Does not exist in parsed data at specified temperature range.")
                continue
            self.profile.loc[temp, self.name] = after
            self.write_changelog("Changed "+self.name+" "+str(temp)+" Calpoint "+str(before)+" To "+str(after) + " From "+self.parsed_path+'\n')
            self.calibrate_curve(plotting=True)

    def auto_update_calpoint(self):
        instructions = []
        self.__load_coefficents(do_print=False)  
        for file in os.listdir():
            if file.endswith(".png"):
                if file.split("_")[0] == self.name:
                    if self.name in self.profile.columns.values:
                        instructions.append([file.split("_")[1], file.split("_")[-1].split('.')[0]])
        self.__execute_instructions(instructions)

    def calibrate_curve(self, plotting=False):
        self.__load_coefficents(do_print=False)
        if self.name in self.profile.columns.values:
            self.datapoints = self.profile.loc[self.profile.index.values, self.name].sort_values()
            self.datapoints = self.datapoints.drop(self.droppit)
            self.datapoints = self.datapoints.dropna(axis=0)
            
            rows = self.datapoints.index.values
            dp = [x for x in self.datapoints]
            popt, pconv = optimize.curve_fit(function, self.datapoints, self.__available_thermistor_temperatures())
            coeff = popt[0:3]
            (a,b,c) = coeff
            self.coefficent_list = ['a', 'b', 'c']
            self.profile.loc[self.coefficent_list, self.name] = coeff
            self.write_coefficents()
            if plotting:
                if "AB" in self.name:
                    xdata = sorted([i for i in range(120,3000)],reverse=True)
                if "CCS" in self.name:
                    xdata = sorted([i for i in range(875,4500)],reverse=True)
                ydata = []
                for i in xdata:
                    ydata.append(function(i,a,b,c))
                k = list(zip(self.datapoints, self.__available_thermistor_temperatures()))
                dpi=150
                figure = plt.figure(figsize=(8,8), dpi=dpi)
                graph=figure.add_subplot(111)
                graph.plot(self.datapoints, self.__available_thermistor_temperatures(), 'ro',label="Initial Points")
                graph.plot(xdata,ydata, label="Curve", color="blue")
                graph.set_title(self.name)
                graph.set_xlabel("Resistance (Ohms)")
                graph.set_ylabel("Temperature (Kelvin)")
                graph.legend(loc="best")
                n = 0
                rows = self.datapoints.index.values
                for i in rows:
                    graph.annotate(i, xy=k[n], xycoords='data', color='k')
                    n += 1
                graph.set_ylim(bottom=0, top=310)
                graph.annotate("Initial Points: "+ str(k), xy=(2*dpi,6*dpi), xycoords='figure pixels')
                graph.annotate("a: "+str(a), xy=(2*dpi,5.85*dpi), xycoords='figure pixels')
                graph.annotate("b: "+str(b), xy=(2*dpi,5.7*dpi), xycoords='figure pixels')
                graph.annotate("c: "+str(c), xy=(2*dpi,5.55*dpi), xycoords='figure pixels')
                graph.annotate("T = a + b*exp(1000*c/r)", xy=(2*dpi,5.4*dpi), xycoords='figure pixels')
                if len(self.datapoints) > 3:
                    chi_expected = [function(i,a,b,c) for i in self.datapoints]
                    chsq = scipy.stats.chisquare(self.__available_thermistor_temperatures(), f_exp=chi_expected)
                    print(chsq)
                    graph.annotate("One-way Chisquared: p = "+str(chsq[1]), xy=(2*dpi,5.25*dpi), xycoords='figure pixels')
                plt.savefig(self.name+".png")
                plt.close('all')
                plt.clf()
        