import sys, random, os
from PySide2 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QObject, pyqtSignal
from slifercal import slifercal as sliferCal

class myWidget(QtWidgets.QWidget):
    def __init__(self):
        #super().__init__()
        # Actually useful self variables
        self.working_directory = os.getcwd()
        self.df_path = "data.csv"
        # Creating widgets and stuff
        self.textEdit = QtWidgets.QLineEdit()
        self.data_file_location_init = self.working_directory+'/'+"data.csv   (Default)"
        self.window = QtWidgets.QWidget()
        self.button_keywords = QtWidgets.QPushButton("Plot Keyword Hits")
        self.keyword_flavortext = QtWidgets.QLabel("Produces graphs from a given dataset based on\nuser-defined keywords that are found inside\n the dataset's logbook.")
        self.button_FSR= QtWidgets.QPushButton("Plot N most-stable regions")
        self.FSR_flavortext = QtWidgets.QLabel("Produces graphs from a given dataset based on\n the standard deviation of a slice of data that\n fits within certain temperature ranges.")
        self.button_CT = QtWidgets.QPushButton("Calibrate Thermistors")
        self.CT_flavortext = QtWidgets.QLabel("Calibrates thermistors based off of graphs created by \"Plot N most-stable regions\" (UNSTABLE)")
        self.button_OV = QtWidgets.QPushButton("Date Viewer")
        self.OV_flavortext = QtWidgets.QLabel("Shows data between two dates selected by the user and automatically saves the plot in the working directory")
        self.df_path_init = QtWidgets.QLineEdit(self.data_file_location_init)
        self.df_browse = QtWidgets.QPushButton("Browse")
        self.df_label = QtWidgets.QLabel("\n\nExperimental data file location")

        #Layout
        self.layout = QtWidgets.QGridLayout()
        self.layout.addWidget(self.button_keywords, 1, 0, 1, 1)
        self.layout.addWidget(self.button_FSR, 1, 1, 1, 2)
        self.layout.addWidget(self.button_CT, 3, 0, 3, 2)
        self.layout.addWidget(self.button_OV, 5,0,5,2)
        self.layout.addWidget(self.df_path_init, 8,0,8,1)
        self.layout.addWidget(self.df_browse, 8,1,8,2)

        self.layout.addWidget(self.keyword_flavortext, 2, 0, 2, 1)
        self.layout.addWidget(self.FSR_flavortext, 2, 1, 2, 2)
        self.layout.addWidget(self.CT_flavortext,4, 0, 4, 2)
        self.layout.addWidget(self.OV_flavortext, 6,0,6,2)
        self.layout.addWidget(self.df_label, 9,0)
        

        #Display
        self.window.setLayout(self.layout)
        self.window.resize(600, 600)
        self.window.show()

        #Trigger
        self.button_OV.clicked.connect(self.omniview_splash)
        self.button_FSR.clicked.connect(self.fsr_splash)
        self.button_CT.clicked.connect(self.ct_splash)
        self.button_keywords.clicked.connect(self.keyword_splash)
        self.df_browse.clicked.connect(self.update_df)

    def update_df(self):
        self.df_path, _ = QtWidgets.QFileDialog.getOpenFileName(self.window, 'Open File', self.working_directory, '*.csv')
        self.df_path_init.setText(self.df_path)

    def fsr_splash(self):
        self.window = QtWidgets.QWidget()
        self.window.resize(600, 600)
        try:
            with open(self.df_path, 'r') as f:
                first_line = f.readline()
        except:
            self.update_data_file_location()
        self.thermistor_names = first_line.split(',')
        for element in self.thermistor_names:
            if element == "Comment" or element == "Time":
                self.thermistor_names.remove(element)
        self.check_boxes = {}
        self.title = QtWidgets.QLabel("Plot N Most-Stable Regions")
        for thermistor in self.thermistor_names:
            self.check_boxes[thermistor] = QtWidgets.QCheckBox(thermistor)
        self.persistance = QtWidgets.QCheckBox("Persistance")
        self.persistance.setChecked(True)
        self.persistance_label = QtWidgets.QLabel("Calculating data is very recource intensive, that is why persistance is an option. If this persistance button is checked, the program will search for a\n"+" "*75+"previously parsed set of data, and attempt to produce graphs from that set.\n\n"+" "*13+ "IF you are generating a NEW data set, UNCHECK the persistence setting, and the program will calculate a new set of data for you.\n\nThis should only be done when calculating an entirely NEW set of experimental data. The current set of data stored in persitance has every single\nthermistor's data parsed and ready to go. You just need to specify how many stable regions for each thermistor in each temperature range that\n"+" "*130+"you want to plot.")
        self.GO = QtWidgets.QPushButton("Start plotting")
        self.n_box = QtWidgets.QComboBox()
        self.n_box.addItems([str(i) for i in range(1, 101)])

        self.layout = QtWidgets.QGridLayout()
        self.title.setStyleSheet("font: 15pt Comic Sans MS")
        self.layout.addWidget(self.title, 0, 2)
        row = 1
        col = 0
        self.layout1 = QtWidgets.QGridLayout()
        for key in self.check_boxes:
            self.layout1.addWidget(self.check_boxes[key], row, col)
            if col >= 8:
                col = 0
                row += 1
                continue
            col += 1
        row += 1
        self.layout1.addWidget(self.persistance, row, 4, row, 7)
        self.layout1.addWidget(self.persistance_label, row+4, 1, row+4, 7)
        self.layout.addLayout(self.layout1, 1, 0, 1, 7)
        row += 1
        self.layout.addWidget(self.n_box)
        row+=1
        self.layout.addWidget(self.GO, row, 6, row, 7) 

        self.window.setLayout(self.layout)
        self.window.show()

        self.GO.clicked.connect(self.fsr_exec)

    def ct_splash(self):
        self.window = QtWidgets.QWidget()
        self.window.resize(800, 600)
        
        self.title ="Calibrate Termistors"

    def keyword_splash(self):
        self.window = QtWidgets.QWidget()
        self.window.resize(600, 600)
        try:
            with open(self.df_path, 'r') as f:
                first_line = f.readline()
        except:
            self.update_data_file_location()
        self.thermistor_names = first_line.split(',')
        for element in self.thermistor_names:
            if element == "Comment" or element == "Time":
                self.thermistor_names.remove(element)
        self.check_boxes = {}
        self.title = QtWidgets.QLabel("Keyword Grapher")
        for thermistor in self.thermistor_names:
            self.check_boxes[thermistor] = QtWidgets.QCheckBox(thermistor)
        self.keywords = QtWidgets.QLineEdit()
        self.keywd_label = QtWidgets.QLabel("Enter keywords, seperated by commas without spaces:")
        self.persistance = QtWidgets.QCheckBox("Persistance")
        self.persistance.setChecked(True)
        self.persistance_label = QtWidgets.QLabel("Querrying takes a fairly long time, that is why persistance is an option. If this persistance button is checked, the program will search for a\n"+" "*55+"previously calculated set of keyword hits, and attempt to produce graphs from that set.\n\n IF you are generating a NEW keyword set, UNCHECK the persistence setting, and the program will overwrite the previous set of keyword\n"+" "*130 +"hits.")
        self.GO = QtWidgets.QPushButton("Start plotting")
        self.keyword_browser_button = QtWidgets.QPushButton("Browse")
        self.keyword_path = self.working_directory+'/'+"keyword_persistence.pk"+'\t'*4+"(Default)"
        self.keyword_browse_path = QtWidgets.QLineEdit(self.keyword_path)
        self.keyword_browse_path.setReadOnly(True)
        self.graphing_in_progress = QtWidgets.QLabel("")


        self.layout = QtWidgets.QGridLayout()
        self.title.setStyleSheet("font: 15pt Comic Sans MS")
        self.layout.addWidget(self.title, 0, 2)
        row = 1
        col = 0
        self.layout1 = QtWidgets.QGridLayout()
        for key in self.check_boxes:
            self.layout1.addWidget(self.check_boxes[key], row, col)
            if col >= 8:
                col = 0
                row += 1
                continue
            col += 1
        row += 1
        self.layout1.addWidget(self.persistance, row, 4, row, 7)
        self.layout1.addWidget(self.persistance_label, row+4, 1, row+4, 7)
        self.layout.addLayout(self.layout1, 1, 0, 1, 7)
        row += 1
        self.layout.addWidget(self.keywd_label, row, 0, row, 1)
        self.layout.addWidget(self.keywords, row, 2, row, 7)
        row+=1
        self.layout.addWidget(self.keyword_browse_path, row, 0 ,row, 4)
        self.layout.addWidget(self.keyword_browser_button, row, 5, row, 7)

        row += 1
        self.layout.addWidget(self.GO, row, 6, row, 7) 
        row += 1
        self.layout.addWidget(self.graphing_in_progress, row, 0, row+3, 7)
        self.graphing_in_progress.hide()

        self.window.setLayout(self.layout)
        self.window.show()

        self.GO.clicked.connect(self.keyword_exec)
        self.keyword_browser_button.clicked.connect(self.browse_keyword)

    def omniview_splash(self):
        self.window = QtWidgets.QWidget()
        self.window.resize(600, 600)
        try:
            with open(self.df_path, 'r') as f:
                first_line = f.readline()
        except:
            self.update_data_file_location()
        self.thermistor_names = first_line.split(',')
        for element in self.thermistor_names:
            if element == "Comment" or element == "Time":
                self.thermistor_names.remove(element)
        self.check_boxes = {}
        self.title = QtWidgets.QLabel("Keyword Grapher")
        for thermistor in self.thermistor_names:
            self.check_boxes[thermistor] = QtWidgets.QCheckBox(thermistor)
        self.keywords = QtWidgets.QLineEdit()
        self.keywd_label = QtWidgets.QLabel("Enter keywords, seperated by commas without spaces:")
        self.persistance = QtWidgets.QCheckBox("Persistance")
        self.persistance.setChecked(True)
        self.persistance_label = QtWidgets.QLabel("Querrying takes a fairly long time, that is why persistance is an option. If this persistance button is checked, the program will search for a\n"+" "*55+"previously calculated set of keyword hits, and attempt to produce graphs from that set.\n\n IF you are generating a NEW keyword set, UNCHECK the persistence setting, and the program will overwrite the previous set of keyword\n"+" "*130 +"hits.")
        self.GO = QtWidgets.QPushButton("Start plotting")
        self.keyword_browser_button = QtWidgets.QPushButton("Browse")
        self.keyword_path = self.working_directory+'/'+"keyword_persistence.pk"+'\t'*4+"(Default)"
        self.keyword_browse_path = QtWidgets.QLineEdit(self.keyword_path)
        self.keyword_browse_path.setReadOnly(True)
        self.graphing_in_progress = QtWidgets.QLabel("")


        self.layout = QtWidgets.QGridLayout()
        self.title.setStyleSheet("font: 15pt Comic Sans MS")
        self.layout.addWidget(self.title, 0, 2)
        row = 1
        col = 0
        self.layout1 = QtWidgets.QGridLayout()
        for key in self.check_boxes:
            self.layout1.addWidget(self.check_boxes[key], row, col)
            if col >= 8:
                col = 0
                row += 1
                continue
            col += 1
        row += 1
        self.layout1.addWidget(self.persistance, row, 4, row, 7)
        self.layout1.addWidget(self.persistance_label, row+4, 1, row+4, 7)
        self.layout.addLayout(self.layout1, 1, 0, 1, 7)
        row += 1
        self.layout.addWidget(self.keywd_label, row, 0, row, 1)
        self.layout.addWidget(self.keywords, row, 2, row, 7)
        row+=1
        self.layout.addWidget(self.keyword_browse_path, row, 0 ,row, 4)
        self.layout.addWidget(self.keyword_browser_button, row, 5, row, 7)

        row += 1
        self.layout.addWidget(self.GO, row, 6, row, 7) 
        row += 1
        self.layout.addWidget(self.graphing_in_progress, row, 0, row+3, 7)
        self.graphing_in_progress.hide()

        self.window.setLayout(self.layout)
        self.window.show()

        self.GO.clicked.connect(self.keyword_exec)
        self.keyword_browser_button.clicked.connect(self.browse_keyword)


    def browse_keyword(self):
        self.keyword_path, _ = QtWidgets.QFileDialog.getOpenFileName(self.window, 'Open File', self.working_directory, '*.pk')
        self.keyword_browse_path.setText(self.keyword_path)

    def browse_df(self):
        self.df_path, _ = QtWidgets.QFileDialog.getOpenFileName(self.window, 'Open File', self.working_directory, '*.csv')
        self.df.setText(self.keyword_path)

    def update_data_file_location(self):
        #
        self.df_path, _ = QtWidgets.QFileDialog.getOpenFileName(self.window, 'Open File', self.working_directory, '*.csv')

    def keyword_exec(self):
        self.graphing_in_progress = QtWidgets.QLabel("Graphing in progress. See Console.")
        self.window.update()
        keywords = self.keywords.text().split(',')
        thermistors = self.get_thermistor_checkbox_states()
        persistance = False
        if self.persistance.isChecked():
            persistance = True
        self.window.close()
        instance = sliferCal(processes=0, datafile_location=self.df_path, logbook_datafile_location=None)
        instance.keyword(keywords, thermistors=thermistors, persistance=persistance)

    def fsr_exec(self):
        self.n = int(self.n_box.currentText())
        instance = sliferCal(processes=0, datafile_location=self.df_path)
        instance.plot_top_n_ranges(self, n=self.n, comments=True)


    def omni_exec(self):
        pass

    def get_thermistor_checkbox_states(self):
        checked = []
        for qt_index in self.check_boxes:
            qt_checkbox = self.check_boxes[qt_index]
            if qt_checkbox.isChecked():
                checked.append(qt_index)
        index = 0
        for element in checked:
            if "\n" in element:
                error = element
                listed_error = list(error)
                listed_error.remove("\n")
                corrected = "".join(listed_error)
                checked[index] = corrected
            index += 1
        return checked


def main():
    if __name__ == "__main__":
        app = QtWidgets.QApplication([])

        widget = myWidget()
        
        sys.exit(app.exec_())

main()
