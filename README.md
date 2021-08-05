
August 2021.
This code is used to run an experiment as described in the enclosed paper (Data_Science_in_Cell_Imaging_S2_Project.pdf)
In order to run it you myst use python 3.x and require a few libraries as pandas, numpy and tensorflow to be installed.
It was developed and run using windows 10 
it is built as a single script and (runner.py) that run a single experiment using the method run()
it expect the plates data to locate in directories tree with the following pattern: "data\profiles.dir\Plate_XYZK\profiles\mean_well_profiles.csv"
and the dataset chemical_annotations, to be located in data\chemical_annotations.csv.
it output the results into the 'ExperimentsResults' directory.
Good Luck
