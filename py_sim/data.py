# this imports bar data from a spreadsheet (csv) that I found online.
# it has all of the data for cortex units and structures
import pandas as pd

def load_cortex_data(filename):
    complete = pd.read_csv(filename)
    units = complete.iloc[:, :12]
    structures = complete.iloc[:, 13:]
    return units, structures