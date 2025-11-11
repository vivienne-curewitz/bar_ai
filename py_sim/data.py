# this imports bar data from a spreadsheet (csv) that I found online.
# it has all of the data for cortex units and structures
import pandas as pd

def load_cortex_data():
    units = pd.read_csv("Cortex_Units.csv")
    structures = pd.read_csv("Cortex_Structures.csv")
    return units, structures

build_power = {
    "Construction Bot": 80,
    "Construction Vehicle": 80,
    "Construction Aircraft": 80,
    "Advanced Construction Bot": 125,
    "Advanced Construction Vehicle": 125,
    "Advanced Construction Aircraft": 125,
    "Cortex Commander": 300,
    "Construction Turret": 200,
}