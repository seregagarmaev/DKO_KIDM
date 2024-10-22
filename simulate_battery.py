from config import battery_varying_load_params as params
from src.simulate_battery import simualte_li_ion_battery

def main():
    simulate_li_ion_battery(params)

if __name__ == "__main__":
    main()