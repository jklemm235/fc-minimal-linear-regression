from pathlib import Path
from logic import fl_algorithm
from helper.run_app_simulation import run_simulation_featurecloud, run_simulation_native

# Set up which data/config is used
# you only need to change this
SCRIPT_FOLDER = Path(__file__).parent
SAMPLE_FOLDER = SCRIPT_FOLDER / "sample_data"
INPUTFOLDERS = [SAMPLE_FOLDER / "lab1", SAMPLE_FOLDER / "lab2", SAMPLE_FOLDER / "lab3"]
OUTPUTFOLDERS = [SAMPLE_FOLDER / "lab1_out", SAMPLE_FOLDER / "lab2_out", SAMPLE_FOLDER / "lab3_out"]
GENERIC_DIR = SAMPLE_FOLDER / "generic_dir"
    # Note: the files in this folder are placed in ALL inputfolders before execution
SIMULATION_TYPE = "featurecloud"  # "featurecloud" or "native"

### LOGIC
# can be ignored
if __name__ == "__main__":
    if SIMULATION_TYPE == "featurecloud":
        run_simulation_featurecloud(
            data_path=str(SAMPLE_FOLDER),
            clientnames=[folder.name for folder in INPUTFOLDERS],
            generic_dir=GENERIC_DIR.name
        )
    elif SIMULATION_TYPE == "native":
        run_simulation_native(
            clientpaths=[str(folder) for folder in INPUTFOLDERS],
            outputfolders=[str(folder) for folder in OUTPUTFOLDERS],
            generic_dir=str(GENERIC_DIR),
            fl_algorithm_function=fl_algorithm
        )
    else:
        raise ValueError(f"Invalid SIMULATION_TYPE: {SIMULATION_TYPE}. Must be 'featurecloud' or 'native'.")
