import sys
import yaml
from jetracermpc.model import JetRacerModel

if __name__ == "__main__":
    config_file = sys.argv[1]
    solver_folder = sys.argv[2]
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    jet_racer_model = JetRacerModel(config)
    jet_racer_model.generate_solver(location=solver_folder)
