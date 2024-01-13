import sys
import yaml
from jetracermpc.model import JetRacerModel

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)

    jet_racer_model = JetRacerModel(config)
    jet_racer_model.generate_solver(sys.argv[2])
