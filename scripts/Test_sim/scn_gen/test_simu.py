import sys  # noqa: I001
import os

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))
print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))

print(CURRENT_PATH)

# from traffic.core import Traffic
from data_orly.src.simulation import Simulator,scn_file_to_minisky,launch_simulation
# from data_orly.src.generation.evaluation import Evaluator

#print(scn_file_to_minisky('/home/arnault/traffic/data_orly/scn/B738_VAE_TCN_Vamp__800.scn'))

launch_simulation('/home/arnault/traffic/data_orly/scn/B738_VAE_TCN_Vamp__800.scn','/home/arnault/traffic/data_orly/sim_logs/test_1.log',3)