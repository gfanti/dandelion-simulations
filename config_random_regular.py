# parameters for simulation
import sim_lib
import graph_lib


n = 100	# number of nodes
verbose = False	# debug?

# Number of graphs used
graph_trials = 1
# graph_trials = 70

# Number of trials per graph
path_trials = 30

# ----- Out-degree of random regular graph ----#
ds = [2]

# ----- Fraction of spies ----#
ps = [ 0.25, 0.3, 0.4, 0.45, 0.5]


BTC_GRAPH_OUT_DEGREE = 8
DIFFUSION = 2



''' What type of graph do you want to run? 
  These simulations are run either on random regular graphs or a variant of a random quasi-regular 
  graphs. You need to specify the type of graph, and the desired input parameters. Here are some
  example options:

    Random regular: sim_graph = graph_lib.RegGraphGen
    				sim_graph_params = {}
    Random quasi-regular with spies: 
    				sim_graph = graph_lib.QuasiRegGraphGen
					sim_graph_params = {'d_anon':2}
  	Random quasi-regular with degree-checking (check k nodes), spies lie about degree: 
  					sim_graph = graph_lib.QuasiRegGraphGenSpies
					sim_graph_params = {'k':2}
    Use a snapshot of the Bitcoin graph: 
    				sim_graph = graph_lib.DataGraphGen
    				sim_graph_params = {}
    Complete graph: 
    				sim_graph = graph_lib.CompleteGraphGen
					sim_graph_params = {}
 '''
# sim_graph = graph_lib.RegGraphGen
# sim_graph_params = {}

sim_graph = graph_lib.QuasiRegGraphGen
sim_graph_params = {'d_anon':2}


''' Populate the simulation settings (Simulator function, p_and_r, edgebased)
 edgebased describes the type of routing used on a not-line graph '''
sim_settings = {}
sim_settings['first_spy_dand_per_tx'] = (sim_lib.FirstSpyLineSimulator, 
										   {'p_and_r':True, 'edgebased':0})
# sim_settings['max_weight_dand'] = (sim_lib.MaxWeightLineSimulator, 
# 								   {'p_and_r':True})
# sim_settings['first_spy_dand_per_edge'] = (sim_lib.FirstSpyLineSimulator, 
# 									{'p_and_r':True, 'edgebased':1})
# sim_settings['first_spy_dand_all_to_one'] = (sim_lib.FirstSpyLineSimulator, True, 2)
# sim_settings['first_spy_dand_one_to_one'] = (sim_lib.FirstSpyLineSimulator, True, 3)
# sim_settings['first_spy_diffusion'] = [sim_lib.FirstSpyDiffusionSimulator]



