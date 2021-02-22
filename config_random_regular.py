# parameters for simulation
import sim_lib
import graph_lib


BTC_GRAPH_OUT_DEGREE = 8


n = 100	# number of nodes
verbose = False	# debug?

# Number of graphs used
graph_trials = 20

# Number of trials per graph
path_trials = 20
# path_trials = 1

# ----- Out-degree of random regular graph ----#
# ds = [4,6,8]
ds = [4]

# ----- Fraction of spies ----#
# ps = [0.02, 0.04, 0.08,  0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5]
ps = [ 0.25, 0.3, 0.4, 0.45, 0.5]




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
					sim_graph_params = {'d_anon':2}
	Random quasi-regular with spies connecting to all nodes:
					sim_graph = graph_lib.QuasiRegGraphGenSpiesOutbound
					sim_graph_params = {'d_anon':2}
    Use a snapshot of the Bitcoin graph:
    				sim_graph = graph_lib.DataGraphGen
    				sim_graph_params = {}
    Complete graph:
    				sim_graph = graph_lib.CompleteGraphGen
					sim_graph_params = {}
 '''
sim_graph = graph_lib.RegGraphGen
sim_graph_params = {}


''' Populate the simulation type and parameters
 Parameters:
	p_and_r:	Compute precision and recall (true) or just precision (false)
	q:			Probability of transitioning to fluff (for dandelion)

 Options:
 	sim_settings['first_spy_dand_per_tx'] =
 		(sim_lib.FirstSpyLineSimulator, {'p_and_r':True, 'edgebased':0})
	sim_settings['first_spy_dand_per_edge'] =
		(sim_lib.FirstSpyLineSimulator, {'p_and_r':True, 'edgebased':1})
	sim_settings['first_spy_dand_all_to_one'] =
		(sim_lib.FirstSpyLineSimulator, {'p_and_r':True, 'edgebased':2})
	sim_settings['first_spy_dand_one_to_one'] =
		(sim_lib.FirstSpyLineSimulator, {'p_and_r':True, 'edgebased':3})
	sim_settings['first_spy_diffusion'] =
		(sim_lib.FirstSpyDiffusionSimulator, {})
	sim_settings['dandelion_lite'] =
		(sim_lib.DandelionLiteSimulator, {})
	sim_settings['max_weight_dand'] = (sim_lib.MaxWeightLineSimulator, {'p_and_r':True})
	sim_settings['first_spy_dand_q_0_25_spies_misbehave'] =
		(sim_lib.FirstSpyLineSimulator, {'p_and_r':True, 'q':0.25})

 '''

sim_settings = {}
sim_settings['first_spy_diffusion'] = (sim_lib.FirstSpyDiffusionSimulator,
										{'p_and_r':True})
sim_settings['first_spy_dand_q_1_000_spies_misbehave'] = \
		(sim_lib.FirstSpyLineSimulator, {'p_and_r':True, 'q':1.0})
sim_settings['first_spy_dand_q_0_25_spies_misbehave'] = \
		(sim_lib.FirstSpyLineSimulator, {'p_and_r':True, 'q':0.25})
sim_settings['first_spy_dand_q_0_00_spies_misbehave'] = \
		(sim_lib.FirstSpyLineSimulator, {'p_and_r':True, 'q':0.0})
sim_settings['dandelion_lite'] = (sim_lib.DandelionLiteSimulator, {'p_and_r':True, \
																   'beyond_stem':True})

# sim_settings['max_weight_dand'] = (sim_lib.MaxWeightLineSimulator, {'p_and_r':True})


