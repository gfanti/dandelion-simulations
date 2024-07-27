# parameters for simulation
import sim_lib
import graph_lib


BTC_GRAPH_OUT_DEGREE = 8


n = 100	# number of nodes
verbose = False	# debug?

# Number of graphs used
graph_trials = 20
# graph_trials = 70

# Number of trials per graph
path_trials = 30

# ----- Out-degree of random regular graph ----#
# ds = [1,2,3]
ds = [BTC_GRAPH_OUT_DEGREE]

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
# sim_graph = graph_lib.RegGraphGen
# sim_graph_params = {}

sim_graph = graph_lib.QuasiRegGraphGenSpies
sim_graph_params = {'d_anon':2}

# sim_graph = graph_lib.QuasiRegGraphGenSpiesOutbound
# sim_graph_params = {'d_anon':2}

''' What type of simulation do you want to run?
	Syntax: sim_settings[sim_name] = (sim_type, parameters)
    - sim_name: a string of your choosing to key the sim_settings dict. This string will be 
    			used in the filename when we save the output data
    - sim_type: a string denoting the type of simulation you want to run. Options:
		- 'sim_lib.FirstSpyLineSimulator' - Dandelion propagation, first-spy estimator
        - 'sim_lib.FirstSpyDiffusionSimulator' - Diffusion propagation, first-spy estimator
        - 'sim_lib.MaxWeightLineSimulator' - Diffusion propagation, max-weight estimator
        - 'sim_lib.MaxWeightDiffusionSimulator' - Diffusion propagation, max-weight estimator
    - parameters: This is a dict of input parameters. You can choose (any subset of):
		- edgebased: what kind of message forwarding to use
			- 0: Randomize per transaction (if unspecified, the default is 0)
			- 1: Per-incoming-edge forwarding
			- 2: All-to-one forwarding
			- 3: One-to-one forwarding
			- 4: (Disabled, do not use)
		- p_and_r: Should we compute precision and recall?
			- True: compute both precision and recall (default)
			- False: compute just precision
		- q: probability of entering stem mode (0.0<=q<=1.0)
			- 0.0 (default)
		- per_tx_q: should we randomize the stem length per transaction?
			- True: yes, randomize when to fluff independently for each transaction (default)
            - False: no, randomize once per graph. Each node is randomly assigned to 
            		 diffuse all incoming messages w.p. q
    
 	Examples: 
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
	sim_settings['max_weight_dand'] = (sim_lib.MaxWeightLineSimulator, {'p_and_r':True})
	sim_settings['first_spy_dand_q_0_25_spies_misbehave'] = 
		(sim_lib.FirstSpyLineSimulator, {'p_and_r':True, 'q':0.25})
		
 '''


sim_settings = {}
# sim_settings['first_spy_dand_per_tx'] = (sim_lib.FirstSpyLineSimulator, 
# 										   {'p_and_r':True, 'edgebased':0})
# sim_settings['max_weight_dand'] = (sim_lib.MaxWeightLineSimulator, {'p_and_r':True})

sim_settings['first_spy_dand_q_0_spies_misbehave'] = \
	(sim_lib.FirstSpyLineSimulator, {'p_and_r':True, 'q':0, 'per_tx_q':False})
sim_settings['first_spy_dand_q_0_25_spies_misbehave'] = \
	(sim_lib.FirstSpyLineSimulator, {'p_and_r':True, 'q':0.25, 'per_tx_q':False})
sim_settings['first_spy_dand_q_0_25_spies_misbehave_pertxrand'] = \
	(sim_lib.FirstSpyLineSimulator, {'p_and_r':True, 'q':0.25, 'per_tx_q':True})
sim_settings['first_spy_dand_q_0_5_spies_misbehave'] = \
	(sim_lib.FirstSpyLineSimulator, {'p_and_r':True, 'q':0.5, 'per_tx_q':False})
sim_settings['first_spy_dand_q_0_5_spies_misbehave_pertxrand'] = \
	(sim_lib.FirstSpyLineSimulator, {'p_and_r':True, 'q':0.5, 'per_tx_q':True})



''' What legend do you want to plot? 
	- legend: list of strings to be used in the plot legend. This should be the 
    		  same length as the number of items in sim_settings. If legend is 
              not specified, we will use the sim_name variables for each sim in
              the legend.
 '''
# legend = []
legend = ['q=0', 
          'q=0.25, Randomize stem per graph', 
          'q=0.25, Randomize stem per tx',
          'q=0.5, Randomize stem per graph',
          'q=0.5, Randomize stem per tx']