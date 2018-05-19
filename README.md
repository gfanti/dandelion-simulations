# dandelion-simulations
These simulations explore the design space for Dandelion spreading, in conjunction with the paper 
Dandelion++: Lightweight Cryptocurrency Networking with Formal Anonymity Guarantees (2018).

To reproduce our experiments, you will need to set the appropriate parameters in 
config_random_regular.py, and then run:

`python random_regular_propagation.py`

The appropriate parameter configurations are listed below. 

### Figure 5: Recall vs number of transactions per node in random 4-regular graphs



### Figure 6: First-spy precision for 4-regular graphs under various routing schemes.

In config_random_regular.py: 

`sim_graph = graph_lib.RegGraphGen
sim_graph_params = {}


sim_settings['first_spy_dand_per_tx'] = (sim_lib.FirstSpyLineSimulator, 
										   {'p_and_r':True, 'edgebased':0})
sim_settings['first_spy_dand_per_edge'] = (sim_lib.FirstSpyLineSimulator, 
									{'p_and_r':True, 'edgebased':1})
sim_settings['first_spy_dand_all_to_one'] = (sim_lib.FirstSpyLineSimulator, 
											{'p_and_r':True, 'edgebased':2})
sim_settings['first_spy_dand_one_to_one'] = (sim_lib.FirstSpyLineSimulator,
											{'p_and_r':True, 'edgebased':3})
sim_settings['first_spy_diffusion'] = (sim_lib.FirstSpyDiffusionSimulator, {})`


The file config_random_regular.py contains the simulation configurations needed to run the 
simulations. Currently, the file is configured to run a random regular graph, with different types 
of routing: per-tx, per-incoming-edge, all-to-one, one-to-one, and diffusion. 

### Figure 7: Average precision for exact 4-regular vs. quasi-4-regular

You will have to run the code twice, with difference parameter settings.

quasi-4-regular:
In config_random_regular.py:

`sim_graph = graph_lib.QuasiRegGraphGen
sim_graph_params = {'d_anon':2}

sim_settings['first_spy_dand_per_tx'] = (sim_lib.FirstSpyLineSimulator, 
										   {'p_and_r':True, 'edgebased':0})
sim_settings['max_weight_dand'] = (sim_lib.MaxWeightLineSimulator, 
										   {'p_and_r':True})`


exact 4-regular
In config_random_regular.py:
`sim_graph = graph_lib.RegGraphGen
sim_graph_params = {}

sim_settings['first_spy_dand_per_tx'] = (sim_lib.FirstSpyLineSimulator, 
										   {'p_and_r':True, 'edgebased':0})
sim_settings['max_weight_dand'] = (sim_lib.MaxWeightLineSimulator, 
										   {'p_and_r':True})`


### Figure 8: Honest-but-curious spies obey graph construction protocol

### Figure 9: Malicious spies make outbound connections to every node


### Figures 14 and 15:


