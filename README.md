
# Dandelion++ Simulations
This repository contains the simulation code used to generate the figures in the paper Dandelion++: Lightweight Cryptocurrency Networking with Formal Anonymity Guarantees (2018). These simulations compare Dandelion spreading (and several of its variants) to diffusion on different types of graphs. 

To reproduce our experiments, you will need to set the appropriate parameters in 
config_random_regular.py, and then run:

`python random_regular_propagation.py`

The appropriate parameter configurations for the different figures are listed below. 

#### Figure 5: Recall vs # of transactions per node in random 4-regular graphs



#### Figure 6: First-spy precision for 4-regular graphs for various routing schemes.

In config_random_regular.py: 

```
# graph type
sim_graph = graph_lib.RegGraphGen
sim_graph_params = {}

# simulation settings
sim_settings['first_spy_dand_per_tx'] = (sim_lib.FirstSpyLineSimulator, 
										   {'p_and_r':True, 'edgebased':0})
sim_settings['first_spy_dand_per_edge'] = (sim_lib.FirstSpyLineSimulator, 
									{'p_and_r':True, 'edgebased':1})
sim_settings['first_spy_dand_all_to_one'] = (sim_lib.FirstSpyLineSimulator, 
											{'p_and_r':True, 'edgebased':2})
sim_settings['first_spy_dand_one_to_one'] = (sim_lib.FirstSpyLineSimulator,
											{'p_and_r':True, 'edgebased':3})
sim_settings['first_spy_diffusion'] = (sim_lib.FirstSpyDiffusionSimulator, {})
```


#### Figure 7: Average precision for exact 4-regular vs. quasi-4-regular

You will have to run the code twice, with difference parameter settings.

quasi-4-regular:

```
# graph type
sim_graph = graph_lib.QuasiRegGraphGen
sim_graph_params = {'d_anon':2}

# simulation settings
sim_settings['first_spy_dand_per_tx'] = (sim_lib.FirstSpyLineSimulator, 
										   {'p_and_r':True, 'edgebased':0})
sim_settings['max_weight_dand'] = (sim_lib.MaxWeightLineSimulator, 
										   {'p_and_r':True})
```


exact 4-regular:

```
# graph type
sim_graph = graph_lib.RegGraphGen
sim_graph_params = {}

# simulation settings
sim_settings['first_spy_dand_per_tx'] = (sim_lib.FirstSpyLineSimulator, 
										   {'p_and_r':True, 'edgebased':0})
sim_settings['max_weight_dand'] = (sim_lib.MaxWeightLineSimulator, 
										   {'p_and_r':True})
```

#### Figure 8: Honest-but-curious spies obey graph construction protocol

```
# main p2p graph degree
ds = [BTC_GRAPH_OUT_DEGREE]

# graph type
sim_graph = graph_lib.QuasiRegGraphGenSpies
sim_graph_params = {'d_anon':2}

# simulation settings
sim_settings['first_spy_dand_q_0_spies_misbehave'] = (sim_lib.FirstSpyLineSimulator, 
										   {'p_and_r':True, 'q':0})
sim_settings['first_spy_dand_q_0_25_spies_misbehave'] = (sim_lib.FirstSpyLineSimulator, 
										   {'p_and_r':True, 'q':0.25})
sim_settings['first_spy_dand_q_0_5_spies_misbehave'] = (sim_lib.FirstSpyLineSimulator, 
										   {'p_and_r':True, 'q':0.5})
```


#### Figure 9: Malicious spies make outbound connections to every node

```
# main p2p graph degree
ds = [BTC_GRAPH_OUT_DEGREE]

# graph type
sim_graph = graph_lib.QuasiRegGraphGenSpiesOutbound
sim_graph_params = {'d_anon':2}

# simulation settings
sim_settings['first_spy_dand_q_0_spies_misbehave'] = (sim_lib.FirstSpyLineSimulator, 
										   {'p_and_r':True, 'q':0})
sim_settings['first_spy_dand_q_0_25_spies_misbehave'] = (sim_lib.FirstSpyLineSimulator, 
										   {'p_and_r':True, 'q':0.25})
sim_settings['first_spy_dand_q_0_5_spies_misbehave'] = (sim_lib.FirstSpyLineSimulator, 
										   {'p_and_r':True, 'q':0.5})
```

#### Figures 14 and 15:


