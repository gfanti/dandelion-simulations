# dataset_main.py
# Simulates diffusion and Dandelion spreading on a dataset

from dataset_graph_rep import *
from utils import *
import time
import numpy as np



# filename = 'data/bitcoin.gexf'
filename = 'data/random_regular.gexf'
# filename = 'data/tree_5.gexf'

# Mean delay of diffusion spreading, in seconds
diffusion_rate = 1.0 / 2.5

args = parse_arguments()

hops_list = [6,10]
spreading_time = 300

G = DataGraphDandelion(filename, spreading_time = spreading_time, lambda1 = diffusion_rate)


timing_results = []
timing_stds = []
for hops in hops_list:
	print('Number of hops ', hops)

	runtimes = []
	for trial in range(args.trials):

		if (trial % 10) == 0:
			print('On trial ', trial+1, ' out of ', args.trials)

		source = random.choice(G.nodes())
	
		# Spread the message
		G.spread_message(source, hops = hops)

		time_to_spread = max(G.received_timestamps.values())

		runtimes.append(time_to_spread)
		# avg_time += time_to_spread


	avg_time  = np.mean(np.array(runtimes))
	std_time = np.std(np.array(runtimes))

	timing_results.append(avg_time)
	timing_stds.append(std_time)
	print('Timing results', timing_results)

print('hops', hops_list)
print('timing', timing_results)
print('standard deviation', timing_stds)	
