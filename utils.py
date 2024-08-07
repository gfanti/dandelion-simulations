# utils.py
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Spreading mechanism constants
TRICKLE = 0
DIFFUSION = 1
# Estimator Constants
FIRST_SPY = 0
MAX_LIKELIHOOD = 1
LOCAL_OPT = 2
CENTRALITY = 3


def write_results(results_names, results_data, param_types, params, run_num = None):
	''' Writes a file containing the parameters, then prints each
	result name with the corresponding data '''

	filename = 'results/results' + "_".join([str(i) for i in params[0]])

	if not (run_num is None):
		filename += '_run' + str(run_num)


	f = open(filename, 'w')

	for (param_type, param) in zip(param_types, params):
		f.write(param_type)
		f.write(': ')
		for item in param:
			f.write("%s " % item)		
		f.write('\n')

	for (result_type, result) in zip(results_names, results_data):
		f.write(result_type)
		f.write('\n')

		for item in result:
			f.write("%s " % item)
		f.write('\n')

	f.close()

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--run", type=int,
	                    help="changes the filename of saved data")
	parser.add_argument("-v", "--verbose", help="increase output verbosity",
	                    action="store_true")
	parser.add_argument("-w", "--write", help="writes the results to file",
	                    action="store_true")
	parser.add_argument("-t","--trials", type=int, help="number of trials",
						default=1)
	parser.add_argument("-s","--spreading", type=int, help="Which spreading protocol to use (0)trickle, (1)diffusion",
						default=0)
	parser.add_argument("-e","--estimator", dest='estimators',default=[], type=int, 
						help="Which estimator to use (0)first-spy, (1)ML, (2)local diffusion", action='append')
	parser.add_argument("--measure_time", help="measure runtime?",
						action="store_true")
	args = parser.parse_args()

	if not (args.run is None):
		args.write = True

	print('---Selected Parameters---')
	print('verbose: ', args.verbose)
	print('write to file: ', args.write)
	print('spreading mechanism: ', args.spreading)
	print('estimators: ', args.estimators)
	print('run: ', args.run)
	print('num trials: ', args.trials, '\n')
	return args

def plot_results(means, stds, ps, labels=None):
	

	# Create the plot
	for idx in range(means.shape[0]):
		print("labels", labels)
		if any(labels):
			plt.errorbar(ps, means[idx], yerr=stds[idx], fmt='o-', label=list(labels)[idx])
		else:
			plt.errorbar(ps, means[idx], yerr=stds[idx], fmt='o-')
			


	# Customize the plot
	# plt.title("Plot of p_means with Error Bars")
	plt.xlabel("Fraction of Spies (p)")
	plt.ylabel("Mean Precision")
	plt.grid(True)
	if any(labels):
		plt.legend()
	plt.show()

def get_num_honest_nodes(G):
	# List of all spies
	spies = nx.get_node_attributes(G,'spy')
	num_spies = list(spies.values()).count(True)
	

	return (G.number_of_nodes() - num_spies)

def plot_graph(G):
	pos = nx.circular_layout(G)
	labels = {}
	for n in G.nodes():
		labels[n] = str(n)
	spies = nx.get_node_attributes(G,'spy')
	print('num spies', spies)
	val_map = {True: 'r',
           		False: 'b'}
	values = [val_map[i] for i in list(spies.values())]

	nx.draw(G, pos, node_color = values)
	nx.draw_networkx_labels(G, pos, labels, font_size = 16)
	plt.show()