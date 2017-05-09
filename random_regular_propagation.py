# test how well we can generate a set of random line segments from a regular random graph

import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import collections
import math
import scipy.io
from graph_lib import *
import sim_lib





def get_num_honest_nodes(G):
	# List of all spies
	spies = nx.get_node_attributes(G,'spy')
	num_spies = spies.values().count(True)
	# print 'num spies', num_spies

	return (G.number_of_nodes() - num_spies)

def plot_graph(G):
	pos = nx.circular_layout(G)
	labels = {}
	for n in G.nodes():
		labels[n] = str(n)
	spies = nx.get_node_attributes(G,'spy')
	print 'num spies', spies
	val_map = {True: 'r',
           		False: 'b'}
	values = [val_map[i] for i in spies.values()]

	nx.draw(G, pos, node_color = values)
	nx.draw_networkx_labels(G, pos, labels, font_size = 16)
	plt.show()



if __name__=='__main__':

	n = 200	# number of nodes
	# d = 2	 # outdegree of graph
	# p = 0.2 # probability of spiesx
	verbose = False	# debug?

	graph_trials = 100
	# graph_trials = 150
	# graph_trials = 40

	# path_trials = 50
	path_trials = 20

	use_main = True		# Use the main P2P graph (true) or the anonymity graph (false)


	d_means = []
	d_stds = []

	# ----- Out-degree of graph ----#
	# ds = [1,2,3]
	ds = [2]

	# ----- Fraction of spies ----#
	# ps = [0.2]
	# ps = np.arange(0.1,0.51,0.1)
	ps = [0.2]

	for d in ds:
		print 'd is ', d
		p_means = []
		p_stds = []

		for p in ps:
			print 'p is', p
			graph_precisions = []
			graph_recalls = []
			for i in range(graph_trials):
				if (i%10 == 0):
					print 'Trial ', i, ' of ', graph_trials

				# Generate the graph
				# gen = RegGraphGen(n, p, d, verbose)  # d-regular
				gen = QuasiRegGraphGen(n, p, d, verbose) # quasi d-regular
				# gen = DataGraphGen('data/bitcoin.gexf', p, verbose) # Bitcoin graph
				# gen = QuasiRegThreshGraphGen(n, p, d, k, verbose) # quasi d-regular
				# gen = CompleteGraphGen(n, p, verbose)  # complete graph
				if use_main:
					G = gen.G
				else:
					G = gen.A
				# print 'G loaded', nx.number_of_nodes(G), ' nodes' 
				
				num_honest_nodes = get_num_honest_nodes(G)

				# Corner cases
				if (num_honest_nodes == n) or (num_honest_nodes == 0):
					if (num_honest_nodes == n):
						graph_precision = 0.0
					elif num_honest_nodes == 0:
						graph_precision = 0.0
					graph_precisions.append(graph_precision)
					continue
				# print G.edges()

				# Initialize precision
				graph_precision = 0
				graph_recall = 0


				for j in range(path_trials):

					# run a simulation
					# sim = sim_lib.FirstSpyLineSimulator(G, num_honest_nodes, verbose, p_and_r = True)
					# sim = sim_lib.FirstSpyDiffusionSimulator(G, num_honest_nodes, verbose)
					sim = sim_lib.MaxWeightLineSimulator(G, num_honest_nodes, verbose)
					
					# retrieve the precision
					graph_precision += sim.precision
					graph_recall += sim.recall

					# print 'precision', precision

				graph_precision = graph_precision / path_trials
				graph_recall = graph_recall / path_trials
				if verbose:
					print 'Graph precision: ', graph_precision
					print 'Graph recall: ', graph_recall

				graph_precisions.append(graph_precision)
				graph_recalls.append(graph_recall)


				if verbose:
					# plot the graph
					plot_graph(G)

			
			# print 'Final result: ', graph_precisions
			mean_precision = np.mean(graph_precisions)
			std_precision = np.sqrt(mean_precision * (1-mean_precision) / graph_trials / path_trials)
			print 'Mean precision:' , mean_precision
			print 'Std precision:' , std_precision

			mean_recall = np.mean(graph_recalls)
			std_recall = np.sqrt(mean_recall * (1-mean_recall) / graph_trials / path_trials)
			print 'Mean recall:' , mean_recall
			print 'Std recall:' , std_recall
			
			p_means.append(mean_precision)
			p_stds.append(std_precision)

		
		d_means += [p_means]
		d_stds += [p_stds]

	print 'Total d_means', np.array(d_means)
	print 'Total d_stds', np.array(d_stds)

	# scipy.io.savemat('results/d_reg_first_spy_approxk4.mat', {'ds' : ds, 'd_means' : np.array(d_means), 'd_stds' : np.array(d_stds)})
	# scipy.io.savemat('results/d_reg_6_ml.mat', {'ds' : ds, 'd_means' : np.array(d_means), 'd_stds' : np.array(d_stds)})
