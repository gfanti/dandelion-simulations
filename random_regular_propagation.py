'''
	This script tests Dandelion spreading on random regular graphs.
	There are several variants, including quasi-regular constructions.
'''

from config_random_regular import *

import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import collections
import math
import scipy.io
import sys
import os




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
	values = [val_map[i] for i in spies.values()]

	nx.draw(G, pos, node_color = values)
	nx.draw_networkx_labels(G, pos, labels, font_size = 16)
	plt.show()

def plot_results(p_means, p_stds, r_means, r_stds, ps,
				 n, settings_list):
	''' Plot the output of our experiments '''
	# Plot the precision results
	for mean, std in zip(p_means, p_stds):
		plt.errorbar(ps, mean, std)
	plt.xlabel("Spy Fraction p")
	plt.ylabel("Precision")
	plt.legend(settings_list)
	plt.title("Deanonymization Precision (lower is better)")
	plt.savefig("results/precision")

	# Plot the recall results
	plt.figure()
	for mean, std in zip(r_means, r_stds):
		plt.errorbar(ps, mean, std)
	plt.xlabel("Spy Fraction p")
	plt.ylabel("Recall")
	plt.legend(settings_list)
	plt.title("Deanonymization Recall (lower is better)")
	plt.savefig("results/recall")

	# Plot precision vs. recall results
	plt.figure()
	for r_mean, p_mean in zip(r_means, p_means):
		plt.plot(r_mean, p_mean, 'o-')
	plt.xlabel("Recall")
	plt.ylabel("Precision")
	legend1 = plt.legend(settings_list)
	for r_mean, p_mean in zip(r_means, p_means):
		lmin = plt.scatter(r_mean[0], p_mean[0], s=100, facecolors = 'none', edgecolors = 'k')
		lmax = plt.scatter(r_mean[-1], p_mean[-1], s=100, marker = 's', facecolors = 'none', edgecolors = 'k')
	plt.legend([lmin, lmax], [f'Spy fraction p = {ps[0]}', f'Spy fraction p = {ps[-1]}'], loc=4)
	plt.gca().add_artist(legend1)
	plt.title("Precision vs. Recall (lower left is best)")
	plt.savefig("results/precision_vs_recall")
	if verbose:
		plt.show()
	

def run_sims(G, num_nodes, verbose, sim_type, sim_params):
	''' Run simulation according to the settings specified in sim_settings'''
	return sim_type(G, num_honest_nodes, verbose, **sim_params)

def generate_graph(n, p, d, verbose, graph_type, graph_params):
	''' Generate a graph with corresponding params
		Input params:
			n     	  number of nodes
			p     	  fraction of spies
			d 	  	  out-degree of random-regular anonymity graph
			verbose   print extra debug messages
			args  	  other args for different classes of graphs, including the type of graph'''
	return graph_type(n, p, d, verbose, **graph_params)


if __name__=='__main__':


	num_sims = len(sim_settings)

	# Initialize precision, recall lists
	p_means = [[] for i in range(num_sims)]
	p_stds  = [[] for i in range(num_sims)]
	r_means = [[] for i in range(num_sims)]
	r_stds  = [[] for i in range(num_sims)]

	# ----- Number of choices for each connection -----#
	if len(sys.argv) > 1:
		q = float(sys.argv[1])
	else:
		q = 0.0
	# print('q is', q)

	if len(sys.argv) > 2:
		semi_honest = bool(sys.argv[2])
	else:
		semi_honest = False
	print('semi_honest:', semi_honest)

	for d in ds:
		print('Graph out-degree d is ', d)

		for p in ps:
			print(f"\nFraction of spies p is {p}")

			# Collect the precision and recall per graph per trial here
			graph_precision = [0 for i in range(num_sims)]
			graph_recall = [0 for i in range(num_sims)]
			graph_precision_std = [0 for i in range(num_sims)]
			graph_recall_std = [0 for i in range(num_sims)]

			for i in range(graph_trials):
				if (i%5 == 0):
					print('Trial ', i, ' of ', graph_trials)

				# Generate the graph
				# gen = sim_graph[1](n, p, d, verbose)  # d-regular graph
				gen = generate_graph(n, p, d, verbose, sim_graph, sim_graph_params)

				# if semi_honest:
				# 	gen = QuasiRegGraphGen(n, p, BTC_GRAPH_OUT_DEGREE, verbose, d_anon = 2) # quasi d-regular w spies
				# else:
				# 	gen = QuasiRegGraphGenSpiesOutbound(n, p, d, verbose) # quasi d-regular w spies, no degree-checking, spies connect to all

				# gen = QuasiRegGraphGenSpies(n, p, k, d, verbose) # quasi d-regular with degree-checking, spies lie about degree
				# gen = DataGraphGen('data/bitcoin.gexf', p, verbose) # Bitcoin graph
				# gen = QuasiRegThreshGraphGen(n, p, d, k, verbose) # quasi d-regular
				# gen = CompleteGraphGen(n, p, verbose)  # complete graph
				G = gen.G
				A = gen.A
				# print 'G loaded', nx.number_of_nodes(G), ' nodes'

				num_honest_nodes = get_num_honest_nodes(G)

				# Corner cases
				if (num_honest_nodes == n) or (num_honest_nodes == 0):
					if (num_honest_nodes == n) or (num_honest_nodes == 0):
						graph_precision += 0.0
						graph_recall += 0.0
					continue
				# print G.edges()

				for j in range(path_trials):
					# run the simulations
					sims = []
					for sim_name, parameters in sim_settings.items():
						sims.append(run_sims(G, num_honest_nodes, verbose, parameters[0],
											 parameters[1]))

					for idx, sim in enumerate(sims):
						graph_precision[idx] += sim.precision
						graph_recall[idx] += sim.recall



				if verbose:
					# plot the graph
					plot_graph(G)

			for idx, sim in enumerate(sims):

				graph_precision[idx] = graph_precision[idx] / path_trials / graph_trials
				graph_precision_std[idx] = np.sqrt(graph_precision[idx] * (1.0-graph_precision[idx]) / graph_trials / path_trials)
				graph_recall[idx] = graph_recall[idx] / path_trials / graph_trials
				graph_recall_std[idx] = np.sqrt(graph_recall[idx] * (1-graph_recall[idx]) / graph_trials / path_trials)
				print('Graph precision: ', graph_precision[idx])
				print('Graph recall: ', graph_recall[idx])

				p_means[idx].append(graph_precision[idx])
				p_stds[idx].append(graph_precision_std[idx])

				r_means[idx].append(graph_recall[idx])
				r_stds[idx].append(graph_recall_std[idx])


	p_means = np.array(p_means)
	p_stds = np.array(p_stds)
	r_means = np.array(r_means)
	r_stds = np.array(r_stds)
	ps = np.array(ps)

	print('Total p_means', p_means)
	# print 'Total p_stds', p_stds

	print('Total r_means', r_means)
	# print 'Total r_stds', r_stds

	if verbose:
		print('Values of p', ps)


	settings_list = np.zeros((num_sims,), dtype=np.object)
	settings_list[:] = [item for item in sim_settings.keys()]
	print(settings_list)
	if not os.path.exists('results'):
		os.makedirs('results')
	scipy.io.savemat('results/sim_data.mat',{'p_means':np.array(p_means),
										  'r_means':np.array(r_means),
										  'p_stds':np.array(p_stds),
										  'r_stds':np.array(r_stds),
										  'ps':np.array(ps),
										  'num_nodes':n,
										  'graph_type':sim_graph.__name__,
										  'sim_settings':settings_list})
	
	plot_results(p_means, p_stds, r_means, r_stds, ps, n, settings_list)