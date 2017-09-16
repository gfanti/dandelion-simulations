''' 
	This script tests Dandelion spreading on random regular graphs.
	There are several variants, including quasi-regular constructions.
'''

import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import collections
import math
import scipy.io
from graph_lib import *
import sim_lib
import sys

BTC_GRAPH_OUT_DEGREE = 8
DIFFUSION = 2



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

	n = 100	# number of nodes
	# d = 2	 # outdegree of graph
	# p = 0.2 # probability of spiesx
	verbose = False	# debug?

	# graph_trials = 50
	# graph_trials = 150
	graph_trials = 70
	# graph_trials = 10

	path_trials = 30
	# path_trials = 10

	# use_main = True		# Use the main P2P graph (true) or the anonymity graph (false)


	p_means, p_stds, p_means2, p_stds2, p_means3, p_stds3 = [], [], [], [], [], []
	r_means, r_stds, r_means2 , r_stds2, r_means3, r_stds3= [], [], [], [], [], []

	# ----- Out-degree of graph ----#
	# ds = [1,2,3]
	# ds = [2]
	ds = [2]

	# ----- Fraction of spies ----#
	# ps = [0.2, 0.7]
	# ps = np.arange(0.1,0.51,0.1)
	ps = [0.02, 0.04, 0.08,  0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5]
	# ps = [0.4, 0.5]

	# ----- Number of choices for each connection -----#
	# k = 4
	# qs = [0.0, 0.1, 0.2]
	if len(sys.argv) > 1:
		q = float(sys.argv[1])
	else:
		q = 0.0
	print 'q is', q

	if len(sys.argv) > 2:
		semi_honest = bool(sys.argv[2])
	else:
		semi_honest = False
	print 'semi_honest:', semi_honest

	for d in ds:
		print 'd is ', d
		# Initialize precision and recall
		graph_precision = 0
		graph_recall = 0

		for p in ps:
			print 'p is', p
			graph_precision = 0
			graph_recall = 0
			graph_precision2 = 0
			graph_recall2 = 0
			graph_precision3 = 0
			graph_recall3 = 0
			for i in range(graph_trials):
				if (i%5 == 0):
					print 'Trial ', i, ' of ', graph_trials

				# Generate the graph
				gen = RegGraphGen(n, p, d, verbose)  # d-regular

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

					# run a simulation
					
					sim1 = sim_lib.FirstSpyLineSimulator(G, num_honest_nodes, verbose, p_and_r = True, edgebased=False)
					sim2 = sim_lib.FirstSpyLineSimulator(G, num_honest_nodes, verbose, p_and_r = True, edgebased=True)
					sim3 = sim_lib.FirstSpyDiffusionSimulator(G, num_honest_nodes, verbose)
					# if semi_honest:
					
					# ===== Uncomment this ===== #
					# if q == DIFFUSION:
					# 	sim = sim_lib.FirstSpyDiffusionSimulator(G, num_honest_nodes, verbose)
					# else:
					# 	sim = sim_lib.MaxWeightLineSimulatorUnknownTerminus(A, G, num_honest_nodes, verbose=verbose, q=q)
					# ========================== #

					# else:
					# 	sim = sim_lib.MaxWeightLineSimulator(A, num_honest_nodes, verbose=verbose, q=q)

					# retrieve the precision
					graph_precision += sim1.precision
					graph_recall += sim1.recall
					graph_precision2 += sim2.precision
					graph_recall2 += sim2.recall
					graph_precision3 += sim3.precision
					graph_recall3 += sim3.recall
					# print 'precision', sim.precision

				

				# graph_precisions.append(graph_precision)
				# graph_recalls.append(graph_recall)


				if verbose:
					# plot the graph
					plot_graph(G)

			graph_precision = graph_precision / path_trials / graph_trials
			std_precision = np.sqrt(graph_precision * (1.0-graph_precision) / graph_trials / path_trials)
			graph_recall = graph_recall / path_trials / graph_trials
			std_recall = np.sqrt(graph_recall * (1-graph_recall) / graph_trials / path_trials)
			print 'Graph precision: ', graph_precision
			print 'Graph recall: ', graph_recall
			
			p_means.append(graph_precision)
			p_stds.append(std_precision)

			r_means.append(graph_recall)
			r_stds.append(std_recall)

			graph_precision2 = graph_precision2 / path_trials / graph_trials
			std_precision2 = np.sqrt(graph_precision2 * (1.0-graph_precision2) / graph_trials / path_trials)

			graph_recall2 = graph_recall2 / path_trials / graph_trials
			# print 'Graph precision: ', graph_precision
			# print 'Graph recall: ', graph_recall
			
			p_means2.append(graph_precision2)
			p_stds2.append(std_precision2)

			r_means2.append(graph_recall2)
			# r_stds.append(std_recall)

			graph_precision3 = graph_precision3 / path_trials / graph_trials
			std_precision3 = np.sqrt(graph_precision3 * (1-graph_precision3) / graph_trials / path_trials)

			graph_recall3 = graph_recall3 / path_trials / graph_trials
			# print 'Graph precision: ', graph_precision
			# print 'Graph recall: ', graph_recall
			
			p_means3.append(graph_precision3)
			p_stds3.append(std_precision3)

			r_means3.append(graph_recall3)
			


	
			# if semi_honest:
			# 	filename = 'results/spy_out_degree/quasi_regular_d_2_max_weight_q_' + str(q).replace('.','_') + '_spies_behave.mat'
			# else:	
			# 	filename = 'results/spy_out_degree/quasi_regular_d_2_max_weight_q_' + str(q).replace('.','_') + '_spies_misbehave.mat'
			# # scipy.io.savemat(filename, {'ds' : np.array(ds), 'ps': np.array(ps), 'n' : n, 'graph_trials': graph_trials, 
			# # 							'path_trials': path_trials, 'q': q,
			# # 							'p_means' : np.array(p_means), 'p_stds' : np.array(p_stds),
			# # 							'r_means': np.array(r_means), 'r_stds' : np.array(r_stds)})

	print 'Total p_means', np.array(p_means)
	print 'Total p_stds', np.array(p_stds)

	# print 'Total r_means', np.array(r_means)
	# print 'Total r_stds', np.array(r_stds)

	# print 'saved to file', filename
	print(np.log(p_stds))
	print('...........')
	print(np.log(p_means))
	plt.figure()
	plt.yscale('log')
	plt.errorbar(ps, p_means, yerr = [p_stds,p_stds],fmt = '-x' ,  label = 'Per-transaction forwarding')
	plt.errorbar(ps, p_means2, yerr = [p_stds2,p_stds2], fmt = '-o', label = 'Per-incoming-edge forwarding')
	plt.errorbar(ps, p_means3, yerr = [p_stds3, p_stds3], fmt = '-^', label = 'Diffusion')
	# plt.plot(ps, np.log(p_means),  '-o', label = 'Random forwarding')
	# plt.plot(ps, np.log(p_means2), '-o', label = 'Incoming edge based forwarding')
	# plt.plot(ps, np.log(p_means3), '-o', label = 'Diffusion')
	# plt.errorbar(ps, np.log(ps), yerr = 1)
	plt.plot(ps,ps, label='p')
	# plt.plot(ps, np.log(np.square(ps)), label='log(p*p)')
	# plt.plot(ps, np.log(np.multiply(np.square(ps),np.log(np.divide(1.0,ps)))), label = 'log(p*p*log(1/p))')
	plt.title('First spy precision on 4-regular graph (Dandelion++)')
	plt.ylabel('log(Precision)')
	plt.xlabel('Fraction of spies, p')
	plt.legend(loc = 0)
	plt.savefig('plot4.jpg')

	# plt.plot(ps, r_means)
	plt.show()
	plt.close()	
