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
import orig_sim_lib as sim_lib
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
	num_sim = 6

	path_trials = 30
	# path_trials = 10

	# use_main = True		# Use the main P2P graph (true) or the anonymity graph (false)

	p_means = []
	p_stds = []
	r_means = []
	r_stds = []
	paths = []
	mean_pathlength = []
	for i in range(num_sim):
		p_means.append([])
		p_stds.append([])
		r_means.append([])
		r_stds.append([])
		paths.append([])
		mean_pathlength.append([])

	# ----- Out-degree of graph ----#
	# ds = [1,2,3]
	# ds = [2]
	ds = [2]

	# ----- Fraction of spies ----#
	# ps = [0.5, 0.7]
	# ps = np.arange(0.1,0.51,0.1)
	# ps = [0.02, 0.04, 0.08,  0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5]
	ps = [0.05, 0.2, 0.5]	

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
		for p in ps:
			print 'p is', p
			graph_precision = np.zeros(num_sim)
			graph_recall = np.zeros(num_sim)
			std_precision = np.zeros(num_sim)
			std_recall = np.zeros(num_sim)
			path_length = np.zeros(num_sim)
			hops = np.zeros((num_sim, 2*n))
			wards = np.zeros((num_sim, 2*n))

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
				sim = [1]*num_sim
				for j in range(path_trials):
					# run a simulation
					
					sim[0] = sim_lib.FirstSpyLineSimulator(G, num_honest_nodes, verbose, p_and_r = True, edgebased=0)#random
					sim[1] = sim_lib.FirstSpyLineSimulator(G, num_honest_nodes, verbose, p_and_r = True, edgebased=1)#intelligent incoming-edge based(no leaves)
					sim[2] = sim_lib.FirstSpyLineSimulator(G, num_honest_nodes, verbose, p_and_r = True, edgebased=2)#all sent out to one successor
					sim[3] = sim_lib.FirstSpyLineSimulator(G, num_honest_nodes, verbose, p_and_r = True, edgebased=3)#different incoming to different outgoing
					sim[4] = sim_lib.FirstSpyDiffusionSimulator(G, num_honest_nodes, verbose) #diffusion
					sim[5] = sim_lib.FirstSpyLineSimulator(G, num_honest_nodes, verbose, p_and_r = True, edgebased=4)#incoming edge based (possible leaves)
					
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
					for k in range(num_sim):
						# print(k)
						graph_precision[k] += sim[k].precision
						graph_recall[k] += sim[k].recall
						# hops[k] += sim[k].hops
						# wards[k] +=sim[k].wards
						# path_length [k] +=sim[k].mean_pathlength
						# print 'precision', sim.precision
				# graph_precisions.append(graph_precision)
				# graph_recalls.append(graph_recall)


				if verbose:
					# plot the graph
					plot_graph(G)


			# f, axes = plt.subplots(num_sim, sharex=True, sharey=True)
			# axes[0].set_title('100 nodes, p=' + str(p))
			# axes[3].set_ylabel('Number of nodes (frequency)')
			# axes[-1].set_xlabel('Path length')

			# f2, axes2 = plt.subplots(num_sim, sharex=True, sharey=True)
			# axes2[0].set_title('100 nodes, p=' + str(p))
			# axes2[3].set_ylabel('Number of nodes (frequency)')
			# axes2[-1].set_xlabel('Ward Size')

			for i in range(num_sim):
				graph_precision[i] = graph_precision[i] / path_trials / graph_trials
				std_precision[i] = np.sqrt(graph_precision[i] * (1.0-graph_precision[i]) / graph_trials / path_trials)
				graph_recall[i] = graph_recall[i] / path_trials / graph_trials
				std_recall[i] = np.sqrt(graph_recall[i] * (1-graph_recall[i]) / graph_trials / path_trials)
				p_means[i].append(graph_precision[i])
				p_stds[i].append(std_precision[i])
				r_means[i].append(graph_recall[i])
				r_stds[i].append(std_recall[i])

				# path_length[i] = path_length[i]/path_trials/graph_trials
				# # paths[i].append(path_length[i])
				# hops[i] = hops[i] / path_trials / graph_trials
				# wards[i] = wards[i] / path_trials / graph_trials
				# axes[i].bar(np.arange(0,int(2*n), 1), hops[i])
				# axes[i].set_title('sim='+ str(i))
				# axes2[i].bar(np.arange(0,int(2*n), 1), wards[i])
				# axes2[i].set_title('sim='+ str(i))
				# mean_pathlength[i].append(np.sum(hops[i]*np.arange(0,int(2*n)))/num_honest_nodes)

				# plt.title('Hops before first spy (500 nodes) - sim='+ str(i)+ ' p=' + str(p))
				# plt.ylabel('Number of source nodes (frequency)')
				# plt.xlabel('Path length')
				# plt.savefig('hops_distri, p=' +str(p) +' sim='+str(i)+'.jpg')

				# plt.show()

			# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
			# plt.setp([a.get_xticklabels() for a in f2.axes[:-1]], visible=False)
			# f.savefig('hops, p=' +str(p)+'.jpg')
			# f2.savefig('wards, p=' +str(p)+'.jpg')
			# if semi_honest:
			# 	filename = 'results/spy_out_degree/quasi_regular_d_2_max_weight_q_' + str(q).replace('.','_') + '_spies_behave.mat'
			# else:	
			# 	filename = 'results/spy_out_degree/quasi_regular_d_2_max_weight_q_' + str(q).replace('.','_') + '_spies_misbehave.mat'
			# # scipy.io.savemat(filename, {'ds' : np.array(ds), 'ps': np.array(ps), 'n' : n, 'graph_trials': graph_trials, 
			# # 							'path_trials': path_trials, 'q': q,
			# # 							'p_means' : np.array(p_means), 'p_stds' : np.array(p_stds),
			# # 							'r_means': np.array(r_means), 'r_stds' : np.array(r_stds)})

	plt.figure()
	plt.errorbar(ps, np.log(p_means[0]), yerr = [np.divide(np.multiply(0.434,p_stds[0]),p_means[0]),np.divide(np.multiply(0.434,p_stds[0]),p_means[0])],fmt = '--x' ,  label = 'Per-transaction')
	plt.errorbar(ps, np.log(p_means[1]), yerr = [np.divide(np.multiply(0.434,p_stds[1]),p_means[1]), np.divide(np.multiply(0.434,p_stds[1]),p_means[1])], fmt = '--o', label = 'Incoming edge based')
	plt.errorbar(ps, np.log(p_means[2]), yerr = [np.divide(np.multiply(0.434,p_stds[2]),p_means[2]), np.divide(np.multiply(0.434,p_stds[2]),p_means[2])], fmt = '--^', label = 'All to one (Coupled)')
	plt.errorbar(ps, np.log(p_means[3]), yerr = [np.divide(np.multiply(0.434,p_stds[3]),p_means[3]), np.divide(np.multiply(0.434,p_stds[3]),p_means[3])], fmt = '--s', label = 'One to one (Decoupled)')
	plt.errorbar(ps, np.log(p_means[4]), yerr = [np.divide(np.multiply(0.434,p_stds[4]),p_means[4]), np.divide(np.multiply(0.434,p_stds[4]),p_means[4])], fmt = '--o', label = 'Diffusion')	
	plt.errorbar(ps, np.log(p_means[5]), yerr = [np.divide(np.multiply(0.434,p_stds[5]),p_means[5]),np.divide(np.multiply(0.434,p_stds[5]),p_means[5])],fmt = '--H' ,  label = 'Incoming edge (with leaves)')
	# for i in range(num_sim):
	# 	plt.plot(ps, mean_pathlength[i], marker = i+1, label = 'simulation - '+str(i))
	plt.plot(ps,np.log(ps), label='p')
	plt.title('Precision on 4-regular graph of 100 (handling cycles)')
	plt.ylabel('Precision')
	plt.xlabel('Fraction of spies, p')
	plt.legend(loc = 0)
	plt.savefig('trial.jpg')
	plt.show()
	plt.close()	