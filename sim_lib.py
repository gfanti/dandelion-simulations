# simulation lib
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import collections
import math


MAIN_GRAPH = 0
ANON_GRAPH = 1

class SpyInfo:
	def __init__(self, spy_id, exit_node, source):
		self.id = spy_id
		self.exit_node = exit_node
		self.source = source


class Simulator(object):
	def __init__(self, G, num_honest_nodes, verbose = False):
		'''	G -- graph over which to spread
		'''
	
		self.G = G
		self.num_honest_nodes = num_honest_nodes
		self.verbose = verbose

class LineSimulator(Simulator):
	def __init__(self, G, verbose = False):
		super(LineSimulator, self).__init__(G, verbose)

	def run_simulation(self, graph = MAIN_GRAPH):
		''' Simulates dandelion spreading over a graph. 
		Parameters:
			graph 	Which graph to spread over. 
					MAIN_GRAPH  =  regular P2P graph
					ANON_GRAPH 	=  anonymity graph 
		'''

		# Run a line simulation over G
		# List of all spies
		spies = nx.get_node_attributes(self.G,'spy')
		spy_mapping = {}

		if (graph == MAIN_GRAPH):
			G = self.G

		# Make a dict of spies
		for node in self.G.nodes():
			# if node is a spy, add it to the dictionary
			if spies[node]:
				spy_mapping[node] = []

		# find the nodes reporting to each spy
		for node in self.G.nodes():
			if spies[node]:
				continue	
			spy_found = False
			pre_tail = node
			tail = node
			path_length = 0
			while True:
				neighbors = self.G.successors(tail)
				pre_tail = tail
				tail = random.choice(neighbors)
				path_length += 1
				# print 'node', node, 'neighbors', neighbors, 'tail', tail
				if spies[tail]:
					spy_mapping[tail].append(SpyInfo(tail, pre_tail, node))
					break
				if path_length > nx.number_of_nodes(self.G):
					# there are no spies on this path, so we'll just assign the 
					#   last node to see the message to a spy
					spy = random.choice(spy_mapping.keys())
					spy_mapping[spy].append(SpyInfo(spy, tail, node))
					break
			if self.verbose:
				print 'Node ', node, ' traversed ', path_length, 'hops'

		return spy_mapping

		

class MaxWeightEstimator(object):
	def __init__(self, G, honest_nodes, weights, verbose = False, p_and_r = False):
		self.G = G
		self.honest_nodes = honest_nodes
		self.weights = weights
		self.verbose = verbose
		self.p_and_r = p_and_r # whether to compute precision AND recall or just precision
		
	def compute_estimate(self):
		''' 
			Computes the source estimates based on the observed spy info. 
		'''
		# First, map the nodes to an ordering from 0 to num_honest_nodes
		num_honest_nodes = len(self.honest_nodes)
		honest_node_mapping = {}
		inv_honest_node_mapping = {}
		cnt = 0
		for node in self.honest_nodes:
			honest_node_mapping[node] = cnt
			inv_honest_node_mapping[cnt] = node
			cnt += 1
		honest_node_indices = honest_node_mapping.values()
		# print 'honest_node_mapping', honest_node_mapping

		# Next, create a random mapping, to randomize the labeling of nodes,
		# since networkx max-weight algorithm doesn't seem to randomize
		mapping = list(np.random.permutation(honest_node_indices))
		msg_mapping = [i+1 for i in np.random.permutation(honest_node_indices)]
		# print 'mapping', mapping
		# print 'message mapping', msg_mapping

		# for msg, lik in self.weights.iteritems():
		# 	print 'msg ', msg, ' has likelihoods ', lik
		inv_mapping = [mapping.index(i) for i in honest_node_indices]

		
		honest_nodes = honest_node_indices # the label of an honest nodes represents its order in mapping
		messages = [-n-1 for n in honest_nodes]

		H = nx.Graph()
		H.add_nodes_from(honest_nodes)	# servers
		H.add_nodes_from(messages) 		# messages
		for msg in self.honest_nodes: 	# for each real message tag
			for src, likelihood in self.weights[msg].iteritems():	# and for each candidate source (+likelihood)
				# add an edge from the relabeled source to the relabeled message with the appropriate weight

				H.add_edge(mapping[honest_node_mapping[src]], -msg_mapping[honest_node_mapping[msg]], weight = likelihood)

		# print nx.adjacency_matrix(H).todense()
		# print 'nodes', H.nodes()
		# left, right = nx.bipartite.sets(H)
		# print 'left, right', left, right
		matching = nx.max_weight_matching(H, maxcardinality = True)
		# print 'matching', matching
		estimate = []
		for a,b in matching.iteritems():
			if a < 0 and b >= 0:
				item = [inv_honest_node_mapping[msg_mapping.index(-a)], inv_honest_node_mapping[mapping.index(b)]]
			elif b < 0 and a >= 0:
				item = [inv_honest_node_mapping[msg_mapping.index(-b)], inv_honest_node_mapping[mapping.index(a)]]
			else:
				continue
			if item in estimate:
				continue
			estimate += [item]

		estimate.sort(key=lambda x: x[0])
		
		# print 'estimate', estimate

		return estimate

	def compute_payout(self, estimate):
		''' 
			Computes the precision from the max-weight matching from before.
		'''

		precision = 0.0
		for item in estimate:
			msg, src = item[0], item[1]
			if msg == src:
				precision += 1.0

		precision = precision / len(self.honest_nodes)
		recall = precision # because max weight outputs a matching, which has same precision and recall

		return precision, recall

class MaxWeightVCEstimator(MaxWeightEstimator):
	def __init__(self, G, honest_nodes, honest_dandelions, weights, verbose = False, p_and_r = False):
		super(MaxWeightVCSimulator, self).__init__(G, honest_nodes, weights, verbose, p_and_r)
		self.honest_dandelions = honest_dandelions
		
	def compute_estimate(self):
		''' 
			Computes the source estimates based on the observed spy info. 
		'''
		# First, map the nodes to an ordering from 0 to num_honest_nodes
		num_honest_nodes = len(self.honest_nodes)
		honest_node_mapping = {}
		inv_honest_node_mapping = {}
		cnt = 0
		for node in self.honest_nodes:
			honest_node_mapping[node] = cnt
			inv_honest_node_mapping[cnt] = node
			cnt += 1
		honest_node_indices = honest_node_mapping.values()
		# print 'honest_node_mapping', honest_node_mapping

		# Next, create a random mapping, to randomize the labeling of nodes,
		# since networkx max-weight algorithm doesn't seem to randomize
		mapping = list(np.random.permutation(honest_node_indices))
		msg_mapping = [i+1 for i in np.random.permutation(honest_node_indices)]
		# print 'mapping', mapping
		# print 'message mapping', msg_mapping

		# for msg, lik in self.weights.iteritems():
		# 	print 'msg ', msg, ' has likelihoods ', lik
		inv_mapping = [mapping.index(i) for i in honest_node_indices]

		
		honest_nodes = honest_node_indices # the label of an honest nodes represents its order in mapping
		messages = [-n-1 for n in honest_nodes]

		H = nx.Graph()
		H.add_nodes_from(honest_nodes)	# servers
		H.add_nodes_from(messages) 		# messages
		for msg in self.honest_nodes: 	# for each real message tag
			for src, likelihood in self.weights[msg].iteritems():	# and for each candidate source (+likelihood)
				# add an edge from the relabeled source to the relabeled message with the appropriate weight

				H.add_edge(mapping[honest_node_mapping[src]], -msg_mapping[honest_node_mapping[msg]], weight = likelihood)

		# print nx.adjacency_matrix(H).todense()
		# print 'nodes', H.nodes()
		# left, right = nx.bipartite.sets(H)
		# print 'left, right', left, right
		matching = nx.max_weight_matching(H, maxcardinality = True)
		# print 'matching', matching
		estimate = []
		for a,b in matching.iteritems():
			if a < 0 and b >= 0:
				item = [inv_honest_node_mapping[msg_mapping.index(-a)], inv_honest_node_mapping[mapping.index(b)]]
			elif b < 0 and a >= 0:
				item = [inv_honest_node_mapping[msg_mapping.index(-b)], inv_honest_node_mapping[mapping.index(a)]]
			else:
				continue
			if item in estimate:
				continue
			estimate += [item]

		estimate.sort(key=lambda x: x[0])
		
		# print 'estimate', estimate

		return estimate

class MaxWeightLineSimulator(LineSimulator):
	def __init__(self, G, num_honest_nodes, verbose = False, p_and_r = True):
		super(MaxWeightLineSimulator, self).__init__(G, verbose)
		self.weights = None
		self.num_honest_nodes = num_honest_nodes
		self.p_and_r = p_and_r

		honest_nodes = [node for node in self.G if not self.G.node[node]['spy']]

		# Run the simulation
		spy_mapping = super(MaxWeightLineSimulator, self).run_simulation()
		# COmpute the weights
		self.compute_weights(spy_mapping)
		# Compute a max-weight matching
		est = MaxWeightEstimator(G, honest_nodes, self.weights, verbose, p_and_r)
		src_estimate = est.compute_estimate()
		# Compute the precision
		if p_and_r:
			self.precision, self.recall = est.compute_payout(src_estimate)
		else:
			self.precision = est.compute_payout(src_estimate)

		
	def compute_weights(self, spy_mapping):
		'''
			Writes a matrix of likelihood weights to self.weights based
			on the likelihood of each node to have generated the observed
			spy information.
		'''
		precision = 0
		recall = 0

		exits = {}
		self.weights = {}
		
		# first, map the spy info to wards: (exit, messages) tuples
		for spy, info in spy_mapping.iteritems():
			
			for item in info: # for each message that exited to spy
				if item.exit_node not in exits:
					exits[item.exit_node] = [item.source]
				else:
					exits[item.exit_node].append(item.source)
		''' for each ward, compute the local neighborhood and the
			likelihood of each node in that neighborhood to end up
			at the given exit node. The likelihoods are equal for 
			all nodes at a given exit node.
		'''
		for exit, ward in exits.iteritems():
			likelihoods = self.compute_likelihoods(exit)
			for msg in ward:
				self.weights[msg] = likelihoods

		# print 'exit', exit
		# print 'ward', ward
		# print 'likelihoods', likelihoods


	def compute_likelihoods(self, exit):
		p = float(self.num_honest_nodes) / self.G.number_of_nodes()
		local_tree_depth = math.floor(1.5 / p)

		spies = [node for node in self.G.nodes() if self.G.node[node]['spy']]

		likelihoods = {}
		shortest_paths = nx.shortest_path(self.G, target = exit)
		for source, path in shortest_paths.iteritems():
			if source in spies:
				continue

			# if the shortest path between node and exit has a spy, assign it 0 likelihood
			# this is an approximation, since a message could have taken a longer path
			if not any([n in spies for n in path[:-1]]):
				degrees = [1.0 / self.G.out_degree(node) for node in path]
				likelihoods[source] = np.prod(degrees)


		return likelihoods

class FirstSpyEstimator(object):
	def __init__(self, G, num_honest_nodes, verbose = False, p_and_r = False):
		self.G = G
		self.num_honest_nodes = num_honest_nodes
		self.verbose = verbose
		self.p_and_r = p_and_r # whether to compute precision AND recall or just precision
	
	def compute_payout(self, spy_mapping):
		precision = 0
		recall = 0
		if self.verbose:
			print '\n'

		exits = {}
		for spy, info in spy_mapping.iteritems():
			if self.verbose:
				print 'spy', spy
				exit_list = [item.exit_node for item in info]
				sources = [item.source for item in info]
				for exit in set(exit_list):
					print 'exit', exit
					print 'sources', [sources[i] for i in range(len(exit_list)) if exit_list[i] == exit]
			
			for item in info:
				# if verbose:
				# 	print 'exit node', item.exit_node
				# 	print 'source', item.source
				if item.exit_node not in exits:
					exits[item.exit_node] = [item.source]
				else:
					exits[item.exit_node].append(item.source)

		for exit, ward in exits.iteritems():
			if exit in ward:
				precision += 1.0 / len(ward)
				recall += 1.0
		if self.verbose:
			print 'num_honest_nodes', self.num_honest_nodes
			print 'total precision', precision
		precision = precision / self.num_honest_nodes
		recall = recall / self.num_honest_nodes
		
		if self.p_and_r:
			return precision, recall
		else:
			return precision


class FirstSpyLineSimulator(LineSimulator):
	def __init__(self, G, num_honest_nodes, verbose = False, p_and_r = False):
		super(FirstSpyLineSimulator, self).__init__(G, verbose)
		self.p_and_r = p_and_r
		self.num_honest_nodes = num_honest_nodes

		spy_mapping = super(FirstSpyLineSimulator, self).run_simulation()
		est = FirstSpyEstimator(self.G, self.num_honest_nodes, self.verbose, p_and_r)
		if p_and_r:
			self.precision, self.recall = est.compute_payout(spy_mapping)
		else:
			self.precision = est.compute_payout(spy_mapping)

class DiffusionSimulator(Simulator):
	def __init__(self, G, num_honest_nodes, verbose = False):
		super(DiffusionSimulator, self).__init__(G, num_honest_nodes, verbose)
		
	def run_simulation(self):
		'''
			Runs a diffusion process and keeps track of the spy that first sees
			each message, as well as the node who delivered the message to the
			spy. The spy observations are returned.
		'''

		# List of all spies
		spies = nx.get_node_attributes(self.G,'spy')

		spy_mapping = {}

		# Make a dict of spies
		for node in self.G.nodes():
			# if node is a spy, add it to the dictionary
			if spies[node]:
				spy_mapping[node] = []

		# find the nodes reporting to each spy
		for node in self.G.nodes():
			if spies[node]:
				continue	
			spy_found = False

			infected = [node]
			boundary_edges = self.get_neighbor_set(infected, infected)
			path_length = 0
			while boundary_edges:

				next = random.choice(boundary_edges)
				source = next[0]
				target = next[1]
				path_length += 1
				# print 'node', node, 'neighbors', neighbors, 'tail', tail
				if spies[target]:
					spy_mapping[target].append(SpyInfo(target, source, node))
					break
				if path_length >= nx.number_of_nodes(self.G):
					# there are no spies on this path, so we'll just assign the 
					#   last node to see the message to a spy

					spy = random.choice(spy_mapping.keys())
					spy_mapping[spy].append(SpyInfo(spy, target, node))
					break
				infected += [target]
				boundary_edges.remove(next)
				boundary_edges = [item for item in boundary_edges if item[1] != target]
				boundary_edges += [[target, item] for item in self.G.neighbors(target) if item not in infected]

				# add next to boundary, remove the infecting node if it is eclipsed
			if self.verbose:
				print 'Node ', node, ' traversed ', path_length, 'hops'

		return spy_mapping

	def get_neighbor_set(self, candidates, infected):
		''' Returns the set of susceptible edges from infected zone to uninfected '''
		neighbors = []
		for node in candidates:
			for neighbor in self.G.neighbors(node):
				if neighbor not in infected:
					neighbors += [[node, neighbor]]

		return neighbors

class FirstSpyDiffusionSimulator(DiffusionSimulator):
	def __init__(self, G, num_honest_nodes, verbose = False, p_and_r = True):
		super(FirstSpyDiffusionSimulator, self).__init__(G, num_honest_nodes, verbose)
		self.p_and_r = p_and_r

		spy_mapping = super(FirstSpyDiffusionSimulator, self).run_simulation()
		est = FirstSpyEstimator(G, num_honest_nodes, verbose, p_and_r)
		if p_and_r:
			self.precision, self.recall = est.compute_payout(spy_mapping)
		else:
			self.precision = est.compute_payout(spy_mapping)
				


	


	