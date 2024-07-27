# simulation lib
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import collections
import math

incedge = False
MAIN_GRAPH = 0
ANON_GRAPH = 1

class SpyInfo:
	def __init__(self, spy_id, exit_node, source, relayed_to_spy = True):
		self.id = spy_id
		self.exit_node = exit_node
		self.source = source
		self.relayed_to_spy = relayed_to_spy



class Simulator(object):
	def __init__(self, A, num_honest_nodes, verbose = False):
		'''	A -- graph over which to spread
		'''
	
		self.A = A
		self.num_honest_nodes = num_honest_nodes
		self.verbose = verbose

class LineSimulator(Simulator):
	def __init__(self, A, verbose = False, q = 0.0, per_tx_q = True):
		super(LineSimulator, self).__init__(A, verbose)
		self.q = q
		self.per_tx_q = per_tx_q

	def run_simulation(self, graph = MAIN_GRAPH):
		''' Simulates dandelion spreading over a graph. 
		Parameters:
			graph 	Which graph to spread over. 
					MAIN_GRAPH  =  regular P2P graph
					ANON_GRAPH 	=  anonymity graph 
		'''

		# Assign the diffusers to the graph
		A = assign_diffusers(self.A, self.q, nx.number_of_nodes(self.A))
		# Run a line simulation over A
		# List of all spies
		spies = nx.get_node_attributes(A,'spy')
		if self.per_tx_q == False:
			diffusers = nx.get_node_attributes(A, 'diffuser')
		spy_mapping = {}
		pred_succ_mapping = {}							#to store mappings predecessor ->node ->successor
		if (graph == MAIN_GRAPH):
			A = self.A
		 
		# Compute the forwarding rules for the graph (which inbound edge maps to which outbound edge)
		for node in A.nodes():
			# store the node's predecessors and successors
			preds = A.predecessors(node)	
			succs = A.successors(node)
			# if node is a spy, add it to the dictionary
			if spies[node]:
				spy_mapping[node] = []
			elif incedge==2: # all-to-one
				pred_succ_mapping[node] = random.choice(succs)
			elif incedge==1: # per-incoming-edge
				pred_succ_mapping[node] = {}		#dict to hold mapping for this node					
				for pred in preds:					#for each pred, we randomly allocate a succ
					pred_succ_mapping[node][pred] = random.choice(succs)
				if len(list(pred_succ_mapping[node].values()))>1:				
					if (list(pred_succ_mapping[node].values()))[0]==(list(pred_succ_mapping[node].values()))[1]:
						pred_succ_mapping[node][node] = (list(pred_succ_mapping[node].values()))[0]
					else:
						pred_succ_mapping[node][node] = random.choice(succs)
				else:		
					pred_succ_mapping[node][node] = list(pred_succ_mapping[node].values())[0]			
			elif incedge==3: # one-to-one
				pred_succ_mapping[node] = {}		#dict to hold mapping for this node					
				
				if len(succs) > 0:
					# map your own transaction randomly to the out-nodes
					pred_succ_mapping[node][node] = random.choice(succs)
					
				# compute the one-to-one mapping
				if len(preds)>0 and len(succs)>0:
					succ_list = [item for item in succs]
					random.shuffle(succ_list)
					for pred in preds:
						succ = succ_list.pop()
						pred_succ_mapping[node][pred] = succ
						if not succ_list:
							succ_list = [item for item in succs]
							random.shuffle(succ_list)
			# elif incedge == 4: #
			# 	pred_succ_mapping[node] = {}
			# 	for pred in preds:
			# 		pred_succ_mapping[node][pred] = random.choice(succs)
			# 	pred_succ_mapping[node][node] = random.choice(succs)

		# find the nodes reporting to each spy
		hops = np.zeros(2*len(A.nodes()))

		# Propagate one transaction per node in the graph
		for node in A.nodes():
			# print("starting node ", node)
			if spies[node]:
				continue	
			spy_found = False
			pre_tail = node
			tail = node
			path_length = 0
			path_list = []
			# print("\n")
			while True:
				if tail in path_list and incedge!=0:
					neighbors = np.array(A.successors(tail))
					if (incedge==1 or incedge==3 or incedge==4):
						if (len(neighbors)>1 and tail in pred_succ_mapping):
							if pre_tail in pred_succ_mapping[tail]:
								now = neighbors[np.where(neighbors!=pred_succ_mapping[tail][pre_tail])[0][0]]
						else:
							now = pred_succ_mapping[tail][pre_tail]	
					else :
						if (len(neighbors)>1):
							now = neighbors[np.where(neighbors!=pred_succ_mapping[tail])[0][0]]
						else:
							now = pred_succ_mapping[tail]		
				elif incedge==1 or incedge==3 or incedge==4:
					if tail in pred_succ_mapping:
						if pre_tail in pred_succ_mapping[tail]:
							now = pred_succ_mapping[tail][pre_tail]		
				elif incedge==2:
					now = pred_succ_mapping[tail]
				elif incedge==0:
					neighbors = list(A.successors(tail))
					now = random.choice(neighbors)
				pre_tail = tail
				path_list.append(tail)
				tail = now
				path_length += 1
				if spies[tail]:
					spy_mapping[tail].append(SpyInfo(tail, pre_tail, node))
					break
				if (self.per_tx_q and np.random.binomial(1, self.q)) or \
					(not self.per_tx_q and diffusers[tail] == True):
					# end the stem
					if tail in spy_mapping:
						spy_mapping[tail].append(SpyInfo(tail, pre_tail, node))
					else:
						spy_mapping[tail] = [SpyInfo(tail, pre_tail, node)]
					break
				if path_length > nx.number_of_nodes(A):
				# if tail in 
					# there are no spies on this path, so we'll just assign the 
					#   last node to see the message to a spy
					spy = random.choice(list(spy_mapping.keys()))
					spy_mapping[spy].append(SpyInfo(spy, tail, node))
					break
			if self.verbose:
				print('Node ', node, ' traversed ', path_length, 'hops')
			hops[path_length] +=1

		return spy_mapping, hops

			
		



class FirstSpyLineSimulator(LineSimulator):
	def __init__(self, A, num_honest_nodes, verbose = False, p_and_r = False, edgebased=0, q=0.0, per_tx_q=True):
		super(FirstSpyLineSimulator, self).__init__(A, verbose, q, per_tx_q)
		self.p_and_r = p_and_r
		self.num_honest_nodes = num_honest_nodes
		global incedge
		incedge=edgebased		#global variable holds the type for random forwarding or pseudorandom/random

		spy_mapping, hops = super(FirstSpyLineSimulator, self).run_simulation()
		self.hops = hops
		est = FirstSpyEstimator(self.A, self.num_honest_nodes, self.verbose, p_and_r)
		if p_and_r:
			self.precision, self.recall, self.wards = est.compute_payout(spy_mapping)
		else:
			self.precision = est.compute_payout(spy_mapping)


class FirstSpyEstimator(object):
	def __init__(self, A, num_honest_nodes, verbose = False, p_and_r = False):
		self.A = A
		self.num_honest_nodes = num_honest_nodes
		self.verbose = verbose
		self.p_and_r = p_and_r # whether to compute precision AND recall or just precision

	
	def compute_payout(self, spy_mapping):
		precision = 0
		recall = 0
		if self.verbose:
			print('\n')

		exits = {}
		for spy, info in spy_mapping.items():
			if self.verbose:
				print('spy', spy)
				exit_list = [item.exit_node for item in info]
				sources = [item.source for item in info]
				for exit in set(exit_list):
					print('exit', exit)
					print('sources', [sources[i] for i in range(len(exit_list)) if exit_list[i] == exit])
			
			for item in info:
				# if verbose:
				# 	print 'exit node', item.exit_node
				# 	print 'source', item.source
				if item.exit_node not in exits:
					exits[item.exit_node] = [item.source]
				else:
					exits[item.exit_node].append(item.source)
		nodes_wards = np.zeros(int(2*len(self.A.nodes()))) 
		for exit, ward in exits.items():
			nodes_wards[len(ward)] +=1
			if exit in ward:
				precision += 1.0 / len(ward)
				recall += 1.0
		if self.verbose:
			print('num_honest_nodes', self.num_honest_nodes)
			print('total precision', precision)
		precision = precision / self.num_honest_nodes
		recall = recall / self.num_honest_nodes
		
		if self.p_and_r:
			return precision, recall, nodes_wards
		else:
			return precision


class MaxWeightEstimator(object):
	def __init__(self, A, honest_nodes, weights, verbose = False, p_and_r = False):
		self.A = A
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
		honest_node_indices = list(honest_node_mapping.values())

		# Next, create a random mapping, to randomize the labeling of nodes,
		# since networkx max-weight algorithm doesn't seem to randomize
		mapping = list(np.random.permutation(honest_node_indices))
		msg_mapping = [i+1 for i in np.random.permutation(honest_node_indices)]

		inv_mapping = [mapping.index(i) for i in honest_node_indices]

		
		honest_nodes = honest_node_indices # the label of an honest nodes represents its order in mapping
		messages = [-n-1 for n in honest_nodes]

		H = nx.Graph()
		H.add_nodes_from(honest_nodes)	# servers
		H.add_nodes_from(messages) 		# messages
		for msg in self.honest_nodes: 	# for each real message tag
			for src, likelihood in self.weights[msg].items():	# and for each candidate source (+likelihood)
				# add an edge from the relabeled source to the relabeled message with the appropriate weight

				H.add_edge(mapping[honest_node_mapping[src]], -msg_mapping[honest_node_mapping[msg]], weight = likelihood)


		matching = nx.max_weight_matching(H, maxcardinality = True)
		estimate = []
		for edge in matching:
			(a, b) = edge
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




class MaxWeightVerCheckEstimator(MaxWeightEstimator):
	def __init__(self, A, honest_nodes, honest_dandelions, weights, verbose = False, p_and_r = False):
		super(MaxWeightVerCheckSimulator, self).__init__(A, honest_nodes, weights, verbose, p_and_r)
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
		honest_node_indices = list(honest_node_mapping.values())

		# Next, create a random mapping, to randomize the labeling of nodes,
		# since networkx max-weight algorithm doesn't seem to randomize
		mapping = list(np.random.permutation(honest_node_indices))
		msg_mapping = [i+1 for i in np.random.permutation(honest_node_indices)]

		inv_mapping = [mapping.index(i) for i in honest_node_indices]

		
		honest_nodes = honest_node_indices # the label of an honest nodes represents its order in mapping
		messages = [-n-1 for n in honest_nodes]

		H = nx.Graph()
		H.add_nodes_from(honest_nodes)	# servers
		H.add_nodes_from(messages) 		# messages
		for msg in self.honest_nodes: 	# for each real message tag
			for src, likelihood in self.weights[msg].items():	# and for each candidate source (+likelihood)
				# add an edge from the relabeled source to the relabeled message with the appropriate weight

				H.add_edge(mapping[honest_node_mapping[src]], -msg_mapping[honest_node_mapping[msg]], weight = likelihood)

		# print nx.adjacency_matrix(H).todense()
		# print 'nodes', H.nodes()
		# left, right = nx.bipartite.sets(H)
		# print 'left, right', left, right
		matching = nx.max_weight_matching(H, maxcardinality = True)
		# print 'matching', matching
		estimate = []
		for a,b in matching.items():
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
	def __init__(self, A, num_honest_nodes, verbose = False, p_and_r = True, q = 0.0):
		super(MaxWeightLineSimulator, self).__init__(A, verbose, q)
		self.weights = None
		self.num_honest_nodes = num_honest_nodes
		self.p_and_r = p_and_r

		honest_nodes = [node for node in self.A if not self.A.nodes[node]['spy']]

		# Run the simulation
		spy_mapping, hops = super(MaxWeightLineSimulator, self).run_simulation()
		# Compute the weights
		self.compute_weights(spy_mapping)
		# Compute a max-weight matching
		est = MaxWeightEstimator(A, honest_nodes, self.weights, verbose, p_and_r)
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
		for spy, info in spy_mapping.items():
			
			for item in info: # for each message that exited to spy
				if item.exit_node not in exits:
					exits[item.exit_node] = [(item.source, item.relayed_to_spy)]
				else:
					exits[item.exit_node].append((item.source, item.relayed_to_spy))
		''' for each ward, compute the local neighborhood and the
			likelihood of each node in that neighborhood to end up
			at the given exit node. The likelihoods are equal for 
			all nodes at a given exit node.
		'''
		for exit, ward in exits.items():
			relayed_ward = [item[0] for item in ward if item[1]]
			not_relayed_ward = [item[0] for item in ward if not item[1]]

			if len(relayed_ward) > 0:
				likelihoods_relayed = self.compute_likelihoods(exit)
			if len(not_relayed_ward) > 0:
				likelihoods_not_relayed = self.compute_likelihoods(exit, relayed_to_spy = False)
			for msg in relayed_ward:
				self.weights[msg] = likelihoods_relayed
			for msg in not_relayed_ward:
				self.weights[msg] = likelihoods_not_relayed

		# print 'exit', exit
		# print 'ward', ward
		# print 'likelihoods', likelihoods


	def compute_likelihoods(self, exit, relayed_to_spy = True):
		''' Compute the likelihoods of each node being the source. 
		Inputs:
			exit 			the stem terminus 
			relayed_to_spy 		whether exit is a valid source node (if exit 
							did not pass the message to a spy, then its 
							not valid 
		Output:
			likelihoods 	a dictionary of likelihoods for each node
		'''
		n_tilde = float(self.num_honest_nodes) / self.A.number_of_nodes()
		local_tree_depth = math.floor(1.5 / n_tilde)

		spies = [node for node in self.A.nodes() if self.A.nodes[node]['spy']]

		likelihoods = {}
		shortest_paths = nx.shortest_path(self.A, target = exit)
		for source, path in shortest_paths.items():
			if source in spies:
				continue
			if (not relayed_to_spy) and (source == exit):
				continue

			if not any([n in spies for n in path[:-1]]):
				# if the shortest path between node and exit has a spy, assign it 0 likelihood
				# this is an approximation, since a message could have taken a longer path
				degrees = [1.0 / self.A.out_degree(node) * (1 - self.q) for node in path]
				if self.q < 1:
					likelihoods[source] = np.prod(degrees) / (1 - self.q)
					if not relayed_to_spy:
						likelihoods[source] = likelihoods[source] / (1 - self.q) * self.q


		return likelihoods

class MaxWeightLineSimulatorUnknownTerminus(MaxWeightLineSimulator):

	def __init__(self, A, G, num_honest_nodes, verbose = False, p_and_r = True, q = 0.0):
		super(MaxWeightLineSimulatorUnknownTerminus, self).__init__(A, num_honest_nodes, verbose, p_and_r, q)
		self.G = G # we're calling self.G=A and self.A=G :( to be consistent with prior code
		
		honest_nodes = [node for node in self.A if not self.A.node[node]['spy']]

		# Run the simulation
		spy_mapping = self.run_simulation()
		# Compute the weights
		self.compute_weights(spy_mapping)
		# Compute a max-weight matching
		est = MaxWeightEstimator(self.A, honest_nodes, self.weights, verbose, p_and_r)
		src_estimate = est.compute_estimate()
		# Compute the precision
		if p_and_r:
			self.precision, self.recall = est.compute_payout(src_estimate)
		else:
			self.precision = est.compute_payout(src_estimate)

	def run_simulation(self):
		''' Simulates dandelion spreading over the anonymity graph, 
			with diffusion over the main graph. 
		'''

		# Assign the diffusers to the graph
		self.A = assign_diffusers(self.A, self.q, len(self.A.nodes()))
		# Run a line simulation over A
		# List of all spies
		spies = nx.get_node_attributes(self.A,'spy')
		if self.per_tx_path == False:
			diffusers = nx.get_node_attributes(self.A, 'diffuser')
		spy_mapping = {}
		# Make a dict of spies
		for node in self.A.nodes():
			# if node is a spy, add it to the dictionary
			if spies[node]:
				spy_mapping[node] = []

		# find the nodes reporting to each spy
		for node in self.A.nodes():
			if spies[node]:
				continue	
			spy_found = False
			pre_tail = node
			tail = node
			path_length = 0
			while True:
				neighbors = list(self.A.successors(tail))
				pre_tail = tail
				tail = random.choice(neighbors)
				path_length += 1
				# print 'node', node, 'neighbors', neighbors, 'tail', tail
				if spies[tail]:
					spy_mapping[tail].append(SpyInfo(tail, pre_tail, node))
					break
				if (self.per_tx_q and np.random.binomial(1, self.q)) or \
					(not self.per_tx_q and diffusers[tail] == True):
					# end the stem
					# now simulate diffusion starting from tail, and use the first-spy est
					# 	to guess who was the source of the diffusion process
					sim2 = FirstSpyDiffusionSimulatorOneSource(self.G, self.num_honest_nodes, tail,
															   verbose=self.verbose)
					tail = sim2.est_src
					# print 'tail1', tail, 'tail2', tail2
					if tail in spy_mapping:
						spy_mapping[tail].append(SpyInfo(tail, tail, node, relayed_to_spy = False))
					else:
						spy_mapping[tail] = [SpyInfo(tail, tail, node, relayed_to_spy = False)]
					break
				if path_length > nx.number_of_nodes(self.A):
					# there are no spies on this path, so we'll just assign the 
					#   last node to see the message to a spy
					spy = random.choice(list(spy_mapping.keys()))
					spy_mapping[spy].append(SpyInfo(spy, tail, node))
					break
			if self.verbose:
				print('Node ', node, ' traversed ', path_length, 'hops')

		return spy_mapping




class FirstSpyLineSimulatorMultipleTx(LineSimulator):
	def __init__(self, A, num_honest_nodes, verbose = False, num_tx = 1):
		super(FirstSpyLineSimulatorMultipleTx, self).__init__(A, verbose)
		self.num_honest_nodes = num_honest_nodes
		self.num_tx = num_tx
		self.spy_distribution = {}
		self.init_spy_distribution()

		# Contribute to that vector
		for tx in range(num_tx):
			spy_mapping = super(FirstSpyLineSimulatorMultipleTx, self).run_simulation()
			self.update_spy_distribution(spy_mapping)

	def init_spy_distribution(self):
		# initialize the spy distribution
		is_spy = nx.get_node_attributes(self.A,'spy')
		spies = [n for n in self.A.nodes() if is_spy[n]]
		honest_nodes = [n for n in self.A.nodes() if not is_spy[n]]
		# cycle over honest nodes
		for node in honest_nodes:
			self.spy_distribution[node] = {}
			for spy in spies:
				self.spy_distribution[node][spy] = 0


	def update_spy_distribution(self, spy_mapping):
		# Update the first spy counts for each source
		for spy, info in spy_mapping.items():
			for item in info:
				self.spy_distribution[item.source][spy] += 1.0 / self.num_tx

class DiffusionSimulator(Simulator):
	def __init__(self, A, num_honest_nodes, verbose = False):
		super(DiffusionSimulator, self).__init__(A, num_honest_nodes, verbose)
		
	def run_simulation(self):
		'''
			Runs a diffusion process and keeps track of the spy that first sees
			each message, as well as the node who delivered the message to the
			spy. The spy observations are returned.
		'''

		# List of all spies
		spies = nx.get_node_attributes(self.A,'spy')

		spy_mapping = {}
		honest = 0
		# Make a dict of spies
		for node in self.A.nodes():
			# if node is a spy, add it to the dictionary
			if spies[node]:
				spy_mapping[node] = []
		hops = np.zeros(int(2*len(self.A.nodes())))
		# find the nodes reporting to each spy
		mean_pathlength = 0
		for node in self.A.nodes():
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
				mean_pathlength+=1
				# print 'node', node, 'neighbors', neighbors, 'tail', tail
				if spies[target]:
					spy_mapping[target].append(SpyInfo(target, source, node))
					break
				if path_length >= nx.number_of_nodes(self.A):
					# there are no spies on this path, so we'll just assign the 
					#   last node to see the message to a spy

					spy = random.choice(list(spy_mapping.keys()))
					spy_mapping[spy].append(SpyInfo(spy, target, node))
					break
				infected += [target]
				boundary_edges.remove(next)
				boundary_edges = [item for item in boundary_edges if item[1] != target]
				boundary_edges += [[target, item] for item in nx.all_neighbors(self.A, target) if item not in infected]

				# add next to boundary, remove the infecting node if it is eclipsed
			if self.verbose:
				print('Node ', node, ' traversed ', path_length, 'hops')
			hops[path_length] +=1

		return spy_mapping, hops

	def get_neighbor_set(self, candidates, infected):
		''' Returns the set of susceptible edges from infected zone to uninfected '''
		neighbors = []
		for node in candidates:
			for neighbor in nx.all_neighbors(self.A, node):
				if (neighbor not in infected) and ([node, neighbor] not in neighbors):
					neighbors += [[node, neighbor]]

		return neighbors

class FirstSpyDiffusionSimulator(DiffusionSimulator):
	def __init__(self, A, num_honest_nodes, verbose = False, p_and_r = True):
		super(FirstSpyDiffusionSimulator, self).__init__(A, num_honest_nodes, verbose)
		self.p_and_r = p_and_r

		spy_mapping, hops = super(FirstSpyDiffusionSimulator, self).run_simulation()
		self.hops = hops
		est = FirstSpyEstimator(A, num_honest_nodes, verbose, p_and_r)
		if p_and_r:
			self.precision, self.recall, self.wards = est.compute_payout(spy_mapping)
		else:
			self.precision = est.compute_payout(spy_mapping)

	def run_simulation(self):
		'''
			Runs a diffusion process and keeps track of the spy that first sees
			each message, as well as the node who delivered the message to the
			spy. The spy observations are returned.
		'''

		# List of all spies
		spies = nx.get_node_attributes(self.A,'spy')

		spy_mapping = {}

		# Make a dict of spies
		for node in self.A.nodes():
			# if node is a spy, add it to the dictionary
			if spies[node]:
				spy_mapping[node] = []

		# find the nodes reporting to each spy
		for node in self.A.nodes():
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
				if path_length >= nx.number_of_nodes(self.A):
					# there are no spies on this path, so we'll just assign the 
					#   last node to see the message to a spy

					spy = random.choice(list(spy_mapping.keys()))
					spy_mapping[spy].append(SpyInfo(spy, target, node))
					break
				infected += [target]
				boundary_edges.remove(next)
				boundary_edges = [item for item in boundary_edges if item[1] != target]
				boundary_edges += [[target, item] for item in nx.all_neighbors(self.A, target) if item not in infected]

				# add next to boundary, remove the infecting node if it is eclipsed
			if self.verbose:
				print('Node ', node, ' traversed ', path_length, 'hops')

		return spy_mapping

class FirstSpyDiffusionSimulatorAsymmetric(DiffusionSimulator):
	def __init__(self, A, num_honest_nodes, verbose = False, p_and_r = True, asymmetric = False):
		super(FirstSpyDiffusionSimulatorAsymmetric, self).__init__(A, num_honest_nodes, verbose)
		self.p_and_r = p_and_r
		self.asymmetric = asymmetric

		spy_mapping = self.run_simulation()
		est = FirstSpyEstimator(A, num_honest_nodes, verbose, p_and_r)
		if p_and_r:
			self.precision, self.recall = est.compute_payout(spy_mapping)
		else:
			self.precision = est.compute_payout(spy_mapping)
				
	def run_simulation(self):
		'''
			Runs a diffusion process and keeps track of the spy that first sees
			each message, as well as the node who delivered the message to the
			spy. The spy observations are returned.
		'''

		# List of all spies
		spies = nx.get_node_attributes(self.A,'spy')
		
		spy_mapping = {}

		# Make a dict of spies
		for node in self.A.nodes():
			# if node is a spy, add it to the dictionary
			if spies[node]:
				spy_mapping[node] = []

		# find the nodes reporting to each spy
		for node in self.A.nodes():
			if spies[node]:
				continue	
			spy_found = False

			infected = [node]
			boundary_edges = self.get_neighbor_set(infected, infected)
			path_length = 0
			while boundary_edges:

				if self.asymmetric:
					# print 'boundary_edges', boundary_edges, 'src node', node

					weights = [2 if tuple(e) in self.A.edges() else 1 for e in boundary_edges]
					# print 'weights', weights
					weights = [float(i) / sum(weights) for i in weights]
					next_idx = np.random.choice(list(range(len(boundary_edges))), 1, p=weights)[0]
					next = boundary_edges[next_idx]
					# print 'chose', next
				else:
					next = random.choice(boundary_edges)
				source = next[0]
				target = next[1]
				path_length += 1
				# print 'node', node, 'neighbors', neighbors, 'tail', tail
				if spies[target]:
					# print 'source node', node, ' reported to target ', target, 'through ', source
					spy_mapping[target].append(SpyInfo(target, source, node))
					break
				if path_length >= nx.number_of_nodes(self.A):
					# there are no spies on this path, so we'll just assign the 
					#   last node to see the message to a spy

					spy = random.choice(list(spy_mapping.keys()))
					spy_mapping[spy].append(SpyInfo(spy, target, node))
					break
				infected += [target]
				boundary_edges.remove(next)
				boundary_edges = [item for item in boundary_edges if item[1] != target]
				boundary_edges += [[target, item] for item in nx.all_neighbors(self.A,target) if item not in infected]

				# add next to boundary, remove the infecting node if it is eclipsed
			if self.verbose:
				print('Node ', node, ' traversed ', path_length, 'hops')


		return spy_mapping

class FirstSpyDiffusionSimulatorOneSource(DiffusionSimulator):
	def __init__(self, A, num_honest_nodes, src, verbose = False):
		H = A.to_undirected()
		super(FirstSpyDiffusionSimulatorOneSource, self).__init__(H, num_honest_nodes, verbose)
		self.src = src

		self.est_src = self.run_simulation(src)

	def run_simulation(self, src):
		'''
			Runs a diffusion process and keeps track of the spy that first sees
			the one message. 
		'''

		# List of all spies
		spies = nx.get_node_attributes(self.A,'spy')

		# run the diffusion process
		infected = [src]
		boundary_edges = self.get_neighbor_set(infected, infected)
		path_length = 0
		while boundary_edges:

			# weights = [2 if tuple(e) in self.A.edges() else 1 for e in boundary_edges]
			# # print 'weights', weights
			# weights = [float(i) / sum(weights) for i in weights]
			# next_idx = np.random.choice(range(len(boundary_edges)), 1, p=weights)[0]
			# next = boundary_edges[next_idx]
			next = random.choice(boundary_edges)
			source = next[0]
			target = next[1]
			path_length += 1
			# print 'src', src, 'neighbors', neighbors, 'tail', tail
			if spies[target] or (path_length >= nx.number_of_nodes(self.A)):
				return source

			infected += [target]
			boundary_edges.remove(next)
			boundary_edges = [item for item in boundary_edges if item[1] != target]
			boundary_edges += [[target, item] for item in nx.all_neighbors(self.A, target) if item not in infected]

		return None

def assign_diffusers(A, q, n):
	diffuser_list = random.sample(list(range(n)), int(math.floor(q*n)))
	diffusers = dict([(k, k in diffuser_list) for k in range(n)])
	nx.set_node_attributes(A, diffusers, 'diffuser')
	return A
	


	