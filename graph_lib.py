# graph_lib.py
# generates graphs for simulation


import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import collections
import math

# Graph construction protocols
NO_VERSION_CHECKING = 0
VERSION_CHECKING = 1

class GraphGen(object):
	def __init__(self, n, p, verbose = False):
		'''	n -- number of nodes
			p -- proportion of spies
		'''
	
		self.n = n
		self.p = p
		self.verbose = verbose
		
		self.G = nx.DiGraph()
		self.G.add_nodes_from(list(range(self.n)))
		self.assign_spies()

	def assign_spies(self):
		spy_list = random.sample(list(range(self.n)), int(math.floor(self.p*self.n)))
		spies = dict([(k, k in spy_list) for k in range(self.n)])
		nx.set_node_attributes(self.G, spies, 'spy')

	def remove_self_loops(self):
		# Remove any length-2 loops
		for node in self.G.nodes():
			for successor in self.G.successors(node):
				if successor == node and self.verbose:
					print('SELF LOOP')
					self.G.remove_edge(node, successor)
					continue

				if node in self.G.successors(successor) and self.verbose:
					print('2-LOOP')
					self.G.remove_edge(node, successor)


class DataGraphGen(object):
	def __init__(self, filename, p, verbose = False):
		'''
			Loads a graph from a gexf data file, and relabels the nodes in 
			chronological order.
		'''
		self.G = nx.read_gexf(filename)
		mapping = {}
		for (idx, node) in zip(list(range(nx.number_of_nodes(self.G))), self.G.nodes()):
			mapping[node] = idx
		nx.relabel_nodes(self.G, mapping, copy=False)
		self.p = p
		self.assign_spies()


	def assign_spies(self):
		n = nx.number_of_nodes(self.G)
		spy_list = random.sample(list(range(n)), int(math.floor(self.p*n)))
		spies = dict([(k, k in spy_list) for k in range(n)])
		nx.set_node_attributes(self.G, spies, 'spy')
		

class RegGraphGen(GraphGen):
	def __init__(self, n, p, d, verbose = False):
		'''	d -- degree of graph (outdegree)
		'''
		super(RegGraphGen, self).__init__(n, p, verbose)
		self.d = d

		# Generate the graph
		self.generate_graph()
		
		self.A = self.G


	
	def generate_graph(self):
		''' Generates a directed random, d-regular graph with d/2 incoming and outgoing edges.'''
		cycle = nx.cycle_graph(self.n)
		
		self.G.add_edges_from(nx.find_cycle(cycle))

		
		
		for i in range(self.d-1):
			perm = np.random.permutation(self.n)
			mapping = {key:value for (key,value) in zip(list(range(self.n)), perm)}
			cycle = nx.relabel_nodes(cycle, mapping) # add a single hamiltonian cycle
			self.G.add_edges_from(nx.find_cycle(cycle))

		self.remove_self_loops()

		if self.verbose:
			degrees = G.degree(G.nodes())
			print(collections.Counter(list(degrees.values())))


class QuasiRegThreshGraphGen(GraphGen):
	def __init__(self, n, p, d, k=0, verbose = False):
		''' Generates an approximately d-regular graph by making d/2 
			outgoing connections at random. The algorithm also asks
			each connection recipient if it already has degree d. If 
			so, the connection originator tries another node. Each 
			connecting node makes k such queries, k >= 0.

			d -- degree of graph (outdegree)
		   	k -- number of times to ask if recipient has d 
		   		 connections before just connecting anyway
		'''

		super(QuasiRegThreshGraphGen, self).__init__(n, p, verbose)
		self.d = d
		self.k = k

		# Generate the graph
		self.generate_graph()


	def generate_graph(self):

		
		# make the connections
		for node in self.G.nodes():
			# make a list of the other nodes in the graph
			other_nodes = self.G.nodes()
			other_nodes.remove(node)

			# choose one at random
			targets = np.random.choice(other_nodes, self.d)

			# check the occupancy of the target at most k times
			for target in targets:
				for k in range(self.k):
					if self.G.in_degree(target) >= self.d:
						target = np.random.choice(other_nodes)
					else:
						break
				# connect to the target
				self.G.add_edge(node, target)

		self.remove_self_loops()

class QuasiRegGraphGen(GraphGen):
	def __init__(self, n, p, d, verbose = False, beta = 1.0, 
				 anon_graph_protocol = NO_VERSION_CHECKING, 
				 d_anon = None):
		''' Generates an approximately d-regular graph by making d/2 
			outgoing connections at random. The algorithm also asks
			each connection recipient if it already has degree d. If 
			so, the connection originator tries another node. Each 
			connecting node makes k such queries, k >= 0.

			n 		number of nodes
			p 		fraction of spies
			d 		degree of main graph (outdegree)
		   	beta 	fraction of honest nodes that run dandelion
	   		anon_graph_protocol 	which graph construction protocol to use (see constants @ 
	   								beginning of file)
			d_anon 		out-degree of anonymity graph
		'''

		super(QuasiRegGraphGen, self).__init__(n, p, verbose)
		self.d = d
		self.beta = beta
		self.anon_graph_protocol = anon_graph_protocol
		self.A = nx.DiGraph() 	# anonymity graph, empty for now
		self.d_anon = d_anon

		# Generate the graph
		self.generate_graph()
		self.A.add_nodes_from(self.G.nodes(data=True))

		if (d_anon is None) or (d_anon == d):
			self.A.add_edges_from(self.G.edges())
		else:
			if beta < 1.0:
				self.assign_dandelion_nodes()
			self.generate_anon_graph()

	def assign_dandelion_nodes(self):
		n_honest = math.ceil((1 - self.p) * self.n)
		dand_list = random.sample([v for v in self.G.nodes() if not self.G.node[v]['spy']], n_honest) 
		dand_list += [v for v in self.G.nodes() if self.G.node[v]['spy']]
		dand_nodes = dict([(k, k in dand_list) for k in range(self.n)])
		nx.set_node_attributes(self.G, 'dand', dand_nodes)


	def generate_graph(self):

		G = nx.DiGraph()

		# make the connections
		for node in self.G.nodes():
			# make a list of the other nodes in the graph
			nodes_other = [i for i in self.G.nodes() if i != node]
			# if (self.k == 1):
			choices = random.sample(nodes_other, self.d)
			for target in choices:
				self.G.add_edge(node,target)

		self.remove_self_loops()


	def generate_anon_graph(self):
		# build up self.A (the anonymity graph)
		for n in self.G.nodes():
			# Don't add edges for non-dandelion nodes
			if self.beta < 1.0 and not self.G.node[n]['dand']:
				continue
			# select outgoing edges from each node's out-edges on the graph
			successors = list(self.G.successors(n))
			candidates = []
			if self.anon_graph_protocol == VERSION_CHECKING:
				dand_nodes = [v for v in successors if self.G.node[v]['dand']]
				if not dand_nodes:
					candidates = successors
				else:
					candidates = dand_nodes
			elif self.anon_graph_protocol == NO_VERSION_CHECKING:
				candidates = successors

			out_edges = random.sample(candidates, min([self.d_anon, len(candidates)]))
			new_edges = [(n, item) for item in out_edges]
			self.A.add_edges_from(new_edges)

class QuasiRegGraphGenEavesdropper(QuasiRegGraphGen):
	def __init__(self, n, theta, d, verbose = False, d_anon = None):
		# First create a graph w no spies
		super(QuasiRegGraphGenEavesdropper, self).__init__(n, 0.0, d, verbose)
		# Then add  the eavesdroppers
		eavesdroppers = list(range(n, n+theta))
		honest_nodes = self.G.nodes()
		self.G.add_nodes_from(eavesdroppers, spy=True)
		for e in eavesdroppers:
			for n in honest_nodes:
				self.G.add_edge(e, n)

class QuasiRegGraphGenSpiesOutbound(QuasiRegGraphGen):
	def __init__(self, n, p, d, verbose = False, d_anon = None):
		''' Generates an approximately d-regular graph by making d/2 
			outgoing connections at random. Spies always connect to all
			honest nodes.

			n 		number of nodes
			p 		fraction of spies
			d 		degree of graph (outdegree)
		'''

		super(QuasiRegGraphGenSpiesOutbound, self).__init__(n, p, d, verbose, d_anon=d_anon)
		
		# Make sure that each spy is connected to all the honest nodes
		nodelist = list(self.G.nodes())
		spies = [n for n in nodelist if self.G.nodes[n]['spy']]
		print(spies, type(spies))
		honest_nodes = list(set(self.G.nodes()) - set(spies))
		for spy in spies:
			# Main P2P graph
			self.G.remove_edges_from(list(self.G.out_edges(spy)))
			self.G.add_edges_from([(spy, i) for i in honest_nodes])
			# Anonymity graph
			self.A.remove_edges_from(list(self.A.out_edges(spy)))
			self.A.add_edges_from([(spy, i) for i in honest_nodes])

class QuasiRegGraphGenSpies(QuasiRegGraphGen):
	def __init__(self, n, p, d = 8, verbose = False, k = 1, beta = 1.0, 
				 anon_graph_protocol = NO_VERSION_CHECKING, 
				 d_anon = None):
		''' Generates an approximately d-regular graph by making d/2 
			outgoing connections at random. The algorithm also asks
			each connection recipient if it already has degree d. If 
			so, the connection originator tries another node. Each 
			connecting node makes k such queries, k >= 0. Spies always 
			lie about their degrees, claiming they have degree 0

			n 		number of nodes
			p 		fraction of spies
			d 		degree of graph (outdegree)
			k 		number of nodes to poll before choosing a connection (default=1)
		   	beta 	fraction of honest nodes that run dandelion
	   		anon_graph_protocol 	which graph construction protocol to use (see global constants)
			d_anon 		out-degree of anonymity graph
		'''

		self.k = k
		self.A = nx.DiGraph() 	# anonymity graph, empty for now
		
		super(QuasiRegGraphGenSpies, self).__init__(n, p, d, verbose, d_anon=d_anon)
		
		
		if beta < 1.0:
			self.assign_dandelion_nodes()
			self.generate_anon_graph()
		else:
			self.A = self.G

	def generate_graph(self):

		# make the connections
		for node in self.G.nodes():
			# make a list of the other nodes in the graph
			nodes_other = [i for i in self.G.nodes() if i != node]
			if (self.k == 1):
				choices = random.sample(nodes_other, self.d)
				for target in choices:
					self.G.add_edge(node,target)
			else:
				for connection in range(self.d):
					choices = random.sample(nodes_other, self.k)
					# print 'choices', choices
					degrees = [self.G.degree(i) if not self.G.node[i]['spy'] else 0 for i in choices]
					# print 'degrees', degrees
					min_degree_indices = [i for i in range(len(degrees)) if degrees[i] == min(degrees)]
					# print 'min degree indices', min_degree_indices
					target = choices[np.random.choice(min_degree_indices)]

					self.G.add_edge(node, target)

		self.remove_self_loops()
	

class CompleteGraphGen(GraphGen):
	def __init__(self, n, p, verbose = False):
		super(CompleteGraphGen, self).__init__(n, p, verbose)

		self.G = self.generate_graph()

	def generate_graph(self):
		
		G = nx.complete_graph(self.n).to_directed()
		spy_list = random.sample(list(range(self.n)), int(math.floor(self.p*self.n)))
		spies = dict([(k, k in spy_list) for k in range(n)])
		# spies = dict([(k, (random.random() < p)) for k in range(n)])
		nx.set_node_attributes(G, spies, 'spy')

		return G