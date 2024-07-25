# data_graph_rep.py
# contains the classes for storing and processing spreading on a data-provided graph

import networkx as nx
import random
# import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate


class DataGraph(nx.Graph):

	def __init__(self, filename, spreading_time = None, lambda1 = 1):
		super(DataGraph, self).__init__(nx.read_gexf(filename))
		self.lambda1 = lambda1 # spreading rate over the diffusion graph
		
		
		# Read graph and label nodes from 1 to N
		mapping = {}
		for (idx, node) in zip(list(range(nx.number_of_nodes(self))), self.nodes()):
			mapping[node] = idx
		nx.relabel_nodes(self, mapping, copy=False)

		self.spreading_time = spreading_time	

# Run diffusion over a provided gexf graph
class DataGraphDiffusion(DataGraph):

	def __init__(self, filename, spreading_time = None, lambda1 = 1):
		''' NB: Here the spreading_time	is actually the number of rings of the graph to infect'''
		super(DataGraphDiffusion, self).__init__(filename, spreading_time, lambda1)
		

		
	def spread_message(self, source = 0, first_spy_only = False, num_corrupt_cnx = 1):
		'''first_spy_only denotes whether this diffusion spread will only be used
		to measure the first spy adversary. In that case, some time-saving optimizations
		can be implemented. Most of the time, this flag will be set to false.'''
		
		self.source = source
		self.num_corrupt_cnx = num_corrupt_cnx

		# Empty the observed timestamps
		self.adversary_timestamps = {} 		# dictionary of adversary report time indexed by node
		self.received_timestamps = {}		# dictionary of message receipt time indexed by node

		# INitialize the process
		current_time = 0
		self.received_timestamps[self.source] = 0
		self.adversary_timestamps[self.source] = self.send_to_adversary(self.source, num_corrupt_cnx)
		if first_spy_only:
			stopping_time = min(self.spreading_time, self.adversary_timestamps[self.source])
		# self.active = [source]
		self.infected = [source]

		stopping_time = self.spreading_time
		
		self.infected_by_source = {}
		
		self.active = [(source, n) for n in self.neighbors(source)]	# number of active edges
		count = 0
		while self.active:
			# Compute the delay
			node, neighbor = random.choice(self.active)	# the edge that will fire next
			current_time = self.exponential_delay(current_time, self.lambda1 * len(self.active))	# associated spreading delay
			# print 'current_time', current_time, 'stopping_time', stopping_time
			count += 1
			if current_time > stopping_time:
				break
			self.received_timestamps[neighbor] = current_time

			# Mark neighbor as infected
			# print 'Order: ', node, ' infects ', neighbor
			self.infected += [neighbor]
			if node == source:
				self.infected_by_source[neighbor] = True
			else:
				self.infected_by_source[neighbor] = False

			# Find the reporting time
			adversary_timestamp = self.send_to_adversary(neighbor, num_corrupt_cnx)
			if adversary_timestamp <= self.spreading_time:
				self.adversary_timestamps[neighbor] = adversary_timestamp

			# Clean up the list of edges
			self.active.remove((node, neighbor))
			self.active += [(neighbor, n) for n in self.neighbors(neighbor) if n not in self.infected]
			new_boundary = [edge for edge in self.active if edge[0] in self.infected and edge[1] not in self.infected]
			self.active = [i for i in new_boundary]
		# print 'num infected nodes: ', len(self.infected)


		# print 'infected nodes', self.infected, len(self.infected)
		# print 'rx timetsamps', [(n,self.received_timestamps[n]) for n in self.infected]
		# print 'timetsamps', [(n,self.adversary_timestamps[n]) for n in self.infected if n in self.adversary_timestamps]

	def exponential_delay(self, current_time, rate):
		return current_time + np.random.exponential(1.0 / rate)

	def send_to_adversary(self, node, num_corrupt_cnx):
		return self.received_timestamps[node] + np.random.exponential(1.0 / num_corrupt_cnx)
		# return self.received_timestamps[node]

# Run diffusion over a provided gexf graph
class DataGraphDandelion(DataGraphDiffusion):

	def __init__(self, filename, spreading_time = None, lambda1 = 1):
		''' Parameters:
			filename		graph filename
			spreading_time	maximum time to spread
			diffusion_delay 1/rate of exponential delay distribution
			hops			num hops in the Dandelion stem
		'''
		super(DataGraphDandelion, self).__init__(filename, spreading_time, lambda1)

		relay_delay_filename = 'data/forwarding_delay.npz'
		network_delay_filename = 'data/network_delay_cdf.npz'

		# load the empirical delay distributions
		# This npz file has two parts: 'bins' and 'prob'
		self.relay_delay_npz = np.load(relay_delay_filename)
		self.network_delay_cdf = np.load(network_delay_filename)


	def sample_from_cdf(self, bins, prob, nsamples = 1):
		inv_cdf = interpolate.interp1d(prob, bins)
		r = np.random.rand(nsamples)
		return inv_cdf(r)
	
		
	def spread_message(self, source = 0, hops = 0):
		''' Spread the message according to Dandelion spreading and keep track of all timestamps'''

		print(len(self.nodes()), 'nodes')

		self.source = source
		
		# Empty the observed timestamps
		self.received_timestamps = {}
		
		# Initialize the process
		current_time = 0
		self.infected = [source]
		active = []

		# Stem phase
		head = self.source
		for hop in range(hops):
			active = self.neighbors(head)
			neighbor = random.choice(active)	# the next stem node
			current_time = self.compute_transmission_delay(current_time, processing = False)	# associated spreading delay
			if current_time > self.spreading_time:
				break
			# self.received_timestamps[neighbor] = current_time
			head = neighbor


			
		# Fluff phase

		unused_edges = self.edges()
		self.received_timestamps[head] = current_time

		active = [(head, n) for n in self.neighbors(head)]
		while active:
			print(len(active), ' active nodes , with ', len(list(self.received_timestamps.keys())), ' infected items')
			new_boundary = []

			# cycle through the active edges
			for edge in active:
				origin = edge[0]
				target = edge[1]
				# If this edge has already been removed, move on
				if not ((origin, target) in unused_edges or (target, origin) in unused_edges):
					continue

				if (target in self.received_timestamps and self.received_timestamps[target] <= self.received_timestamps[origin]):
					continue
				# Remove the edge from unused edges
				try:
					unused_edges.remove((origin,target))
				except:
					unused_edges.remove((target, origin))

				# Compute the delay on this edge
				infection_time = self.compute_transmission_delay(self.received_timestamps[origin])
				if (infection_time > self.spreading_time) or \
				   (target in self.received_timestamps and self.received_timestamps[target] <= infection_time):
					continue
				# Update the receipt time
				self.received_timestamps[target] = infection_time
				# Add the new neighbors to the new_boundary
				new_boundary += [(target, n) for n in self.neighbors(target) if \
								not (n in self.received_timestamps and self.received_timestamps[n] <= self.received_timestamps[target]) \
								and ((target,n) in unused_edges or (n, target) in unused_edges)]

			active = [item for item in new_boundary]

		print('num infected nodes', len(list(self.received_timestamps.keys())))

		# Fluff phase (exponential mechanism, event-based simulation)

		# self.active = [(head, n) for n in self.neighbors(head)]	# number of active edges
		# while self.active:
		# 	# Compute the delay
		# 	node, neighbor = random.choice(self.active)	# the edge that will fire next
		# 	# print self.active, len(self.active)
		# 	current_time = self.compute_transmission_delay(current_time, self.lambda1 * len(self.active))	# associated spreading delay
		# 	# print 'current_time', current_time, 'stopping_time', stopping_time
		# 	print 'num infected: ', len(self.infected), 'out of ', len(self.nodes()), 'at time ', current_time
		# 	if current_time > self.spreading_time:
		# 		break
		# 	self.received_timestamps[neighbor] = current_time

		# 	# Mark neighbor as infected
		# 	# print 'Order: ', node, ' infects ', neighbor
		# 	self.infected += [neighbor]
			
		# 	# Clean up the list of edges
		# 	self.active.remove((node, neighbor))
		# 	self.active += [(neighbor, n) for n in self.neighbors(neighbor) if n not in self.infected]
		# 	new_boundary = [edge for edge in self.active if edge[0] in self.infected and edge[1] not in self.infected]
		# 	self.active = [i for i in new_boundary]

	def compute_transmission_delay(self, current_time, processing = True):
		''' Computes the delay from 
				(a) Diffusion delays
				(b) Natural transaction processing delay
				(c) Network delay
			Inputs:
				current_time 			the current time of the simulation
				diffusion_delay 		the exponential rate of the next timestamp (not used in the data-driven model)
				processing 				tells whether we should include processing delay in the total delay
		'''
		# Diffusion delay
		# final_time = self.exponential_delay(current_time, diffusion_rate)
		# final_time = current_time + 
		if processing:
			forwarding_delay = np.random.choice(self.relay_delay_npz['bins'], size=1, 
												replace=False, p=self.relay_delay_npz['prob'])
		else:
			forwarding_delay = 0.0
		# final_time += 0.001
		# Network delay (in seconds)
		network_delay = self.sample_from_cdf(self.network_delay_cdf['bins'], self.network_delay_cdf['prob'], 1) / 1000.0

		final_time = current_time + forwarding_delay + network_delay
		# final_time += 0.2

		return final_time