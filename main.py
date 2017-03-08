import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os
import pickle

from sklearn.cross_validation import train_test_split
from sklearn.neighbors.kde import KernelDensity
from sklearn.neighbors.regression import KNeighborsRegressor

__doc__ = """
"""

# STATS FOR PRODUCT
MEAN_PRODUCT = 19.714
SIGMA_PRODUCT = 2.3981

# ###### AGGREGATED NETWORK
# STATS FOR ACCESS NETWORK
MEAN_MSAN = 186.63  # AVG NUMBER OF USERS
SIGMA_MSAN = 90.16  # STD DEVIATION
NETWORK_SIZE = 1000  # ACCESS NETWORK SIZE
MAX_USERS = 720  # MSAN CAPACITY USERS
# ---------------
TRAFFIC_DIST_SIZE = 288  # NUMBER OF TRAFFIC ELEMENTS FOR EACH 5 MINUTES AVG FROM 0h TO 24h
USER_TRAFFIC_MEAN = 0.8  # AVG TRAFFIC DURING 20h TO 24h IN Kbps
USER_TRAFFIC_SIGMA = 0.7  # STD DEV FOR USER TRAFFIC 20h TO 24h IN Kbps
# -----------------

METRO_ETH_RING_MEAN = 2654.44  # USERS AT METRO ETHERNET NETWORK MEAN
METRO_ETH_RING_SIGMA = 2196.75  # STD DEVIATION
# -----------------
METRO_ETH_ARD_MEAN = 604.0  # USERS AT METRO ETHERNET NETWORK MEAN
METRO_ETH_ARD_SIGMA = 282.0  # STD DEVIATION


# ########################


def main():
	start_time = time.time()
	# READING USERS WHICH HAVE BEEN POOLED FOR STATISTICS
	# 	VALUES FROM THIS DATASET WILL BE USED TO SIMULATE USER'S
	# BEHAVIOR OVER TIME.

	#users = create_user_info('users_sampled', 'rra')

	#predictions = user_traffic_model(users_dict=users)

	#user_model_visualization(predictions)

	#user_traffic_visualization(users)

	# INITIATE ACCESS NETWORK
	if os.path.exists('access_network.p'):
		with open('access_network.p', mode='rb') as f:
			access_network = pickle.load(f)
			f.close()
	else:
		with open('access_network.p', mode='wb') as f:
			access_network = create_access_network()
			pickle.dump(access_network, f, protocol=pickle.HIGHEST_PROTOCOL)
			f.close()

	# ACCESS NETWORT STATS
	print('Access Network Stats:\n'
		  '\tNumber of Devices: %d\n'
		  '\tAvg Users per Device: %0.2f Users\n'
		  % (access_network.shape[0],
			 np.sum(access_network[:, :, :] > 0) / TRAFFIC_DIST_SIZE / access_network.shape[0]))

	# get bras interface list

	#bras_list = get_bras_int_data_list()

	# x_axis = np.linspace(0, 5 * TRAFFIC_DIST_SIZE,TRAFFIC_DIST_SIZE)
	# plt.plot(x_axis,net_array[0, 0, :],x_axis,net_array[0, 1, :], label = 'User Traffic')
	# plt.legend(loc='upper left')
	# plt.ylabel('Tráfego [Mbps]')
	# plt.xlabel('Período [min]')
	# plt.show()

	end_time = time.time()

	print("-------- Total Time: %0.3f [s] -------" % (end_time - start_time))


def get_normal_dist(MEAN, SIGMA, SIZE=1):
	return abs(np.random.normal(MEAN, SIGMA, SIZE))


def get_bras_int_subscriber(input_list):
	X = np.array([input_list]).transpose()

	kernel = 'gaussian'

	kde = KernelDensity(kernel=kernel, bandwidth=150).fit(X)
	sample = kde.sample()

	return int(sample)


def get_bras_int_data_list():
	x = []
	with open('bras_int_subs', 'r') as f:
		for line in f:
			x.append(int(line.strip('\n')))

	return x


def create_access_network(n_access=NETWORK_SIZE, n_ports=MAX_USERS, time_series_entries=TRAFFIC_DIST_SIZE,
						  access_mean_traffic=MEAN_MSAN, access_sigma_traffic=SIGMA_MSAN,
						  user_traffic_mean=USER_TRAFFIC_MEAN, user_traffic_sigma=USER_TRAFFIC_SIGMA):

	"""
	Creates access network matrix based on:
	:param n_access: Number of Devices of this type
	:param n_ports: Number of Ports of this device type
	:param time_series_entries: Number of Entries for Time series [5 min step]. Example: 288 -> 00:00 to 23:55
	:param access_mean_traffic: Mean Aggregated Traffic for this device
	:param access_sigma_traffic: Distribution Traffic for this device
	:return: Dense Matrix Accumulated Traffic
	"""

	access_t0 = time.time()
	# CREATE ACCESS NETWORK DEVICE WITH 'MAX_USERS' ACCESS
	access_network = np.ndarray(shape=(n_access, n_ports, time_series_entries), dtype=np.float16)

	for i in range(NETWORK_SIZE):

		# INITIALIZING USERS - GENERATING RANDOM VALUES FROM NORMAL DISTRIBUTION
		# THIS FUNCTION GENERATES RANDOM USERS TO MSAN ACCORDING TO (MEAN_MSAN AND SIGMA_MSAN)
		num_of_users = int(get_normal_dist(access_mean_traffic, access_sigma_traffic))

		for j in range(MAX_USERS):
			if j <= num_of_users:
				# CREATE USER TRAFFIC FROM TIME SERIES 'TRAFFIC_DIST_SIZE'
				access_network[i, j, :] = get_normal_dist(user_traffic_mean, user_traffic_sigma, time_series_entries)
			else:
				access_network[i, j, :] = 0

		# SHUFFLING INSERTED USERS INTO 720 POSITIONS ARRAY
		np.random.shuffle(access_network[i, :, :])
	print('Access Network Created %0.3f[s]' % (time.time() - access_t0))
	return access_network


def create_user_info(users_data, rra_mapping):
	""" Return numpy array containing users traffic for the last 24h of collected traffic """

	RRA_LENGTH = 288
	d = {}

	user_model_pickle = 'user_model.p'

	if not os.path.exists(user_model_pickle):

		with open(users_data, 'r') as read_users:
			for row in read_users:
				s = row.split('\t')
				d[s[0]] = {'product': s[1]}

			read_users.close()

		with open(rra_mapping, 'r') as read_rra:
			for row in read_rra:
				s = row.split(' ')
				user = s[6].strip('\n')
				rra = int(s[1].strip(','))
				dest = dict(d[user])
				dest.update({'rra': rra})
				d[user] = dest

			read_rra.close()

		csv_files = os.listdir(os.path.join(os.getcwd(), 'csv'))

		for f in csv_files:
			for k_idx in d.keys():
				try:
					if str(d[k_idx]['rra']) in f and f.endswith('.csv'):
						dest = dict(d[k_idx])
						dest.update({'file': f})
						d[k_idx] = dest
				except Exception as e:
					pass

		for k_idx in d.keys():
			try:
				# OPEN USER'S STATISTICS DATA AND APPEND IT TO USER DICT
				with open(os.path.join(os.getcwd(), 'csv/' + d[k_idx]['file']), 'r') as f:
					for i, row in enumerate(f):
						s = row.split(' ')
						if ('port' not in s[0] or 'port' not in s[1]) and i < RRA_LENGTH:
							dest = d[k_idx]
							# dest.update({int(s[0]): s[1]})
							if 'NaN' not in s[2]:
								# print(d[k_idx]['file'], "- ", k_idx, i)
								traffic = round(float(s[2].split('e')[0]) * (10 ** (int((s[2].split('e')[1])) - 1)), 3)
								dest.update(
									{int(s[0]): traffic}
								)
							else:
								dest.update({int(s[0]): s[1]})
							d[k_idx] = dest
			except KeyError as e:
				print(e, 'for ', d[k_idx], 'not found')

		with open(user_model_pickle, mode='wb') as f:
			pickle._dump(d, f)

	else:
		with open(user_model_pickle, mode='rb') as f:
			d = pickle.load(f)

	# ITERATE OVER ALL POLLED USERS AND APPEND TIMESTAMP DATA

	return d


def user_traffic_visualization(user_dict):
	for k_idx in user_dict:
		if len(user_dict[k_idx]) == (TRAFFIC_DIST_SIZE + 2):
			x_values, y_values = [], []
			dict_graph = user_dict[k_idx]
			del dict_graph['rra']
			del dict_graph['file']
			del dict_graph['product']
			for i, v in sorted(dict_graph.items()):
				x_values.append(datetime.datetime.strptime(
					datetime.datetime.fromtimestamp(i).strftime("%d/%m/%y %H:%M"), "%d/%m/%y %H:%M")
				)
				# x_values.append(int(i))
				try:
					y_values.append(float(v.split('e+')[0]) * (10 ** (int((v.split('e+')[1])) - 1)))
				except:
					# IF COUNT EXCEPTION, INSERT LAST THREE VALUES
					y_values.append(sum(y_values[-3:]) / 3)
			plt.plot(x_values, y_values)
			plt.xlabel('Time Series')
			plt.ylabel('User - %s' % k_idx)
			plt.legend(loc='upper left')
			plt.show()
	return


def user_model_visualization(user_model):
	print((np.sum(user_model[0]) * 300 / 1e6) * 10.8)
	print((np.sum(user_model[1]) * 300 / 1e6) * 10.8)
	print((np.sum(user_model[2]) * 300 / 1e6) * 10.8)
	print((np.sum(user_model[3]) * 300 / 1e6) * 10.8)
	print((np.sum(user_model[4]) * 300 / 1e6) * 10.8)
	return 0


def user_traffic_model(users_dict):
	def _save_traffic_model(user_models):
		with open('user_traffic_dataset.p', mode='wb') as f:
			pickle.dump(user_models, f, pickle.HIGHEST_PROTOCOL)

	# USER TRAFFIC PREDICTION USES k-NN ALGORITHM TO PREDICT USER'S
	# TRAFFIC USAGE FROM N DATA SERIES INPUT.

	X_10 = []
	X_15 = []
	X_25 = []
	X_35 = []
	X_50 = []
	y_10 = []
	y_15 = []
	y_25 = []
	y_35 = []
	y_50 = []

	if not os.path.exists('user_traffic_dataset.p'):

		for k_idx in users_dict:
			if '10240' in users_dict[k_idx]['product']:
				for j_idx in users_dict[k_idx].keys():
					if j_idx not in ['rra', 'product', 'file']:
						if users_dict[k_idx][j_idx] != 'NaN':
							X_10.append([ts2day_integer(j_idx)])
							y_10.append(users_dict[k_idx][j_idx])
			elif '15360' in users_dict[k_idx]['product']:
				for j_idx in users_dict[k_idx].keys():
					if j_idx not in ['rra', 'product', 'file']:
						if users_dict[k_idx][j_idx] != 'NaN':
							X_15.append([ts2day_integer(j_idx)])
							y_15.append(users_dict[k_idx][j_idx])
			elif '25600' in users_dict[k_idx]['product']:
				for j_idx in users_dict[k_idx].keys():
					if j_idx not in ['rra', 'product', 'file']:
						if users_dict[k_idx][j_idx] != 'NaN':
							X_25.append([ts2day_integer(j_idx)])
							y_25.append(users_dict[k_idx][j_idx])
			elif '35840' in users_dict[k_idx]['product']:
				for j_idx in users_dict[k_idx].keys():
					if j_idx not in ['rra', 'product', 'file']:
						if users_dict[k_idx][j_idx] != 'NaN':
							X_35.append([ts2day_integer(j_idx)])
							y_35.append(users_dict[k_idx][j_idx])
			elif '51200' in users_dict[k_idx]['product']:
				for j_idx in users_dict[k_idx].keys():
					if j_idx not in ['rra', 'product', 'file']:
						if users_dict[k_idx][j_idx] != 'NaN':
							X_50.append([ts2day_integer(j_idx)])
							y_50.append(users_dict[k_idx][j_idx])

		_save_traffic_model([[X_10, X_15, X_25, X_35, X_50], [y_10, y_15, y_25, y_35, y_50]])

	else:
		with open('user_traffic_dataset.p', mode='rb') as f:
			user_traffic_data = pickle.load(f)
			X_10 = user_traffic_data[0][0]
			X_15 = user_traffic_data[0][1]
			X_25 = user_traffic_data[0][2]
			X_35 = user_traffic_data[0][3]
			X_50 = user_traffic_data[0][4]
			y_10 = user_traffic_data[1][0]
			y_15 = user_traffic_data[1][1]
			y_25 = user_traffic_data[1][2]
			y_35 = user_traffic_data[1][3]
			y_50 = user_traffic_data[1][4]

	# CROSS VALIDATION

	X_train_10, X_test_10, y_train_10, y_test_10 = train_test_split(X_10, y_10, test_size=0.4)
	X_train_15, X_test_15, y_train_15, y_test_15 = train_test_split(X_15, y_15, test_size=0.4)
	X_train_25, X_test_25, y_train_25, y_test_25 = train_test_split(X_25, y_25, test_size=0.4)
	X_train_35, X_test_35, y_train_35, y_test_35 = train_test_split(X_35, y_35, test_size=0.4)
	X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(X_50, y_50, test_size=0.4)

	# TRAINING
	neighbor_10 = KNeighborsRegressor(n_neighbors=1024)
	neighbor_15 = KNeighborsRegressor(n_neighbors=1024)
	neighbor_25 = KNeighborsRegressor(n_neighbors=1024)
	neighbor_35 = KNeighborsRegressor(n_neighbors=1024)
	neighbor_50 = KNeighborsRegressor(n_neighbors=1024)

	neighbors = [neighbor_10.fit(X=X_train_10, y=y_train_10), neighbor_15.fit(X=X_train_15, y=y_train_15),
				 neighbor_25.fit(X=X_train_25, y=y_train_25), neighbor_35.fit(X=X_train_35, y=y_train_35),
				 neighbor_50.fit(X=X_train_50, y=y_train_50)]

	x_axis = []
	for i in range(TRAFFIC_DIST_SIZE):
		x_axis.append([float(i)])

	predictions = [neighbor_10.predict(X_test_10).reshape(-1, 1),
				   neighbor_15.predict(X_test_15).reshape(-1, 1),
				   neighbor_25.predict(X_test_25).reshape(-1, 1),
				   neighbor_35.predict(X_test_35).reshape(-1, 1),
				   neighbor_50.predict(X_test_50).reshape(-1, 1)]

	scores = [neighbor_10.score(predictions[0], np.array(y_test_10).reshape(-1, 1)),
			  neighbor_15.score(predictions[1], np.array(y_test_15).reshape(-1, 1)),
			  neighbor_25.score(predictions[2], np.array(y_test_25).reshape(-1, 1)),
			  neighbor_35.score(predictions[3], np.array(y_test_35).reshape(-1, 1)),
			  neighbor_50.score(predictions[4], np.array(y_test_50).reshape(-1, 1))]

	print(scores)

	for i, v in enumerate(neighbors):
		plt.subplot(2, 3, i + 1)
		plt.plot(x_axis, v.predict(x_axis), c='g', label='prediction')
	plt.show()

	return [neighbor_10.predict(x_axis).reshape(-1, 1),
			neighbor_15.predict(x_axis).reshape(-1, 1),
			neighbor_25.predict(x_axis).reshape(-1, 1),
			neighbor_35.predict(x_axis).reshape(-1, 1),
			neighbor_50.predict(x_axis).reshape(-1, 1)]


def ts2day_integer(ts):
	# RETURNS INDEX FROM 0 TO 287 TO INDICATE 5 MINUTE PERIOD OF THE DAY
	# 0 -> 00:00, 287-> 23:55

	H = datetime.datetime.fromtimestamp(ts).strftime("%H")
	M = datetime.datetime.fromtimestamp(ts).strftime("%M")

	return int(H) * 12 + int(M) / 5


if __name__ == '__main__':
	main()
