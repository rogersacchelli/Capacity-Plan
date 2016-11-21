import numpy as np
import matplotlib.pyplot as plt
import time
import math
from sklearn.neighbors.kde import KernelDensity

__doc__ = """




                                         +---------+      10G
                                         |  METRO  +-----------------+                                +--------+
  +--------+      1G                     |   ARD   |                 |                       20G      |        |
  |  MSAN  +---------------+             +---------+                 |                    +-----------+  BRAS  |
  +--------+               |                  |                      |                    |-----------+        |
                           |                  |                      |                    ||          +--------+
                      +---------+    10G      |                +-----------+      +--------------+         |
                      |  METRO  +-------------+                |           | 10G  |              |         |
                      |   ARD   +-------------+                |   METRO   +------+  ROUTER AGG  |         |
                      +---------+    10G      |                |  HEADEND  +------+              |         |
  +--------+               |                  |                |           | 10G  |              |         |
  |  MSAN  +---------------+                  |                +-----------+      +--------------+         |
  +--------+      1G                          |                      |                    ||               |
                                              |                      |                    ||          +---------+
                                         +---------+                 |                    |-----------+         |
                                         |  METRO  |      10G        |                    +-----------+  RDIST  |
                                         |   ARD   +-----------------+                        20G     |         |
                                         +---------+                                                  +---------+




"""

# STATS FOR PRODUCT
MEAN_PRODUCT = 19.714
SIGMA_PRODUCT = 2.3981

# ###### AGGREGATED NETWORK
# STATS FOR ACCESS NETWORK
MEAN_MSAN = 186.63          # AVG NUMBER OF USERS
SIGMA_MSAN = 90.16          # STD DEVIATION
NETWORK_SIZE = 1        # ACCESS NETWORK SIZE
MAX_USERS = 720             # MSAN CAPACITY USERS
# ---------------
# TRAFFIC_DIST_SIZE = 288    # NUMBER OF TRAFFIC ELEMENTS FOR EACH 5 MINUTES AVG FROM 0h TO 24h
TRAFFIC_DIST_SIZE = 48      # NUMBER OF TRAFFIC ELEMENTS FOR EACH 5 MINUTES AVG FROM 20h TO 24h
USER_TRAFFIC_MEAN = 0.8     # AVG TRAFFIC DURING 20h TO 24h IN Kbps
USER_TRAFFIC_SIGMA = 0.7    # STD DEV FOR USER TRAFFIC 20h TO 24h IN Kbps
# -----------------

METRO_ETH_RING_MEAN = 2654.44        # USERS AT METRO ETHERNET NETWORK MEAN
METRO_ETH_RING_SIGMA = 2196.75       # STD DEVIATION
# -----------------
METRO_ETH_ARD_MEAN = 604.0        # USERS AT METRO ETHERNET NETWORK MEAN
METRO_ETH_ARD_SIGMA = 282.0       # STD DEVIATION
# ########################


def main():

	start_time = time.time()

	net_array = np.ndarray(shape=(NETWORK_SIZE, MAX_USERS, TRAFFIC_DIST_SIZE),dtype=np.float32)

	# initialize access users
	for i in range(NETWORK_SIZE):
		num_of_users = get_normal_dist(MEAN_MSAN, SIGMA_MSAN)
		for j in range(MAX_USERS):
			if j <= num_of_users:
				net_array[i, j, :] = get_normal_dist(USER_TRAFFIC_MEAN, USER_TRAFFIC_SIGMA, TRAFFIC_DIST_SIZE)
			else:
				net_array[i, j, :] = 0
		np.random.shuffle(net_array[i, :, :])

	# get bras interface list

	bras_list = get_bras_int_data_list()

	x_axis = np.linspace(0, 5 * TRAFFIC_DIST_SIZE,TRAFFIC_DIST_SIZE)

	# plt.plot(x_axis,net_array[0, 0, :],x_axis,net_array[0, 1, :], label = 'User Traffic')
	# plt.legend(loc='upper left')
	# plt.ylabel('Tráfego [Mbps]')
	# plt.xlabel('Período [min]')
	# plt.show()

	end_time = time.time()

	print ("-------- Total Time: %02f -------" % (end_time - start_time))


def get_normal_dist(MEAN, SIGMA, SIZE=1):
	return np.round(abs(np.random.normal(MEAN, SIGMA, SIZE)))

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

if __name__ == '__main__':
	main()