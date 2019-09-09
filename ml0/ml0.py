import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import operator

def prt_stats(attr,df):
	min = df[attr].min()
	max = df[attr].max()
	mean = df[attr].mean()

	print("MIN:",min)
	print("MAX:",max)
	print("AVG",mean)

def plot_rate(attr,df):
	d = {a:0 for a in range(6)}

	for index, row in df.iterrows():
		d[math.trunc(row[attr])] += 1

	objects = ('< 1', '< 2', '< 3', '< 4', '< 5', '5')
	y_pos = np.arange(len(objects))
	val = list(d.values())

	plt.bar(y_pos, val, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)

	plt.show()

def plot_all_vals(attr,df):
	d = {}

	for index, row in df.iterrows():
		key = row[attr]
		if key in d:
			d[key] += 1
		else:
			d[key] = 1

	objects = list(d.keys())
	y_pos = np.arange(len(objects))
	val = list(d.values())

	plt.bar(y_pos, val, align='center', alpha=0.5)
	# plt.xticks(y_pos, objects, rotation='vertical')
	plt.xticks(y_pos, objects)

	plt.show()

def plot_size(attr,df):
	d = {a:0 for a in range(7)} # >1 >500 >250 >125 >50 >25 <25

	for index, row in df.iterrows():
		cur_val = row[attr] / 1024 / 1024
		if cur_val < 25:
			d[0] += 1
		elif cur_val < 50:
			d[1] += 1
		elif cur_val < 125:
			d[2] += 1
		elif cur_val < 250:
			d[3] += 1
		elif cur_val < 500:
			d[4] += 1
		elif cur_val < 1024:
			d[5] += 1
		else:
			d[6] += 1

	objects = ('< 25Mb', '< 50Mb', '< 125Mb', '< 250Mb', '< 500Mb', '< 1Gb', '> 1Gb')
	y_pos = np.arange(len(objects))
	val = list(d.values())

	plt.bar(y_pos, val, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)

	plt.show()

def plot_price(attr,df):
	d = {a:0 for a in range(7)} # free, >1, >2, >3, >4, >5, <5

	for index, row in df.iterrows():
		cur_val = row[attr]
		if cur_val == 0:
			d[0] += 1
		elif cur_val < 1:
			d[1] += 1
		elif cur_val < 2:
			d[2] += 1
		elif cur_val < 3:
			d[3] += 1
		elif cur_val < 4:
			d[4] += 1
		elif cur_val < 5:
			d[5] += 1
		else:
			d[6] += 1

	objects = ('free', '< 1$', '< 2$', '< 3$', '< 4$', '< 5$', '> 5$')
	y_pos = np.arange(len(objects))
	val = list(d.values())

	plt.bar(y_pos, val, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)

	plt.show()


def plot_rcv(attr,df):
	d = {a:0 for a in range(7)} 
	max1 = df[attr].max()
	for index, row in df.iterrows():
		cur_val = row[attr]
		if cur_val == 0:
			d[0] += 1
		elif cur_val < 5:
			d[1] += 1
		elif cur_val < 25:
			d[2] += 1
		elif cur_val < 100:
			d[3] += 1
		elif cur_val < 250:
			d[4] += 1
		elif cur_val < 500:
			d[5] += 1
		else:
			d[6] += 1

	objects = ('0', '< 5', '< 25', '< 100', '< 250', '< 500', '> 500')
	y_pos = np.arange(len(objects))
	val = list(d.values())

	plt.bar(y_pos, val, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)

	plt.show()

def plot_rct(attr,df):
	d = {a:0 for a in range(7)} # 0, >1, >2, >3, >4, >5, < 1kk
	max1 = df[attr].max()
	for index, row in df.iterrows():
		cur_val = row[attr]
		if cur_val == 0:
			d[0] += 1
		elif cur_val < 100:
			d[1] += 1
		elif cur_val < 500:
			d[2] += 1
		elif cur_val < 2000:
			d[3] += 1
		elif cur_val < 7500:
			d[4] += 1
		elif cur_val < 20000:
			d[5] += 1
		else:
			d[6] += 1

	objects = ('0', '< 5', '< 25', '< 100', '< 250', '< 500', '> 500')
	y_pos = np.arange(len(objects))
	val = list(d.values())

	plt.bar(y_pos, val, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)

	plt.show()

def plot_nsd(attr,df): # number of supporting devices
	d = {a:0 for a in range(3)} # 0, >1, >2, >3, >4, >5, < 1kk
	max1 = df[attr].max()
	for index, row in df.iterrows():
		cur_val = row[attr]
		if cur_val < 35:
			d[0] += 1
		elif cur_val < 40:
			d[1] += 1
		else:
			d[2] += 1

	objects = ('< 35', '< 40', '> 40')
	y_pos = np.arange(len(objects))
	val = list(d.values())

	plt.bar(y_pos, val, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)

	plt.show()

def plot_isu(attr,df):
	d = {a:0 for a in range(6)}

	for index, row in df.iterrows():
		d[math.trunc(row[attr])] += 1

	objects = ('0', '1', '2', '3', '4', '5')
	y_pos = np.arange(len(objects))
	val = list(d.values())

	plt.bar(y_pos, val, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)

	plt.show()

def plot_ln(attr,df):
	d = {a:0 for a in range(7)} # 0, >1, >2, >3, >4, >5, < 1kk
	max1 = df[attr].max()
	for index, row in df.iterrows():
		cur_val = row[attr]
		if cur_val < 2:
			d[0] += 1
		elif cur_val < 3:
			d[1] += 1
		elif cur_val < 5:
			d[2] += 1
		elif cur_val < 10:
			d[3] += 1
		elif cur_val < 15:
			d[4] += 1
		elif cur_val < 20:
			d[5] += 1
		else:
			d[6] += 1

	objects = ('< 2', '< 3', '< 5', '< 10', '< 15', '< 20', '> 20')
	y_pos = np.arange(len(objects))
	val = list(d.values())

	plt.bar(y_pos, val, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)

	plt.show()

def plot_lic(attr,df):
	d = {0:0, 1:0} # 0, >1, >2, >3, >4, >5, < 1kk
	for index, row in df.iterrows():
		d[row[attr]] += 1

	objects = ('No', 'Yes')
	y_pos = np.arange(len(objects))
	val = list(d.values())

	plt.bar(y_pos, val, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)

	plt.show()

def plot_genre(attr,df):
	d = {}

	for index, row in df.iterrows():
		key = row[attr]
		if key in d:
			d[key] += 1
		else:
			d[key] = 1
	c = 0
	keys = []
	vals = []

	for k,v in sorted(d.items(), key=operator.itemgetter(1) ,reverse=True):
    
	    keys.append(k)
	    vals.append(v)
	    c += 1
	    if c == 5:
	        break

	y_pos = np.arange(len(keys))

	plt.bar(y_pos, vals, align='center', alpha=0.5)
	# plt.xticks(y_pos, objects, rotation='vertical')
	plt.xticks(y_pos, keys)

	plt.show()

def plot(attr,df):
	if attr == 'user_rating' or attr == 'user_rating_ver':
		plot_rate(attr,df)
	elif attr == 'size_bytes':
		plot_size(attr,df)
	elif attr == 'currency' or attr == "cont_rating":
		plot_all_vals(attr,df)
	elif attr == "prime_genre":
		plot_genre(attr,df)
	elif attr == "price":
		plot_price(attr,df)
	elif attr == 'rating_count_ver':
		plot_rcv(attr,df)
	elif attr == 'rating_count_tot':
		plot_rct(attr,df)
	elif attr == "sup_devices.num":
		plot_nsd(attr,df)
	elif attr == "ipadSc_urls.num":
		plot_isu(attr,df)
	elif attr == "lang.num":
		plot_ln(attr,df) 
	elif attr == "vpp_lic":
		plot_lic(attr,df)
	else:
		print("Bad attribute :(")


df = pd.read_csv(r'/home/stdstring/ML/src/AppleStore.csv')
attr = "lang.num"
# print(df[[attr]])

prt_stats(attr,df)
plot(attr,df)
