import pandas as pd
import numpy as np
import re
import operator
from pyparsing import *
import matplotlib.pyplot as plt

words = {}

filter = ["the","to","a","and","of","a",
"in","with","for","is","on","or","from",
"as","this","by","that","are","it","will",
"at","an","if"]

df = pd.read_csv("src/appleStore_description.csv",engine='python',error_bad_lines=False)

words = {}
attr = "app_desc"
for index, row in df.iterrows():

    st = row[attr]
    st = re.sub('[&!@*,-.0\[\]•123456789#$]', '', st)
    st = st.lower()

    plot = st.split()


    for wrd in plot:

	    if wrd in words:
	        words[wrd] += 1
	    else:
	        words[wrd] = 1

keys=[]
vals=[]
c = 0

for k,v in sorted(words.items(), key=operator.itemgetter(1) ,reverse=True):
    
    if k in filter:
        continue
    keys.append(k)
    vals.append(v)
    # print(str(v) + " → " + k)
    c += 1
    if c == 10:
        break

y_pos = np.arange(len(vals))
plt.bar(y_pos, vals, align='center', alpha=0.5)
plt.xticks(y_pos, keys)
plt.ylabel('Word')
plt.title('Word Countage')

plt.show()