# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:06:20 2021

@author: Sathiya vigraman M
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Association rules

with open('groceries.csv') as G:
    groceries = G.read()

groceries = groceries.split('\n')

gr_list = []

for i in groceries:
    gr_list.append(i.split(','))

all_gr_list = [i for item in gr_list for i in item]

from collections import Counter

item_freq = Counter(all_gr_list)

item_freq = sorted(item_freq.items(), key = lambda x:x [1], reverse=True)


item = [i[0] for i in item_freq]

freq = [i[1] for i in item_freq]

plt.figure(figsize=(15, 15))
plt.bar(item[:10], freq[:10])

gr_df = pd.DataFrame(pd.Series(gr_list), columns = ['groceries'])

gr_df = gr_df.iloc[:9835, ] #last row is empty so we removed it

X = gr_df['groceries'].str.join(sep='*').str.get_dummies(sep='*')

#pip install mlxtend

from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(X, min_support=0.005, max_len=3,use_colnames = True)

frequent_itemsets.head()
frequent_itemsets.sort_values('support', ascending=False, inplace=True)

plt.plot(frequent_itemsets.iloc[:10, :])

ass_rules = association_rules(frequent_itemsets, metric = 'lift')
ass_rules.columns
ass_rules.head()
#antecedents - item 1
#consequents - item 2

#support - high buy
pd.set_option('display.max_columns', None)

ass_rules.sort_values('lift', ascending=False).head()














