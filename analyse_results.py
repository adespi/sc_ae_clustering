import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#results = pd.read_pickle('test_results.pkl')
results = pd.read_pickle('test_results_10X_PBMC_select_2100.pkl')
temperatures = sorted(set(results['Temperature']))
reducing_factor = sorted(set(results['reducing_factor']))
methods = ['TSNE kmeans', 'PCA kmeans', 'kmeans', 'argmax']
dfs=[]
#nbr
for x, method in enumerate([methods[2]]):
   #for x, method in enumerate(methods):
   dfs.append(pd.DataFrame(np.zeros([len(reducing_factor), len(temperatures)]), columns=temperatures))
   for i, factor in enumerate(reducing_factor):
      for j, temp in enumerate(temperatures):
         dfs[x].iloc[i, j] = np.max(results[(results['reducing_factor'] == factor) & (results['Temperature'] == temp) & (results['Method'] == method)]['ACC'])
         #print(np.max(results[(results['reducing_factor'] == factor) & (results['Temperature'] == temp)]['ACC']))
      dfs[x].rename(index={i:factor}, inplace=True)

   print(dfs[x])

   #fig, ax = plt.subplots(figsize=(6,6))
   #ax.imshow(dfs[x], interpolation='nearest')#, extent=[1.5,3325,1,0.30])
   plt.imshow(dfs[x], interpolation='nearest')#, extent=[1.5,3325,1,0.30])
   plt.set_cmap('gnuplot')
   #ax.set_aspect(2000)
   plt.colorbar()
   plt.show()


"""
On the output of 10X_PBMC_select_2100 we see two hot spots at [1-2](0.4-0.5)//11(129.7463378906) and 8(0.85)//[11-10](129.7463378906, 86.4975585937)
max is reached at epochs 40 to 50 (actual temp of 0.2 and 15)
python scDeepCluster_tests.py --data_file 10X_PBMC_select_2100.h5 --n_clusters 8 --pretrain_epochs 55 --temp 100 --reducing_factor 0.85 2>/dev/null |grep ACC | cut -f1,2,5,7,9,11 -d" "|tr --delete ,


#dfs[0].iloc[2,11]
#dfs[x].iloc[i, j] = results[(results['reducing_factor'] == reducing_factor[2]) & (results['Temperature'] == temperatures[11]) & (results['Method'] == method)]
see_for_epoch = results[(results['reducing_factor'] == reducing_factor[2]) & (results['Temperature'] == temperatures[11]) & (results['Method'] == method)]
see_for_epoch = results[(results['reducing_factor'] == reducing_factor[8]) & (results['Temperature'] == temperatures[11]) & (results['Method'] == method)]
#plt.plot(see_for_epoch['ACC']*100)
plt.plot(see_for_epoch['Actual Temperature'])
plt.show()
"""


"""
I find everything really unstable
For mouse, their result ACCURACY 0.5481 and 0.6167 after fine-tuning. Our is 
For 10X_PBMC, their result ACCURACY 0.7736 and 0.7626 after fine-tuning. 
"""