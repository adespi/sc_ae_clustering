import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


def main(x, cell_types):
    genes_to_drop = np.where(x.values.sum(axis=1)==0)[0]
    x = x.drop(genes_to_drop, 0)
    individuals = np.array([int(cell.split(".")[1]) for cell in x.columns.to_list()])
    means=np.zeros([len(set(individuals)), len(set(cell_types)), x.shape[0]])
    for cell_type in set(cell_types):
        for individual in (set(individuals)):
            means[individual-1, cell_type] = np.mean(x[x.columns[(cell_types == cell_type) * (individuals == individual)]],axis =1)
    
    
    np.nan_to_num(means, copy =False)
    means = means/np.repeat(means.mean(axis = 2)[..., np.newaxis], means.shape[2], axis =2)
    np.nan_to_num(means, copy =False)
    
    q_values_array = []
    for cell_type in set(cell_types) :
        p_values = ttest_ind(means[:,cell_type], np.delete(means, cell_type, axis=1).reshape(-1, x.shape[0]))[1]
        q_values = multipletests(p_values)[1]
        print(sum(q_values < 0.01))
        q_values_array.append(q_values)# < 0.01)
    
    q_values_array = np.array(q_values_array)
    outstanding_genes = q_values_array<0.001
    
    plt.imshow(1/q_values_array.T, aspect = 7/17926)
    plt.show()
    
    plt.imshow(outstanding_genes.T, aspect = 7/17926)
    plt.show()
    
    outstanding_genes = outstanding_genes/np.sum(outstanding_genes,axis=0)
    np.nan_to_num(outstanding_genes, copy =False)
    plt.imshow(outstanding_genes.T, aspect = 7/17926)
    plt.show()
    
    colAnnotations = pd.read_csv("sc_data/gene_names.csv")['x']
    colAnnotations = colAnnotations.drop(genes_to_drop, 0)
    
    
    important_genes = []
    for group in set(cell_types) :
        important_genes.append(np.array([colAnnotations[outstanding_genes[group]==1], q_values_array[group,outstanding_genes[group]==1]]).T)
    
    genes = np.concatenate(important_genes)[:,0]
    gene_idx = np.concatenate([np.where(colAnnotations.values == genes[i])[0] for i in range(len(genes))])
    
    
    print(important_genes)



if __name__ == "__main__":
    main(x = pd.read_feather("sc_data/snRNA_AD_brain.feather"), cell_types = np.load("outputs npy/vae/y_pred_vae_mu_no_gn.npy"))