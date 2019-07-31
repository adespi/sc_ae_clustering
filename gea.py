import scanpy as sc
import numpy as np
import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt
from gseapy.parser import Biomart
from gseapy.plot import barplot, dotplot
from gseapy.plot import gseaplot
import os

# load data
group = np.load("results_brain/ae/y_pred.npy")
adata = pd.read_feather("sc_data/snRNA_AD_brain.feather").T
adata.columns=pd.read_csv("sc_data/gene_names.csv")['x']

adata = sc.AnnData(adata)
adata.obs["group"]=group.astype(np.str)

# rank gene by importance for clusters
sc.tl.rank_genes_groups(adata, "group", n_genes = 500)

r = adata.uns['rank_genes_groups']['names']
#pd.DataFrame.from_records(r).to_csv('marker_genes_group.csv')
#", ".join(pd.DataFrame.from_records(r)['1'].values)


for x in pd.DataFrame.from_records(r).columns:
    print("group :"+x, end = "\r")
    # rank gene by importance for clusters
    glist = pd.DataFrame.from_records(r)[x].tolist()
    bm = Biomart()
    if not os.path.exists("test"):
        os.makedirs("test")
    results = bm.query(dataset='hsapiens_gene_ensembl',
                    attributes=['external_gene_name', 'go_id'],
                    filters={'hgnc_symbol': glist},
                    # save output file
                    filename="test/query_"+x+".results.txt")

    enr = gp.enrichr(gene_list=glist,
                    description='test_name',
                    gene_sets=['KEGG_2016'],
                    outdir="test/enrichr_kegg_group"+x,
                    cutoff=0.5 # test dataset, use lower value from range(0,1)
                    )


    # to save your figure, make sure that ``ofname`` is not None
    #barplot(enr.res2d,title='',ofname="test/enrichr_kegg_group"+x+"/bar_plot.pdf")
    dotplot(enr.res2d, title='',ofname="test/enrichr_kegg_group"+x+"/dot_plot.pdf")


    #pd.DataFrame.from_records(adata.uns['rank_genes_groups']['scores'])[x]
    rnk= pd.concat([pd.DataFrame.from_records(r)[x],pd.DataFrame.from_records(adata.uns['rank_genes_groups']['scores'])[x]],axis=1)
    rnk.columns=[0,1]

    pre_res = gp.prerank(rnk=rnk, gene_sets='KEGG_2016',
                        processes=4,
                        permutation_num=100, # reduce number to speed up testing
                        outdir='test/prerank_report_kegg_group_'+x, format='pdf')

    #terms = pre_res.res2d.index

    # to save your figure, make sure that ofname is not None
    #gseaplot(rank_metric=pre_res.ranking, term=terms[0], **pre_res.results[terms[0]],ofname="test/enrichr_kegg_group"+x+"/prerank.pdf")


    #gs_res = gp.gsea(data=pd.DataFrame.from_records(r), # or data='./P53_resampling_data.txt'
    #                 gene_sets='KEGG_2016', # enrichr library names
    #                 cls= './data/P53.cls', # cls=class_vector
    #                 # set permutation_type to phenotype if samples >=15
    #                 permutation_type='phenotype')



