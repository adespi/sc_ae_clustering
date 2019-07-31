# sc_ae_clustering
This model is an autoencoder. It has three different versions : AE, VAE and IAF

To run all the model on data, use this command
```for x in ae vae iaf; do python scDeepCluster_broad_data_clean.py --n_clusters 15 --data_file sc_data/snRNA_AD_brain.feather --pretrain_epochs 100 --batch_size 50 --model $x; done```

To run without plotting T-SNE : `--plot_TNSE False`
To reuse existing model : `--weights results/ae/weights.h5`