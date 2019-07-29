: "
cd /data1/antoine/scDeepCluster/code/
temp
temp
temp --n_clusters 8 --pretrain_epochs 400
temp --n_clusters 8 --pretrain_epochs 100 --ae_weights ae_weights.h5
tempCluster_model.png results/
temp
temp
temp
echo 'temp epo ACC NMI ARI LOSS' &&
for epo in `seq 10 20 150`; do
    for t in `seq 10 20 150`; do
        echo -n $t $epo ' '
        python scDeepCluster.py --data_file 10X_PBMC.h5 --n_clusters 8 --pretrain_epochs $epo --temp $t 2>/dev/null |grep Final
    done #| cut -f1,2,5,7,9,11 -d" "|tr --delete ,
done
"

: '
ssh -X antoine@KellisGPUCompute2.local
cd /data1/antoine/scDeepCluster/code/
conda activate connect5
bash scDeepCluster_tests.bash
'

if [ -e test_results.pkl ]
then
    read -p "test_results.pkl already exists, continue ? " -n 1 -r
    echo    # (optional) move to a new line
else
    REPLY=Y
fi

if [[ $REPLY =~ ^[Yy]$ ]]
then
    #echo 'temp epo ACC NMI ARI LOSS'> output_results &&
    for factor in 0.3 0.4 0.5 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 ; do # 0.3 0.4 0.5 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do
        for i in `seq 1 20`; do
            t=`echo "scale=10; 1.5 ^ $i" | bc`
            #python scDeepCluster_tests.py --data_file 10X_PBMC_select_2100.h5 --n_clusters 8 --pretrain_epochs 150 --temp $t --reducing_factor $factor --save_results True #2>/dev/null |grep Final | cut -f1,2,5,7,9,11 -d" "|tr --delete ,
            python scDeepCluster_tests.py --data_file mouse_bladder_cell_select_2100.h5 --n_clusters 16 --pretrain_epochs 150 --temp $t --reducing_factor $factor --save_results True #2>/dev/null |grep Final | cut -f1,2,5,7,9,11 -d" "|tr --delete ,
            #python scDeepCluster_tests.py --data_file mouse_ES_cell_select_2100.h5 --n_clusters 4 --pretrain_epochs 150 --temp $t --reducing_factor $factor --save_results True #2>/dev/null |grep Final | cut -f1,2,5,7,9,11 -d" "|tr --delete ,
            #python scDeepCluster_tests.py --data_file worm_neuron_cell_select_2100.h5 --n_clusters 10 --pretrain_epochs 150 --temp $t --reducing_factor $factor --save_results True #2>/dev/null |grep Final | cut -f1,2,5,7,9,11 -d" "|tr --delete ,
        done
    done #>> output_results
fi

