python -u -m torch.distributed.launch --nproc_per_node=2 /home/jovyan/shared-scratch/kabdelma/ocp/main.py --mode train \
    --config-yml /home/jovyan/shared-scratch/kabdelma/ocp/configs/oc22/s2ef/gemnet-oc/gemnet_oc_oc20_oc22_degen_edges.yml \
    --checkpoint /home/jovyan/shared-scratch/kabdelma/high_miller_idx/gnoc_oc22_oc20_all_s2ef.pt \
    --optim.lr_initial=1.e-6 \
    --optim.batch_size=4 \
    --optim.eval_batch_size=4 \
    --optim.eval_every=100 \
    --optim.num_epochs=20 \
    --identifier hif_goc_all_oc20_oc22_finetune_lr1e_6_adslabs_slabs \
    --distributed;

