python  /home/jovyan/shared-scratch/kabdelma/ocp/main.py --mode predict \
    --config-yml /home/jovyan/shared-scratch/kabdelma/ocp/configs/oc22/s2ef/gemnet-oc/gemnet_oc_oc20_oc22_degen_edges.yml \
    --checkpoint /home/jovyan/shared-scratch/kabdelma/high_miller_idx/checkpoints/2024-06-18-20-01-04-hif_goc_all_oc20_oc22_finetune_lr1e_6_adslabs_slabs/best_checkpoint.pt \
    --optim.eval_batch_size=4 \
    --identifier hif_goc_inference_all_oc20_oc22_slab_adslabs_finetuned;

    # -u -m torch.distributed.launch --nproc_per_node=2 --distributed