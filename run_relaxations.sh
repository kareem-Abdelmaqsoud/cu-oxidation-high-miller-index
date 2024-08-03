python /home/jovyan/shared-scratch/kabdelma/ocp/main.py --mode run-relaxations \
    --config-yml /home/jovyan/shared-scratch/kabdelma/ocp/configs/oc22/s2ef/gemnet-oc/gemnet_oc_oc20_oc22_degen_edges.yml \
    --checkpoint /home/jovyan/shared-scratch/kabdelma/high_miller_idx/gnoc_oc22_oc20_all_s2ef.pt \
    --model.otf_graph=True \
    --optim.eval_batch_size=32 \
    --identifier high_miller_adsorbml_relax ;