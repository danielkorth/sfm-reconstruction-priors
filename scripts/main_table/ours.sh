#! /bin/bash

# final parameters for the report
python src/structure_from_motion.py -m \
    data.num_images=15,20,25 \
    data=v2_30scenes \
    cost=combined_similar_scale_conf \
    sfm/feature_matching=mast3r \
    model=dust3r_512_dpt \
    cost.dust3r_weight=0.01 \
    sfm.reconstruction.global_optimization.rigid_alignment_ransac=True \
    sfm.reconstruction.global_optimization.ransac_threshold=0.1 \
    sfm.reconstruction.global_optimization.ransac_iterations=100 \
    sfm.reconstruction.global_optimization.pointmaps_align_only_inliers=True \
    sfm.reconstruction.global_optimization.use_confmaps=True \
    hparams=mast3r
