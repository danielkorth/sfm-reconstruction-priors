#! /bin/bash

# Script to run all SfM methods on the dataset
# This script combines the functionality of:
# - colmap.sh
# - baseline.sh
# - ours.sh
# - dust3r_go.sh
# - mast3r_sfm.sh
# - vggt_pred.sh

# python src/sfm_colmap.py -m \
#     data=v2_30scenes \
#     data.num_images=15,20,25

python src/structure_from_motion.py -m \
    data=v2_30scenes \
    data.num_images=15,20,25 \
    sfm/feature_matching=mast3r \
    cost=ba \
    hparams=mast3r

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

python src/sfm_dust3r_go.py -m \
    model=dust3r_512_dpt \
    data=v2_30scenes \
    data.num_images=15,20,25

python src/sfm_dust3r_go.py -m \
    model=mast3r_sfm \
    data=v2_30scenes \
    data.num_images=15,20,25

python src/sfm_vggt.py -m \
    model=vggt_mv \
    data=v2_30scenes \
    data.num_images=15,20,25
