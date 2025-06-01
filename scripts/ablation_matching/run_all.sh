#! /bin/bash

# Script to run all matching ablation experiments
# This script combines the functionality of:
# - sift_baseline.sh
# - sift_ours.sh
# - mast3r_baseline.sh
# - mast3r_ours.sh

python src/structure_from_motion.py -m \
    data=v2_10scenes_2 \
    cost=ba \
    sfm/feature_matching=sift \
    data.num_images=15

python src/structure_from_motion.py -m \
    data=v2_10scenes_2 \
    cost=combined_similar_scale_conf \
    sfm/feature_matching=sift \
    data.num_images=15 \
    cost.dust3r_weight=0.005 \
    sfm.reconstruction.global_optimization.rigid_alignment_ransac=True \
    sfm.reconstruction.global_optimization.ransac_threshold=0.1 \
    sfm.reconstruction.global_optimization.ransac_iterations=100 \
    sfm.reconstruction.global_optimization.pointmaps_align_only_inliers=True \
    sfm.reconstruction.global_optimization.use_confmaps=True

python src/structure_from_motion.py -m \
    data=v2_10scenes_2 \
    cost=ba \
    sfm/feature_matching=mast3r \
    data.num_images=15 \
    hparams=mast3r

python src/structure_from_motion.py -m \
    data=v2_10scenes_2 \
    cost=combined_similar_scale_conf \
    sfm/feature_matching=mast3r \
    model=dust3r_512_dpt \
    data.num_images=15 \
    cost.dust3r_weight=0.01 \
    sfm.reconstruction.global_optimization.rigid_alignment_ransac=True \
    sfm.reconstruction.global_optimization.ransac_threshold=0.1 \
    sfm.reconstruction.global_optimization.pointmaps_align_only_inliers=True \
    sfm.reconstruction.global_optimization.use_confmaps=True \
    hparams=mast3r
