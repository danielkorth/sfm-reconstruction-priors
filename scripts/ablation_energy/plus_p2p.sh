#! /bin/bash

python src/structure_from_motion.py -m \
    data=v2_10scenes_2 \
    cost=combined_similar_scale \
    sfm/feature_matching=mast3r \
    model=dust3r_512_dpt \
    data.num_images=15 \
    cost.dust3r_weight=0.01 \
    hparams=mast3r
