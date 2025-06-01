#! /bin/bash

python src/sfm_dust3r_go.py -m \
    model=mast3r_sfm \
    data=v2_30scenes \
    data.num_images=15,20,25
