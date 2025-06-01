#! /bin/bash

python src/sfm_vggt.py -m \
    model=vggt_mv \
    data=v2_30scenes \
    data.num_images=15,20,25
