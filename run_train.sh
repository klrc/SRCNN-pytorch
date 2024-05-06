python train.py --train-file "assets/91-image_x3.h5" \
                --eval-file "assets/Set5_x3.h5" \
                --outputs-dir "outputs" \
                --scale 3 \
                --lr 1e-4 \
                --batch-size 128 \
                --num-epochs 400 \
                --num-workers 8 \
                --seed 123    