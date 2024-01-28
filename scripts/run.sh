for m in "seq2seqHD" "AZAZ"; do
    CUDA_VISIBLE_DEVICES=0 python3 -u main.py \
    --method $m \
    --root_path ./data/ \
    --test_bsz 1 \
    --data Exchange \
    --seq_len 6 \
    --pred_len 3 \
    --learning_rate 1e-3 \
    --l2_lambda 2e-3 \
    --label_len 6 \
    --hvs_len 1000
done
