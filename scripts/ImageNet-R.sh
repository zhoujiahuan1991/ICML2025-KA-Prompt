

method=KA-Prompt_reproduce/
epochs=5

for seed in  3407 0 42; do
    echo "Running Python script with seed: $seed"  

    CUDA_VISIBLE_DEVICES=4 python main.py --info=seed-${seed}  --dataset=imagenet-r \
    --pool_size=225 --prompt_num=4 --topN=2 --prompt_comp --prompt_per_task=15 --use_prompt_penalty_3  \
    --output_path=Repeat_models/${method}/ImageNet-R/   --fuse_prompt  --use_ema_c --adapt_ema_c --adapt_h=6.5 --lr=0.007 \
    --resize_test=224 --resize_train=224   --greedy_init --tau 0.01 --aux_weight 0.1 --seed=${seed} \
    --query_pos -1 --n_epochs=${epochs}
done


python cal_mean_std_af.py Repeat_models/${method}/ImageNet-R/


