

method=KA-Prompt_reproduce/
epochs=5

for seed in  3407 0 42; do
    echo "Running Python script with seed: $seed"  

    CUDA_VISIBLE_DEVICES=7 python main.py --info=seed-${seed} --dataset=imagenet-c \
    --pool_size=15 --prompt_num=20 --topN=1 --prompt_comp --prompt_per_task=1 --use_prompt_penalty_3  \
    --output_path=Repeat_models/${method}/ImageNet-C/   --fuse_prompt  --use_ema_c --adapt_ema_c --adapt_h=6.5  --lr=0.004 \
    --resize_test=230 --resize_train=230  --greedy_init  --tau 0.01 --aux_weight 1.0 --seed=${seed} \
    --query_pos -1 --n_epochs=${epochs}
done

python cal_mean_std_af.py Repeat_models/${method}/ImageNet-C/
