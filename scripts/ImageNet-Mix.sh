method=KA-Prompt_reproduce/
epochs=5

for seed in  3407 0 42; do
    echo "Running Python script with seed: $seed"  
    CUDA_VISIBLE_DEVICES=5 python main.py --info=seed-${seed} --dataset=imagenet-cr \
    --pool_size=750 --prompt_num=4 --topN=3 --prompt_comp --prompt_per_task=25 --use_prompt_penalty_3 \
    --fuse_prompt --output_path=Repeat_models/${method}/ImageNet-Mix/  --use_ema_c --adapt_ema_c --adapt_h=8.7 --lr=0.005 \
    --resize_test=224 --resize_train=224 --greedy_init  --tau 0.01 --aux_weight  1.0 --seed=${seed} \
    --query_pos -1 --n_epochs=${epochs}


done


python cal_mean_std_af.py Repeat_models/${method}/ImageNet-Mix/

