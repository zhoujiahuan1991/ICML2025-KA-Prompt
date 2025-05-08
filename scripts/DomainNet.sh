
method=KA-Prompt_reproduce/


for seed in    0 42 3407; do
    echo "Running Python script with seed: $seed"    
    CUDA_VISIBLE_DEVICES=6 python main.py --info=seed-${seed} --dataset=domain-net \
    --pool_size=150 --prompt_num=8 --topN=3 --prompt_comp --prompt_per_task=25 --use_prompt_penalty_3 \
    --fuse_prompt --use_ema_c  --output_path=Repeat_models/${method}/DomainNet/ --adapt_ema_c --adapt_h=10 --lr=0.0006  \
    --resize_test=224 --resize_train=224 --greedy_init  --tau 0.01 --aux_weight  0.1 --seed=${seed} --n_epochs 10 \
    --query_pos -1  
   
done

python cal_mean_std_af.py Repeat_models/${method}/DomainNet/
