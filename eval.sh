DATASET=math
MODEL=meta-llama/Llama-2-7b-hf
ADD_SOFT_PROMPT=True
N_PREFIX=3
N_SPECIAL=3
EFFICIENT=lora+prompt-tuning
STEP_TYPE=vae
CUDA_VISIBLE_DEVICES=2 python eval.py \
    --base_model_name_or_path $MODEL \
    --model_name_or_path your_checkpoint \
    --add_soft_prompts $ADD_SOFT_PROMPT\
    --num_general_prefix_tokens $N_PREFIX \
    --num_special_prefix_tokens $N_SPECIAL \
    --parameter_efficient_mode $EFFICIENT \
    --dataset $DATASET \
    --batch_size 2 \
    --max_length 1024 \
    --seed 100 \
    --extract_step_type_tokens $STEP_TYPE \
    --embedding_model_name $MODEL \
    --num_plan_types 5 \
    --num_test 1000 \
    --load_in_8bit True \
    # --prompt_template alpaca \
    # --use_calculator True \