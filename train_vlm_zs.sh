for ratio in 0.2 0.3 0.4 0.5
do
for model in clip_vitb16 clip_resnet50
do
for seed in 0 #1 42
do
    echo $ratio $model $seed 
    CUDA_VISIBLE_DEVICES=$1 python train_vlm.py --model $model --seed $seed --num_classes 25\
    --batch_size 256 --mode zero_shot\
    --data_dir /data/eSkin_Data/All_Images\
    --train_csv_path ./splits/random/test_ratio=${ratio}/train.csv\
    --test_csv_path ./splits/random/test_ratio=${ratio}/test.csv
done
done
done


ratio=999 
for model in clip_vitb16 clip_resnet50 
do
for seed in 0 #1 42
do
    echo $ratio $model $seed 
    CUDA_VISIBLE_DEVICES=$1 python train_vlm.py --model $model --seed $seed --num_classes 25\
    --batch_size 256  --mode zero_shot\
    --data_dir /data/eSkin_Data/All_Images\
    --train_csv_path ./splits/uniform/test_number=5/train.csv\
    --test_csv_path ./splits/uniform/test_number=5/test.csv\
    # --reweight
done
done