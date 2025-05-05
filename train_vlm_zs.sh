test_csv_path=./splits/official_split/test.csv

for model in clip_vitb16 siglip
do
for seed in 0
do  
    # default template
    echo $ratio $model $seed 
    CUDA_VISIBLE_DEVICES=$1 python train_vlm.py --model $model --seed $seed --num_classes 24\
    --batch_size 256 --mode zero_shot\
    --data_dir /data/eSkin_Data/All_Images\
    --train_csv_path ./splits/official_split/train.csv\
    --test_csv_path $test_csv_path --template custom
    echo "done done done"

    # medical template
    echo $ratio $model $seed 
    CUDA_VISIBLE_DEVICES=$1 python train_vlm.py --model $model --seed $seed --num_classes 24\
    --batch_size 256 --mode zero_shot\
    --data_dir /data/eSkin_Data/All_Images\
    --train_csv_path ./splits/official_split/train.csv\
    --test_csv_path $test_csv_path --template default
    echo "done done done"
done
done