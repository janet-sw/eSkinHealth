test_csv_path=./splits/official_split/test.csv

for ratio in 0.01 0.1
do
for seed in 0 42 1234
do
for model in dinov2 resnet50 vitb16 resnet18
do
    echo $ratio $model $seed 
    CUDA_VISIBLE_DEVICES=$1 python train_vm.py --model $model --seed $seed --num_classes 24\
    --batch_size 64 --epochs 200 --lr 0.01 --mode linear\
    --data_dir /data/eSkin_Data/All_Images\
    --train_csv_path ./splits/few_shot/ratio_${ratio}/seed_${seed}/train.csv\
    --test_csv_path $test_csv_path
    echo "done done done"
done
done
done


ratio=1
for model in dinov2 resnet50 vitb16 resnet18
do
for seed in 0 
do
    echo $ratio $model $seed 
    CUDA_VISIBLE_DEVICES=$1 python train_vm.py --model $model --seed $seed --num_classes 24\
    --batch_size 64 --epochs 200 --lr 0.01  --mode linear\
    --data_dir /data/eSkin_Data/All_Images\
    --train_csv_path ./splits/official_split/train.csv\
    --test_csv_path $test_csv_path\
    # --reweight
    echo "done done done"
done
done

ratio=-1
for model in dinov2 resnet50 vitb16 resnet18
do
for seed in 0 
do
    echo $ratio $model $seed 
    CUDA_VISIBLE_DEVICES=$1 python train_vm.py --model $model --seed $seed --num_classes 24\
    --batch_size 64 --epochs 200 --lr 0.01  --mode linear\
    --data_dir /data/eSkin_Data/All_Images\
    --train_csv_path ./splits/official_split/train.csv\
    --test_csv_path $test_csv_path\
    # --reweight
    echo "done done done"
done
done