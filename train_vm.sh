# for ratio in 0.2 0.3 0.4 0.5
# do
# for model in  dinov2 #resnet50 vitb16 #resnet18 dinov2
# do
# for seed in 0 #1 42
# do
#     echo $ratio $model $seed 
#     CUDA_VISIBLE_DEVICES=$1 python train_vm.py --model $model --seed $seed --num_classes 30\
#     --batch_size 256 --epochs 200 --lr 0.01\
#     --data_dir /data/eSkin_Data/All_Images\
#     --train_csv_path /media/janet/nips25_benchmark/classification_splits/random/test_ratio=${ratio}/train.csv\
#     --test_csv_path /media/janet/nips25_benchmark/classification_splits/random/test_ratio=${ratio}/test.csv
# done
# done
# done


ratio=999 
for model in dinov2  resnet50 vitb16 #resnet18
do
for seed in 0 #1 42
do
    echo $ratio $model $seed 
    CUDA_VISIBLE_DEVICES=$1 python train_vm.py --model $model --seed $seed --num_classes 30\
    --batch_size 256 --epochs 200 --lr 0.01\
    --data_dir /data/eSkin_Data/All_Images\
    --train_csv_path /media/janet/nips25_benchmark/classification_splits/uniform/test_number=5/train.csv\
    --test_csv_path /media/janet/nips25_benchmark/classification_splits/uniform/test_number=5/test.csv\
    # --reweight
done
done