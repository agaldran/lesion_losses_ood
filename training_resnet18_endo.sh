#########################################################################################################################################
# RESNET-18
# FOLD1
# CE/DICE alone
python train_losses.py --save_path endotect/resnet18_only_ce_f1    --csv_train data_endotect/train_f1.csv --loss1 ce    --mixture only_loss1 --model_name fpnet_resnet18_W
python train_losses.py --save_path endotect/resnet18_only_dice_f1  --csv_train data_endotect/train_f1.csv --loss1 dice  --mixture only_loss1 --model_name fpnet_resnet18_W
#CE + DICE
python train_losses.py --save_path endotect/resnet18_ce_combo_dice_f1 --csv_train data_endotect/train_f1.csv    --loss1 ce --loss2 dice --mixture combo --model_name fpnet_resnet18_W
python train_losses.py --save_path endotect/resnet18_ce_linear_dice_f1 --csv_train data_endotect/train_f1.csv   --loss1 ce --loss2 dice --mixture linear --model_name fpnet_resnet18_W
python train_losses.py --save_path endotect/resnet18_ce_finetune_dice_f1 --csv_train data_endotect/train_f1.csv --loss1 ce --loss2 dice --mixture fine_tune_loss2 --model_name fpnet_resnet18_W

# FOLD2
# CE/DICE alone
python train_losses.py --save_path endotect/resnet18_only_ce_f2    --csv_train data_endotect/train_f2.csv --loss1 ce    --mixture only_loss1 --model_name fpnet_resnet18_W
python train_losses.py --save_path endotect/resnet18_only_dice_f2  --csv_train data_endotect/train_f2.csv --loss1 dice  --mixture only_loss1 --model_name fpnet_resnet18_W
#CE + DICE
python train_losses.py --save_path endotect/resnet18_ce_combo_dice_f2 --csv_train data_endotect/train_f2.csv    --loss1 ce --loss2 dice --mixture combo --model_name fpnet_resnet18_W
python train_losses.py --save_path endotect/resnet18_ce_linear_dice_f2 --csv_train data_endotect/train_f2.csv   --loss1 ce --loss2 dice --mixture linear --model_name fpnet_resnet18_W
python train_losses.py --save_path endotect/resnet18_ce_finetune_dice_f2 --csv_train data_endotect/train_f2.csv --loss1 ce --loss2 dice --mixture fine_tune_loss2 --model_name fpnet_resnet18_W

# FOLD3
# CE/DICE alone
python train_losses.py --save_path endotect/resnet18_only_ce_f3    --csv_train data_endotect/train_f3.csv --loss1 ce    --mixture only_loss1 --model_name fpnet_resnet18_W
python train_losses.py --save_path endotect/resnet18_only_dice_f3  --csv_train data_endotect/train_f3.csv --loss1 dice  --mixture only_loss1 --model_name fpnet_resnet18_W
#CE + DICE
python train_losses.py --save_path endotect/resnet18_ce_combo_dice_f3 --csv_train data_endotect/train_f3.csv    --loss1 ce --loss2 dice --mixture combo --model_name fpnet_resnet18_W
python train_losses.py --save_path endotect/resnet18_ce_linear_dice_f3 --csv_train data_endotect/train_f3.csv   --loss1 ce --loss2 dice --mixture linear --model_name fpnet_resnet18_W
python train_losses.py --save_path endotect/resnet18_ce_finetune_dice_f3 --csv_train data_endotect/train_f3.csv --loss1 ce --loss2 dice --mixture fine_tune_loss2 --model_name fpnet_resnet18_W

# FOLD4
# CE/DICE alone
python train_losses.py --save_path endotect/resnet18_only_ce_f4    --csv_train data_endotect/train_f4.csv --loss1 ce    --mixture only_loss1 --model_name fpnet_resnet18_W
python train_losses.py --save_path endotect/resnet18_only_dice_f4  --csv_train data_endotect/train_f4.csv --loss1 dice  --mixture only_loss1 --model_name fpnet_resnet18_W
# CE + DICE
python train_losses.py --save_path endotect/resnet18_ce_combo_dice_f4 --csv_train data_endotect/train_f4.csv    --loss1 ce --loss2 dice --mixture combo --model_name fpnet_resnet18_W
python train_losses.py --save_path endotect/resnet18_ce_linear_dice_f4 --csv_train data_endotect/train_f4.csv   --loss1 ce --loss2 dice --mixture linear --model_name fpnet_resnet18_W
python train_losses.py --save_path endotect/resnet18_ce_finetune_dice_f4 --csv_train data_endotect/train_f4.csv --loss1 ce --loss2 dice --mixture fine_tune_loss2 --model_name fpnet_resnet18_W

# FOLD5
# CE/DICE alone
python train_losses.py --save_path endotect/resnet18_only_ce_f5    --csv_train data_endotect/train_f5.csv --loss1 ce    --mixture only_loss1 --model_name fpnet_resnet18_W
python train_losses.py --save_path endotect/resnet18_only_dice_f5  --csv_train data_endotect/train_f5.csv --loss1 dice  --mixture only_loss1 --model_name fpnet_resnet18_W
# CE + DICE
python train_losses.py --save_path endotect/resnet18_ce_combo_dice_f5 --csv_train data_endotect/train_f5.csv    --loss1 ce --loss2 dice --mixture combo --model_name fpnet_resnet18_W
python train_losses.py --save_path endotect/resnet18_ce_linear_dice_f5 --csv_train data_endotect/train_f5.csv   --loss1 ce --loss2 dice --mixture linear --model_name fpnet_resnet18_W
python train_losses.py --save_path endotect/resnet18_ce_finetune_dice_f5 --csv_train data_endotect/train_f5.csv --loss1 ce --loss2 dice --mixture fine_tune_loss2 --model_name fpnet_resnet18_W

#########################################################################################################################################




































