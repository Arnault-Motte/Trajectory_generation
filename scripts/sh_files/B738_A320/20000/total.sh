#creat dir 
mkdir -p data_orly/src/generation/models/saved_weights/limited_one_typecode/B738_A320
mkdir -p data_orly/src/generation/models/saved_weights/limited_one_typecode/B738_A320/10000
# mkdir -p data_orly/data_orly/scn/B738_A320/100

#training

uv run python data_orly/scripts/script_train/train_CVAE.py --cond_pseudo 0 --max_num_flights 10000 --data data_orly/data/new_data/TO_LFPO_7_B_train_no_nan.pkl --weights data_orly/src/generation/models/saved_weights/limited_one_typecode/B738_A320/10000/CVAE_B738_A320_lr_1_3_8.pth --typecodes B738 A320 --num_flights 1.0 1.0 --data_save data_orly/data/sampled_data/combined_data/B738_A320_10000_1e3.pkl --cuda 1 --scale 1 --balanced 1 --weights_data 0 --pseudo_in 800 --l_dim 64
uv run python data_orly/scripts/script_train/train_VAE.py --data data_orly/data/sampled_data/combined_data/B738_A320_10000_1e3.pkl --weights data_orly/src/generation/models/saved_weights/limited_one_typecode/B738_A320/10000/VAE_A320_8.pth --typecodes A320 --cuda 2 --scale 1 --l_dim 64 --pseudo_in 800
uv run python data_orly/scripts/script_train/train_VAE.py --data data_orly/data/sampled_data/combined_data/B738_A320_10000_1e3.pkl --weights data_orly/src/generation/models/saved_weights/limited_one_typecode/B738_A320/10000/VAE_B738_8.pth --typecodes B738 --cuda 2 --scale 1 --l_dim 64 --pseudo_in 800

#loss
# mkdir -p data_orly/results/loss/B738_A320
# mkdir -p data_orly/results/loss/B738_A320/100

# uv run data_orly/scripts/loss/loss_compute.py --model data_orly/src/generation/models/saved_weights/limited_one_typecode/B738_A320/100/CVAE_both_B738_A320_all2.pth --file  data_orly/data/sampled_data/combined_data/B738_A320_all2.pkl --typecode B738 --typecodes B738 A320 --cond 1 --loss_file data_orly/results/loss/B738_A320/100/loss.csv --origin data_orly/data/takeoffs_LFPO_07.pkl --cuda 0 --l_dim 32 --pseudo_in 600
# uv run data_orly/scripts/loss/loss_compute.py --model data_orly/src/generation/models/saved_weights/limited_one_typecode/B738_A320/100/CVAE_both_B738_A320_all2.pth --file  data_orly/data/sampled_data/combined_data/B738_A320_all2.pkl --typecode A320 --typecodes B738 A320 --cond 1 --loss_file data_orly/results/loss/B738_A320/100/loss.csv --origin data_orly/data/takeoffs_LFPO_07.pkl --cuda 0 --l_dim 32 --pseudo_in 600
# uv run data_orly/scripts/loss/loss_compute.py --model data_orly/src/generation/models/saved_weights/limited_one_typecode/B738_A320/100/VAE_A320_all.pth --file  data_orly/data/sampled_data/combined_data/B738_A320_all2.pkl --typecode A320 --typecodes  A320 --cond 0 --loss_file data_orly/results/loss/B738_A320/100/loss.csv --origin data_orly/data/takeoffs_LFPO_07.pkl --cuda 0 --l_dim 32 --pseudo_in 600
# uv run data_orly/scripts/loss/loss_compute.py --model data_orly/src/generation/models/saved_weights/limited_one_typecode/B738_A320/100/VAE_B738_lim.pth --file  data_orly/data/sampled_data/combined_data/B738_A320_all2.pkl --typecode B738 --typecodes B738  --cond 0 --loss_file data_orly/results/loss/B738_A320/100/loss.csv --origin data_orly/data/takeoffs_LFPO_07.pkl --cuda 0 --l_dim 32 --pseudo_in 600


# #scn files
mkdir -p data_orly/scn/B738_A320/10000
mkdir -p data_orly/src/generation/saved_traff/B738_A320/10000

uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data data_orly/data/sampled_data/combined_data/B738_A320_10000_1e3.pkl --weight_file data_orly/src/generation/models/saved_weights/limited_one_typecode/B738_A320/10000/VAE_A320_8.pth --nf 2000 --cond 0 --typecodes A320 --scene_file B738_A320/10000 --l_dim 64 --pseudo_in 800
uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data data_orly/data/sampled_data/combined_data/B738_A320_10000_1e3.pkl --weight_file data_orly/src/generation/models/saved_weights/limited_one_typecode/B738_A320/10000/VAE_B738_8.pth --nf 2000 --cond 0 --typecodes B738 --scene_file B738_A320/10000 --l_dim 64 --pseudo_in 800
uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data data_orly/data/sampled_data/combined_data/B738_A320_10000_1e3.pkl --weight_file data_orly/src/generation/models/saved_weights/limited_one_typecode/B738_A320/10000/CVAE_B738_A320_lr_1_3_8.pth --nf 2000 --typecodes B738 A320 --cond 1 --typecode_to_gen B738 --scene_file B738_A320/10000 --l_dim 64 --pseudo_in 800
uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data data_orly/data/sampled_data/combined_data/B738_A320_10000_1e3.pkl --weight_file data_orly/src/generation/models/saved_weights/limited_one_typecode/B738_A320/10000/CVAE_B738_A320_lr_1_3_8.pth --nf 2000 --typecodes B738 A320 --cond 1 --typecode_to_gen A320 --scene_file B738_A320/10000 --l_dim 64 --pseudo_in 800