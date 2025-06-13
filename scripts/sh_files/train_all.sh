#  uv run python data_orly/scripts/script_train/train_CVAE.py --cond_pseudo 0 --data data_orly/data/new_data/TO_LFPO_7_B_train_no_nan.pkl --weights data_orly/src/generation/models/saved_weights/CVAE_10_new_data_t.pth --cuda 1 --scale 1 --balanced 1 --weights_data 0 --pseudo_in 1000 --l_dim 64
# uv run python data_orly/scripts/script_train/train_VAE.py --data data_orly/data/new_data/TO_LFPO_7_B_train_no_nan.pkl --weights data_orly/src/generation/models/saved_weights/VAE_A320_all_new_data.pth --typecodes A320 --cuda 0 --scale 1 --l_dim 64 --pseudo_in 1000
# uv run python data_orly/scripts/script_train/train_VAE.py --data data_orly/data/new_data/TO_LFPO_7_B_train_no_nan.pkl --weights data_orly/src/generation/models/saved_weights/VAE_B738_all_new_data.pth --typecodes B738 --cuda 0 --scale 1 --l_dim 64 --pseudo_in 1000


#SCN file generation

#scn files
mkdir -p data_orly/scn/test_all
mkdir -p data_orly/src/generation/saved_traff/test_all

# uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data data_orly/data/new_data/TO_LFPO_7_B_train_no_nan.pkl --weight_file data_orly/src/generation/models/saved_weights/VAE_A320_all_new_data.pth --nf 2000 --cond 0 --typecodes A320 --scene_file test_all --l_dim 64 --pseudo_in 1000 --cuda 1
# uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data data_orly/data/new_data/TO_LFPO_7_B_train_no_nan.pkl --weight_file data_orly/src/generation/models/saved_weights/VAE_B738_all_new_data.pth --nf 2000 --cond 0 --typecodes B738 --scene_file test_all --l_dim 64 --pseudo_in 1000 --cuda 1
uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data data_orly/data/new_data/TO_LFPO_7_B_train_no_nan.pkl --weight_file data_orly/src/generation/models/saved_weights/CVAE_10_new_data.pth --nf 2000 --cond 1 --typecode_to_gen A320 --scene_file test_all --l_dim 64 --pseudo_in 1000 --cuda 1
uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data data_orly/data/new_data/TO_LFPO_7_B_train_no_nan.pkl --weight_file data_orly/src/generation/models/saved_weights/CVAE_10_new_data.pth --nf 2000 --cond 1 --typecode_to_gen B738 --scene_file test_all --l_dim 64 --pseudo_in 1000 --cuda 1