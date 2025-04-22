#training

# uv run python data_orly/scripts/script_train/train_CVAE.py --data data_orly/data/takeoffs_LFPO_07.pkl --weights data_orly/src/generation/models/saved_weights/limited_one_typecode/test_B738_A21N/CVAE_both_B738_A21N_151.pth --typecodes B738 A21N --num_flights -1 151 --data_save data_orly/data/sampled_data/combined_data/B738_A21N_151.pkl --cuda 0 --scale 0.9
#uv run python data_orly/scripts/script_train/train_VAE.py --data data_orly/data/sampled_data/combined_data/B738_A21N_151.pkl --weights data_orly/src/generation/models/saved_weights/limited_one_typecode/test_B738_A21N/VAE_A21N_151.pth --typecodes A21N --cuda 0 --scale 0.5
# uv run python data_orly/scripts/script_train/train_VAE.py --data data_orly/data/sampled_data/combined_data/B738_A21N_151.pkl --weights data_orly/src/generation/models/saved_weights/limited_one_typecode/test_B738_A21N/VAE_B738.pth --typecodes B738 --cuda 0 --scale 1

#scn files

# uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data "data_orly/data/sampled_data/combined_data/B738_A21N_151.pkl" --weight_file "data_orly/src/generation/models/saved_weights/limited_one_typecode/test_B738_A21N/VAE_A21N_151.pth" --nf 2000 --cond 0 --typecodes A21N
# uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data "data_orly/data/sampled_data/combined_data/B738_A21N_151.pkl" --weight_file "data_orly/src/generation/models/saved_weights/limited_one_typecode/test_B738_A21N/VAE_B738.pth" --nf 2000 --cond 0 --typecodes B738
uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data "data_orly/data/sampled_data/combined_data/B738_A21N_151.pkl" --weight_file "data_orly/src/generation/models/saved_weights/limited_one_typecode/test_B738_A21N/CVAE_both_B738_A21N_151.pth" --nf 2000 --typecodes B738 A21N --cond 1 --typecode_to_gen B738
uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data "data_orly/data/sampled_data/combined_data/B738_A21N_151.pkl" --weight_file "data_orly/src/generation/models/saved_weights/limited_one_typecode/test_B738_A21N/CVAE_both_B738_A21N_151.pth" --nf 2000 --typecodes B738 A21N --cond 1 --typecode_to_gen A21N 
