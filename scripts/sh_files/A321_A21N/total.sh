#training

uv run python data_orly/scripts/script_train/train_CVAE.py --data data_orly/data/takeoffs_LFPO_07.pkl --weights data_orly/src/generation/models/saved_weights/limited_one_typecode/CVAE_both_A321_A21N_200.pth --typecodes A321 A21N --num_flights -1 150 --data_save data_orly/data/sampled_data/combined_data/A321_A21N_200.pkl --cuda 0 --scale 1
uv run python data_orly/scripts/script_train/train_VAE.py --data data_orly/data/sampled_data/combined_data/A321_A21N_200.pkl --weights data_orly/src/generation/models/saved_weights/limited_one_typecode/VAE_TCN_Vampprior_take_off_7_alt_cond_A321_full.pth --typecodes A321 --cuda 0 --scale 1
uv run python data_orly/scripts/script_train/train_VAE.py --data data_orly/data/sampled_data/combined_data/A321_A21N_200.pkl --weights data_orly/src/generation/models/saved_weights/limited_one_typecode/VAE_TCN_Vampprior_take_off_7_alt_cond_A21N_150.pth --typecodes A21N --cuda 0 --scale 0.4

#scn files

uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data "data_orly/data/sampled_data/combined_data/A321_A21N_200.pkl" --weight_file "data_orly/src/generation/models/saved_weights/limited_one_typecode/VAE_TCN_Vampprior_take_off_7_alt_cond_A21N_150.pth" --nf 2000 --cond 0 --typecodes A21N
uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data "data_orly/data/sampled_data/combined_data/A321_A21N_200.pkl" --weight_file "data_orly/src/generation/models/saved_weights/limited_one_typecode/VAE_TCN_Vampprior_take_off_7_alt_cond_A321_full.pth" --nf 2000 --cond 0 --typecodes A321
uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data "data_orly/data/sampled_data/combined_data/A321_A21N_200.pkl" --weight_file "data_orly/src/generation/models/saved_weights/limited_one_typecode/CVAE_both_A321_A21N_200.pth" --nf 2000 --typecodes A320 A321 --cond 1 --typecode_to_gen A321
uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data "data_orly/data/sampled_data/combined_data/A321_A21N_200.pkl" --weight_file "data_orly/src/generation/models/saved_weights/limited_one_typecode/CVAE_both_A321_A21N_200.pth" --nf 2000 --typecodes A320 A321 --cond 1 --typecode_to_gen A21N 
