#creat dir 
mkdir -p data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321
mkdir -p data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/50
mkdir -p data_orly/data_orly/scn/A320_A321/50

#training

uv run python data_orly/scripts/script_train/train_CVAE.py --data data_orly/data/sampled_data/combined_data/A320_A321_all.pkl --weights data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/50/CVAE_both_A320_A321_50.pth --typecodes A320 A321 --num_flights 1.0 0.5 --data_save data_orly/data/sampled_data/combined_data/A320_A321_50.pkl --cuda 0 --scale 1 --balanced 1
uv run python data_orly/scripts/script_train/train_VAE.py --data data_orly/data/sampled_data/combined_data/A320_A321_50.pkl --weights data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/50/VAE_A321_50.pth --typecodes A321 --cuda 0 --scale 0.8
#uv run python data_orly/scripts/script_train/train_VAE.py --data data_orly/data/sampled_data/combined_data/A320_A321_50.pkl --weights data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/50/VAE_A320.pth --typecodes A320 --cuda 0 --scale 0.9

#scn files
mkdir -p data_orly/scn/A320_A321/50
mkdir -p data_orly/src/generation/saved_traff/A320_A321
mkdir -p data_orly/src/generation/saved_traff/A320_A321/50

uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data "data_orly/data/sampled_data/combined_data/A320_A321_50.pkl" --weight_file "data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/50/VAE_A321_50.pth" --nf 2000 --cond 0 --typecodes A321 --scene_file A320_A321/50
#uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data "data_orly/data/sampled_data/combined_data/A320_A321_50.pkl" --weight_file "data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/50/VAE_A320.pth" --nf 2000 --cond 0 --typecodes A320 --scene_file A320_A321/50
uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data "data_orly/data/sampled_data/combined_data/A320_A321_50.pkl" --weight_file "data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/50/CVAE_both_A320_A321_50.pth" --nf 2000 --typecodes A320 A321 --cond 1 --typecode_to_gen A320 --scene_file A320_A321/50
uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data "data_orly/data/sampled_data/combined_data/A320_A321_50.pkl" --weight_file "data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/50/CVAE_both_A320_A321_50.pth" --nf 2000 --typecodes A320 A321 --cond 1 --typecode_to_gen A321 --scene_file A320_A321/50