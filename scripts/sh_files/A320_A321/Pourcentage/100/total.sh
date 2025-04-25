#creat dir 
mkdir -p data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321
mkdir -p data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/100
mkdir -p data_orly/data_orly/scn/A320_A321/100

#training

# uv run python data_orly/scripts/script_train/train_CVAE.py --data data_orly/data/takeoffs_LFPO_07.pkl --weights data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/100/CVAE_both_A320_A321_all.pth --typecodes A320 A321 --num_flights 1.0 1.0 --data_save data_orly/data/sampled_data/combined_data/A320_A321_all.pkl --cuda 0 --scale 0.7 --balanced 1
# uv run python data_orly/scripts/script_train/train_VAE.py --data data_orly/data/sampled_data/combined_data/A320_A321_all.pkl --weights data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/100/VAE_A321_all.pth --typecodes A321 --cuda 0 --scale 0.7
# uv run python data_orly/scripts/script_train/train_VAE.py --data data_orly/data/sampled_data/combined_data/A320_A321_all.pkl --weights data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/100/VAE_A320_lim.pth --typecodes A320 --cuda 0 --scale 0.8

#loss
mkdir -p data_orly/results/loss/A320_A321
mkdir -p data_orly/results/loss/A320_A321/100

# uv run data_orly/scripts/loss/loss_compute.py --model data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/100/CVAE_both_A320_A321_all.pth --file  data_orly/data/sampled_data/combined_data/A320_A321_all.pkl --typecode A320 --typecodes A320 A321 --cond 1 --loss_file data_orly/results/loss/A320_A321/100/loss.csv --origin data_orly/data/takeoffs_LFPO_07.pkl
# uv run data_orly/scripts/loss/loss_compute.py --model data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/100/CVAE_both_A320_A321_all.pth --file  data_orly/data/sampled_data/combined_data/A320_A321_all.pkl --typecode A321 --typecodes A320 A321 --cond 1 --loss_file data_orly/results/loss/A320_A321/100/loss.csv --origin data_orly/data/takeoffs_LFPO_07.pkl
# uv run data_orly/scripts/loss/loss_compute.py --model data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/100/VAE_A321_all.pth --file  data_orly/data/sampled_data/combined_data/A320_A321_all.pkl --typecode A321 --typecodes  A321 --cond 0 --loss_file data_orly/results/loss/A320_A321/100/loss.csv --origin data_orly/data/takeoffs_LFPO_07.pkl
uv run data_orly/scripts/loss/loss_compute.py --model data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/100/VAE_A320_lim.pth --file  data_orly/data/sampled_data/combined_data/A320_A321_all.pkl --typecode A320 --typecodes A320  --cond 0 --loss_file data_orly/results/loss/A320_A321/100/loss.csv --origin data_orly/data/takeoffs_LFPO_07.pkl



#scn files
# mkdir -p data_orly/scn/A320_A321/100
# mkdir -p data_orly/src/generation/saved_traff/A320_A321/100

# uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data "data_orly/data/sampled_data/combined_data/A320_A321_all.pkl" --weight_file "data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/100/VAE_A321_all.pth" --nf 2000 --cond 0 --typecodes A321 --scene_file A320_A321/100
# uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data "data_orly/data/sampled_data/combined_data/A320_A321_all.pkl" --weight_file "data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/100/VAE_A320.pth" --nf 2000 --cond 0 --typecodes A320 --scene_file A320_A321/100
# uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data "data_orly/data/sampled_data/combined_data/A320_A321_all.pkl" --weight_file "data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/100/CVAE_both_A320_A321_all.pth" --nf 2000 --typecodes A320 A321 --cond 1 --typecode_to_gen A320 --scene_file A320_A321/100
# uv run python data_orly/scripts/Test_sim/scn_gen/scn_gen.py --data "data_orly/data/sampled_data/combined_data/A320_A321_all.pkl" --weight_file "data_orly/src/generation/models/saved_weights/limited_one_typecode/A320_A321/100/CVAE_both_A320_A321_all.pth" --nf 2000 --typecodes A320 A321 --cond 1 --typecode_to_gen A321 --scene_file A320_A321/100