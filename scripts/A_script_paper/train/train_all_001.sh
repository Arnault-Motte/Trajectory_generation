mkdir -p data/sampled_data/combined_data/full
mkdir -p src/generation/models/saved_weights/full


# #limit A320 to 10% so 2000 trajectories
# #train the whole dataset

uv run python scripts/script_train/train_CVAE.py --cond_pseudo 0 --data_save "/data/data/arnault/data/final_data/CVAE_full_A320_001.pkl" --data /data/data/arnault/data/final_data/TO_LFPO_final.pkl  --weights src/generation/models/saved_weights/full/CVAE_full_A320_001.pth --typecodes B738 A320 A321 A319 A20N A318 A21N A359 E145 A333 --num_flights 1.0 0.01 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 --cuda 1 --scale 1 --balanced 0 --weights_data 0 --pseudo_in 1000 --l_dim 64
uv run python scripts/script_train/train_VAE.py --data "/data/data/arnault/data/final_data/CVAE_full_A320_001.pkl" --weights src/generation/models/saved_weights/full/VAE_A320_001.pth --typecodes A320 --cuda 1 --scale 0.5 --l_dim 32 --pseudo_in 600 #--sample 0.1 --save_sample "data/sampled_data/combined_data/full/VAE_A320_010.pkl" #10% of the A320 sampled
uv run python scripts/script_train/train_VAE.py --data /data/data/arnault/data/final_data/LFPO_all_A320_010_3.pkl --weights src/generation/models/saved_weights/full/VAE_B738_010_3.pth --typecodes B738 --cuda 0 --scale 1 --l_dim 64 --pseudo_in 1000 #--sample 0.1 --save_sample "data/sampled_data/combined_data/full/VAE_A320_010.pkl" #10% of the A320 sampled


uv run python scripts/script_train/train_CVAE.py --cond_pseudo 0 --data_save "/data/data/arnault/data/final_data/CVAE_full_A320_0005.pkl" --data /data/data/arnault/data/final_data/TO_LFPO_final.pkl  --weights src/generation/models/saved_weights/full/CVAE_full_A320_0005.pth --typecodes B738 A320 A321 A319 A20N A318 A21N A359 E145 A333 --num_flights 1.0 0.005 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 --cuda 1 --scale 1 --balanced 0 --weights_data 0 --pseudo_in 1000 --l_dim 64
uv run python scripts/script_train/train_VAE.py --data "/data/data/arnault/data/final_data/CVAE_full_A320_0005.pkl" --weights src/generation/models/saved_weights/full/VAE_A320_0005.pth --typecodes A320 --cuda 1 --scale 0.5 --l_dim 32 --pseudo_in 600 #--sample 0.1 --save_sample "data/sampled_data/combined_data/full/VAE_A320_010.pkl" #10% of the A320 sampled


uv run python scripts/script_train/train_CVAE.py --cond_pseudo 1 --spec 0 --data_save "/data/data/arnault/data/final_data/CVAE_5_A320_0005_cond.pkl" --data "/data/data/arnault/data/final_data/TO_LFPO_final.pkl"  --weights src/generation/models/saved_weights/full/CVAE_5_A320_0005_cond.pth --typecodes B738 A320 A321 A318 A21N  --num_flights 1.0 0.005 1.0 1.0 1.0 --cuda 1 --scale 1 --balanced 0 --weights_data 0 --pseudo_in 1000 --l_dim 64 --batch_size 500
uv run python scripts/script_train/train_CVAE.py --cond_pseudo 0 --spec 0 --data_save "/data/data/arnault/data/final_data/CVAE_5_A320_0005.pkl" --data "/data/data/arnault/data/final_data/TO_LFPO_final.pkl"  --weights src/generation/models/saved_weights/full/CVAE_5_A320_0005.pth --typecodes B738 A320 A321 A318 A21N  --num_flights 1.0 0.005 1.0 1.0 1.0 --cuda 1 --scale 1 --balanced 0 --weights_data 0 --pseudo_in 1000 --l_dim 64 --batch_size 500

uv run python scripts/script_train/train_CVAE.py --cond_pseudo 0 --spec 0 --data_save "/data/data/arnault/data/final_data/CVAE_5_A320_0005_500.pkl" --data "/data/data/arnault/data/final_data/TO_LFPO_final.pkl"  --weights src/generation/models/saved_weights/full/CVAE_500_A320_0005.pth --typecodes B738 A320 A321 A319 A20N A318 A21N A359 E145 A333  --num_flights 1.0 0.005 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 --cuda 1 --scale 1 --balanced 0 --weights_data 0 --pseudo_in 500 --l_dim 64 --batch_size 500



typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')

for elem in "${typecodes[@]}"; do 
    uv run scripts/script_train/train_VAE.py --data /data/data/arnault/data/final_data/LFPO_all_A320_010_3.pkl --weights "src/generation/models/saved_weights/full/VAE_${elem}_A320_010_3.pth" --typecodes "$elem" --cuda 0 --scale 1 --l_dim 64 --pseudo_in 1000
done

#SCN file generation

#scn files
mkdir -p scn/test_all
mkdir -p src/generation/saved_traff/test_all


# for the full test
# uv run python scripts/Test_sim/scn_gen/scn_gen.py --data /data/data/arnault/data/final_data/TO_LFPO_final.pkl --weight_file src/generation/models/saved_weights/full/VAE_A320_all.pth --nf 2000 --cond 0 --typecodes A320 --scene_file test_all --l_dim 64 --pseudo_in 1000 --cuda 0
# uv run python scripts/Test_sim/scn_gen/scn_gen.py --data /data/data/arnault/data/final_data/TO_LFPO_final.pkl --weight_file src/generation/models/saved_weights/full/VAE_B738_all.pth --nf 2000 --cond 0 --typecodes B738 --scene_file test_all --l_dim 64 --pseudo_in 1000 --cuda 0
# uv run python scripts/Test_sim/scn_gen/scn_gen.py --data /data/data/arnault/data/final_data/TO_LFPO_final.pkl --weight_file src/generation/models/saved_weights/full/CVAE_total.pth --nf 2000 --cond 1 --typecode_to_gen A320 --scene_file test_all --l_dim 64 --pseudo_in 1000 --cuda 0
# uv run python scripts/Test_sim/scn_gen/scn_gen.py --data /data/data/arnault/data/final_data/TO_LFPO_final.pkl --weight_file src/generation/models/saved_weights/full/CVAE_total.pth --nf 2000 --cond 1 --typecode_to_gen B738 --scene_file test_all --l_dim 64 --pseudo_in 1000 --cuda 0


# for the 010 test
# uv run python scripts/Test_sim/scn_gen/scn_gen.py --data /data/data/arnault/data/final_data/LFPO_all_A320_010.pkl --weight_file src/generation/models/saved_weights/full/VAE_A320_010_2.pth --nf 2000 --cond 0 --typecodes A320 --scene_file test_all --l_dim 64 --pseudo_in 1000 --cuda 0
# uv run python scripts/Test_sim/scn_gen/scn_gen.py --data /data/data/arnault/data/final_data/LFPO_all_A320_010.pkl --weight_file src/generation/models/saved_weights/full/VAE_B738_010_2.pth --nf 2000 --cond 0 --typecodes B738 --scene_file test_all --l_dim 64 --pseudo_in 1000 --cuda 0
# uv run python scripts/Test_sim/scn_gen/scn_gen.py --data /data/data/arnault/data/final_data/LFPO_all_A320_010.pkl --weight_file src/generation/models/saved_weights/full/CVAE_full_A320_010_2.pth --nf 2000 --cond 1 --typecode_to_gen A320 --scene_file test_all --l_dim 64 --pseudo_in 1000 --cuda 0
# uv run python scripts/Test_sim/scn_gen/scn_gen.py --data /data/data/arnault/data/final_data/LFPO_all_A320_010.pkl --weight_file src/generation/models/saved_weights/full/CVAE_full_A320_010_2.pth --nf 2000 --cond 1 --typecode_to_gen B738 --scene_file test_all --l_dim 64 --pseudo_in 1000 --cuda 0


#same but alternative pseudo_in
#train the whole dataset
# uv run python scripts/script_train/train_CVAE.py --cond_pseudo 0 --data /data/data/arnault/data/final_data/LFPO_all_A320_010.pkl --data_save /data/data/arnault/data/final_data/LFPO_all_A320_010_2.pkl --weights src/generation/models/saved_weights/full/CVAE_full_A320_010_1200.pth --typecodes B738 A320 A321 A319 A20N A318 A21N A359 E145 A333 --num_flights 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 --cuda 0 --scale 1 --balanced 0 --weights_data 0 --pseudo_in 1200 --l_dim 64
# uv run python scripts/script_train/train_VAE.py --data /data/data/arnault/data/final_data/LFPO_all_A320_010.pkl --weights src/generation/models/saved_weights/full/VAE_A320_010_600.pth --typecodes A320 --cuda 0 --scale 1 --l_dim 64 --pseudo_in 600 #--sample 0.1 --save_sample "data/sampled_data/combined_data/full/VAE_A320_010.pkl" #10% of the A320 sampled

