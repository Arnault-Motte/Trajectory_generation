#Dist compute
mkdir -p data_orly/results/distances/all
mkdir -p data_orly/results/distances/all/test_10



#Ref dist
# uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff /data/data/arnault/data/final_data/TO_LFPO_test_final_sub_samp.pkl --log_file "data_orly/sim_logs/all/log_all.log"  --ignored data_orly/scn/final_data/final_train_denied_flight.pkl --saved_name "all/test_dist.pkl" --typecodes B738 
# uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff /data/data/arnault/data/final_data/TO_LFPO_test_final_sub_samp_A320.pkl --log_file "data_orly/sim_logs/all/train_A320.log"  --ignored data_orly/scn/final_data/train_A320_10_denied_flight.pkl --saved_name "all/dist_A320_train.pkl" 
uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff /data/data/arnault/data/final_data/TO_LFPO_test_old_A320.pkl  --log_file "data_orly/sim_logs/all/A320_old.log"  --ignored data_orly/scn/final_data/A320_old_denied_flight.pkl --saved_name "all/dist_A320_old_data.pkl" 

#CVAE dist
# uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff data_orly/src/generation/saved_traff/test_all/CVAE_full_A320_010_B738_A321_A319_A20N_A318_A21N_A320_A359_E145_A333_B738_1000_2000.pkl  --log_file "data_orly/sim_logs/all/B738_both.log"  --ignored data_orly/scn/test_all/CVAE_full_A320_010_B738_A321_A319_A20N_A318_A21N_A320_A359_E145_A333_B738_1000_2000_denied_flight.pkl --saved_name "all/B738_both.pkl" --typecodes B738 
# uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff data_orly/src/generation/saved_traff/test_all/CVAE_full_A320_010_B738_A321_A319_A20N_A318_A21N_A320_A359_E145_A333_A320_1000_2000.pkl --log_file "data_orly/sim_logs/all/A320_both.log"  --ignored data_orly/scn/test_all/CVAE_full_A320_010_B738_A321_A319_A20N_A318_A21N_A320_A359_E145_A333_A320_1000_2000_denied_flight.pkl --saved_name "all/A320_both.pkl" --typecodes A320 
# #VAE dist
# uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff data_orly/src/generation/saved_traff/test_all/VAE_B738_010_B738__1000_2000.pkl  --log_file data_orly/sim_logs/all/B738_solo.log --ignored data_orly/scn/test_all/VAE_B738_010_B738__1000_2000_denied_flight.pkl --saved_name "all/B738_solo.pkl" --typecodes B738
# uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff data_orly/src/generation/saved_traff/test_all/VAE_A320_010_A320__1000_2000.pkl  --log_file data_orly/sim_logs/all/A320_solo.log --ignored data_orly/scn/test_all/VAE_A320_010_A320__1000_2000_denied_flight.pkl --saved_name "all/A320_solo.pkl" --typecodes A320


# #Plot them
# mkdir -p data_orly/figures/distances/all

# uv run python data_orly/scripts/Test_sim/scn_gen/plot_distance.py --og_traff /data/data/arnault/data/final_data/TO_LFPO_test_final_sub_samp.pkl --true_dist "data_orly/results/distances/all/test_dist.pkl" --typecode B738 --labels VAE CVAE --save_path "data_orly/figures/distances/all/B738.png" --file_paths  "data_orly/results/distances/all/B738_solo.pkl" "data_orly/results/distances/all/B738_both.pkl" 
uv run python data_orly/scripts/Test_sim/scn_gen/plot_distance.py --og_traff /data/data/arnault/data/final_data/TO_LFPO_test_final_sub_samp.pkl --true_dist "data_orly/results/distances/all/test_dist.pkl" --typecode A320 --labels VAE CVAE TRAIN_DATA OLD_DATA --save_path "data_orly/figures/distances/all/A320.png" --file_paths  "data_orly/results/distances/all/A320_solo.pkl" "data_orly/results/distances/all/A320_both.pkl" "data_orly/results/distances/all/dist_A320_train.pkl" "data_orly/results/distances/all/dist_A320_old_data.pkl" 

#compute the losses

