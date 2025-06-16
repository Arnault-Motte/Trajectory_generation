#Dist compute
mkdir -p data_orly/results/distances/B738_A320
mkdir -p data_orly/results/distances/B738_A320/10000



#Ref dist


#CVAE dist
uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff /data/data/arnault/data/sampled_data/combined_data/B738_A320_10000_1e3.pkl --log_file "data_orly/sim_logs/B738_A320/10000/B738_both.log"  --ignored data_orly/scn/B738_A320/10000/CVAE_B738_A320_lr_1_3_8_A320_B738_B738_800_2000_denied_flight.pkl --saved_name "B738_A320/10000/B738_both.pkl" --typecodes B738 
uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff /data/data/arnault/data/sampled_data/combined_data/B738_A320_10000_1e3.pkl --log_file "data_orly/sim_logs/B738_A320/10000/A320_both.log"  --ignored data_orly/scn/B738_A320/10000/CVAE_B738_A320_lr_1_3_8_A320_B738_A320_800_2000_denied_flight.pkl --saved_name "B738_A320/10000/A320_both.pkl" --typecodes A320 
#VAE dist
uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff /data/data/arnault/data/sampled_data/combined_data/B738_A320_10000_1e3.pkl  --log_file data_orly/sim_logs/B738_A320/10000/B738_solo.log --ignored data_orly/scn/B738_A320/10000/VAE_B738_8_B738__800_2000_denied_flight.pkl --saved_name "B738_A320/10000/B738_solo.pkl" --typecodes B738
uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff /data/data/arnault/data/sampled_data/combined_data/B738_A320_10000_1e3.pkl  --log_file data_orly/sim_logs/B738_A320/10000/A320_solo.log --ignored data_orly/scn/B738_A320/10000/VAE_A320_8_A320__800_2000_denied_flight.pkl --saved_name "B738_A320/10000/A320_solo.pkl" --typecodes A320


#Plot them
mkdir -p data_orly/figures/distances/A320_A321
mkdir -p data_orly/figures/distances/A320_A321/100

uv run python data_orly/scripts/Test_sim/scn_gen/plot_distance.py --og_traff data_orly/data/sampled_data/combined_data/A320_A321_all.pkl --true_dist "data_orly/results/distances/d_TO_7_Real_more_points.pkl" --typecode A320 --labels VAE CVAE --save_path "data_orly/figures/distances/A320_A321/100/A320.png" --file_paths  "data_orly/results/distances/A320_A321/100/A320_solo.pkl" "data_orly/results/distances/A320_A321/100/A320_both.pkl"
uv run python data_orly/scripts/Test_sim/scn_gen/plot_distance.py --og_traff data_orly/data/sampled_data/combined_data/A320_A321_all.pkl --true_dist "data_orly/results/distances/d_TO_7_Real_more_points.pkl" --typecode A321 --labels VAE CVAE --save_path "data_orly/figures/distances/A320_A321/100/A321.png" --file_paths  "data_orly/results/distances/A320_A321/100/A321_solo.pkl" "data_orly/results/distances/A320_A321/100/A321_both.pkl"

#compute the losses

