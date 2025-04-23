#Dist compute
mkdir -p data_orly/results/distances/A320_A321
mkdir -p data_orly/results/distances/A320_A321/100


uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff "data_orly/src/generation/saved_traff/A320_A321/100/CVAE_both_A320_A321_all_A320_A321_A320_800_2000.pkl" --log_file "data_orly/sim_logs/A320_A321/100/A320_both.log"  --ignored "data_orly/scn/A320_A321/100/CVAE_both_A320_A321_all_A320_A321_A320_800_2000_denied_flight.pkl" --saved_name "A320_A321/100/A320_both.pkl" --typecodes A320
uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff "data_orly/src/generation/saved_traff/A320_A321/100/CVAE_both_A320_A321_all_A320_A321_A321_800_2000.pkl" --log_file "data_orly/sim_logs/A320_A321/100/A321_both.log"  --ignored "data_orly/scn/A320_A321/100/CVAE_both_A320_A321_all_A320_A321_A321_800_2000_denied_flight.pkl" --saved_name "A320_A321/100/A321_both.pkl" --typecodes A321
uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff "data_orly/src/generation/saved_traff/A320_A321/100/VAE_A320_A320__800_2000.pkl"  --log_file ""data_orly/sim_logs/A320_A321/100/A320_solo.log"" --ignored "data_orly/scn/A320_A321/100/VAE_A320_A320__800_2000_denied_flight.pkl" --saved_name "A320_A321/100/A320_solo.pkl" --typecodes A320
uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff "data_orly/src/generation/saved_traff/A320_A321/100/VAE_A321_all_A321__800_2000.pkl"  --log_file ""data_orly/sim_logs/A320_A321/100/A321_solo.log"" --ignored "data_orly/scn/A320_A321/100/VAE_A321_all_A321__800_2000_denied_flight.pkl" --saved_name "A320_A321/100/A321_solo.pkl" --typecodes A321

#Plot them
mkdir -p data_orly/figures/distances/A320_A321
mkdir -p data_orly/figures/distances/A320_A321/100

uv run python data_orly/scripts/Test_sim/scn_gen/plot_distance.py --og_traff data_orly/data/sampled_data/combined_data/A320_A321_all.pkl --true_dist "data_orly/results/distances/d_TO_7_Real_more_points.pkl" --typecode A321 --labels VAE CVAE --save_path "data_orly/figures/distances/A320_A321/100/A320.png" --file_paths  "data_orly/results/distances/A320_A321/100/A320_solo.pkl" "data_orly/results/distances/A320_A321/100/A320_both.pkl"
uv run python data_orly/scripts/Test_sim/scn_gen/plot_distance.py --og_traff data_orly/data/sampled_data/combined_data/A320_A321_all.pkl --true_dist "data_orly/results/distances/d_TO_7_Real_more_points.pkl" --typecode A21N --labels VAE CVAE --save_path "data_orly/figures/distances/A320_A321/100/A321.png" --file_paths  "data_orly/results/distances/A320_A321/100/A321_solo.pkl" "data_orly/results/distances/A320_A321/100/A321_both.pkl"