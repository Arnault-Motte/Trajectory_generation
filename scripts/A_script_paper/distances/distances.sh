#Dist compute
mkdir -p data_orly/scripts/A_script_paper/distances/A320




#Ref dist

# uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff /data/data/arnault/data/final_data/TO_LFPO_test_final_sub_samp2.pkl --log_file "data_orly/sim_logs/all/train_A320.log"  --ignored data_orly/scripts/A_script_paper/scn/data_denied_flight.pkl --saved_name "dist_total.pkl" 
# uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff /data/data/arnault/data/final_data/TO_LFPO_test_final_sub_samp3.pkl --log_file "data_orly/scripts/A_script_paper/scn/data3.log"  --ignored data_orly/scripts/A_script_paper/scn/data3_denied_flight.pkl --saved_name "dist_total.pkl" 


# #CVAE dist
# # uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff data_orly/scripts/A_script_paper/scn/CVAE_ONNX_A320.pkl --log_file "data_orly/scripts/A_script_paper/scn/A320_CVAE.log"  --ignored data_orly/scripts/A_script_paper/scn/CVAE_ONNX_A320_denied_flight.pkl --saved_name "A320_CVAE_dist.pkl" --typecodes A320 
# uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff data_orly/scripts/A_script_paper/scn/CVAE_ONNX_A320.pkl --log_file "data_orly/scripts/A_script_paper/scn/CVAE_ONNX_A320_cond.log"  --ignored data_orly/scripts/A_script_paper/scn/CVAE_ONNX_A320_denied_flight.pkl --saved_name "A320_CVAE_dist.pkl" --typecodes A320 

# #VAE dist
# uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff data_orly/src/generation/saved_traff/test_all/VAE_B738_010_B738__1000_2000.pkl  --log_file data_orly/sim_logs/all/B738_solo.log --ignored data_orly/scn/test_all/VAE_B738_010_B738__1000_2000_denied_flight.pkl --saved_name "all/B738_solo.pkl" --typecodes B738
# uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff data_orly/scripts/A_script_paper/scn/VAE_ONNX_A320.pkl --log_file "data_orly/scripts/A_script_paper/scn/A320_VAE.log"  --ignored data_orly/scripts/A_script_paper/scn/VAE_ONNX_A320_denied_flight.pkl --saved_name "A320_VAE_dist.pkl" --typecodes A320 
# uv run python data_orly/scripts/Test_sim/scn_gen/dist_test.py --og_traff data_orly/scripts/A_script_paper/scn/VAE_ONNX_A320.pkl --log_file "data_orly/scripts/A_script_paper/scn/VAE_ONNX_A320_cond.log"  --ignored data_orly/scripts/A_script_paper/scn/VAE_ONNX_A320_denied_flight.pkl --saved_name "A320_VAE_dist.pkl" --typecodes A320 

# #Plot them
# mkdir -p data_orly/figures/distances/all

uv run python data_orly/scripts/Test_sim/scn_gen/plot_distance.py --og_traff /data/data/arnault/data/final_data/TO_LFPO_test_final_sub_samp3.pkl --true_dist "data_orly/results/distances/dist_total.pkl" --typecode A320 --labels VAE CVAE --save_path "data_orly/scripts/A_script_paper/distances/A320/distance_A320_plot.png" --file_paths  "data_orly/results/distances/A320_VAE_dist.pkl" "data_orly/results/distances/A320_CVAE_dist.pkl" 
