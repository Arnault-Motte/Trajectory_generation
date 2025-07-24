#Dist compute
mkdir -p scripts/A_script_paper/distances/A320




#Ref dist

uv run python scripts/Test_sim/scn_gen/dist_test.py --og_traff /data/data/arnault/data/final_data/TO_LFPO_test_final_sub_samp2.pkl --log_file "sim_logs/all/train_A320.log"  --ignored scripts/A_script_paper/scn/data_denied_flight.pkl --saved_name "dist_total.pkl" 


#CVAE dist
uv run python scripts/Test_sim/scn_gen/dist_test.py --og_traff scripts/A_script_paper/scn/CVAE_ONNX_A320.pkl --log_file "scripts/A_script_paper/scn/A320_CVAE.log"  --ignored scripts/A_script_paper/scn/CVAE_ONNX_A320_denied_flight.pkl --saved_name "A320_CVAE_dist.pkl" --typecodes A320 
# #VAE dist
# uv run python scripts/Test_sim/scn_gen/dist_test.py --og_traff src/generation/saved_traff/test_all/VAE_B738_010_B738__1000_2000.pkl  --log_file sim_logs/all/B738_solo.log --ignored scn/test_all/VAE_B738_010_B738__1000_2000_denied_flight.pkl --saved_name "all/B738_solo.pkl" --typecodes B738
uv run python scripts/Test_sim/scn_gen/dist_test.py --og_traff scripts/A_script_paper/scn/VAE_ONNX_A320.pkl --log_file "scripts/A_script_paper/scn/A320_VAE.log"  --ignored scripts/A_script_paper/scn/VAE_ONNX_A320_denied_flight.pkl --saved_name "A320_VAE_dist.pkl" --typecodes A320 


# #Plot them
# mkdir -p figures/distances/all

uv run python scripts/Test_sim/scn_gen/plot_distance.py --og_traff /data/data/arnault/data/final_data/TO_LFPO_test_final_sub_samp2.pkl --true_dist "scripts/A_script_paper/distances/A320/dist_total.pkl" --typecode B738 --labels VAE CVAE --save_path "scripts/A_script_paper/distances/A320/distance_A320_plot.png" --file_paths  "scripts/A_script_paper/distances/A320/A320_VAE_dist.pkl" "scripts/A_script_paper/distances/A320/A320_CVAE_dist.pkl" 
