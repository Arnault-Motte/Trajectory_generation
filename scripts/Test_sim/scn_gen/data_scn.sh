mkdir -p data_orly/scn/final_data

# uv run data_orly/scripts/Test_sim/scn_gen/data_scn_gen.py --data /data/data/arnault/data/final_data/TO_LFPO_test_final.pkl --data_s /data/data/arnault/data/final_data/TO_LFPO_test_final_sub_samp.pkl --scn_file data_orly/scn/final_data/final_train.scn --nf 3500 #30000 files in total



# uv run data_orly/scripts/Test_sim/scn_gen/data_scn_gen.py --data /data/data/arnault/data/final_data/LFPO_all_A320_010.pkl --data_s /data/data/arnault/data/final_data/TO_LFPO_test_final_sub_samp_A320.pkl --scn_file data_orly/scn/final_data/train_A320_10.scn --typecode A320 --nf 3500 #30000 files in total


uv run data_orly/scripts/Test_sim/scn_gen/data_scn_gen.py --data "/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl" --data_s /data/data/arnault/data/final_data/TO_LFPO_test_final_sub_samp2.pkl --scn_file data_orly/scripts/A_script_paper/scn/data.scn --nf 3500 #30000 files in total