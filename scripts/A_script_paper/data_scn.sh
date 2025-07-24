mkdir -p scn/final_data

# uv run scripts/Test_sim/scn_gen/data_scn_gen.py --data /data/data/arnault/data/final_data/TO_LFPO_test_final.pkl --data_s /data/data/arnault/data/final_data/TO_LFPO_test_final_sub_samp.pkl --scn_file scn/final_data/final_train.scn --nf 3500 #30000 files in total



# uv run scripts/Test_sim/scn_gen/data_scn_gen.py --data /data/data/arnault/data/final_data/LFPO_all_A320_010.pkl --data_s /data/data/arnault/data/final_data/TO_LFPO_test_final_sub_samp_A320.pkl --scn_file scn/final_data/train_A320_10.scn --typecode A320 --nf 3500 #30000 files in total


uv run scripts/Test_sim/scn_gen/data_scn_gen.py --navpoint_path "temp_navp/tcvae_generation_navpoints.csv" --data "/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl" --data_s /data/data/arnault/data/final_data/TO_LFPO_test_final_sub_samp3.pkl --scn_file scripts/A_script_paper/scn/data3.scn --nf 40000 --typecodes  B738 A320 A321 A319 A20N A318 A21N #30000 files in total