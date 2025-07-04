# uv run data_orly/scripts/A_script_paper/alt_profiles.py --x_col "CAS" --onnx_dir "data_orly/models_paper/CVAE_minority" --plot_path "data_orly/figures/paper/altitude_profiles/alt_profiles_new/speed_alt_gen.png" --typecodes B738 A320 A321 
uv run data_orly/scripts/A_script_paper/alt_profiles.py --x_col "CAS" --data "/data/data/arnault/data/takeoffs_LFPO_07.pkl" --plot_path "data_orly/figures/paper/altitude_profiles/alt_profiles_new/speed_alt_dataold.png" --typecodes B738 A320 A321

# uv run data_orly/scripts/A_script_paper/alt_profiles.py --x_col "timedelta" --onnx_dir "data_orly/models_paper/CVAE_minority" --plot_path "data_orly/figures/paper/altitude_profiles/alt_profiles_new/time_alt_gen.png" --typecodes B738 A320 A321 
uv run data_orly/scripts/A_script_paper/alt_profiles.py --x_col "timedelta" --data "/data/data/arnault/data/takeoffs_LFPO_07.pkl" --plot_path "data_orly/figures/paper/altitude_profiles/alt_profiles_new/time_alt_data_old.png" --typecodes B738 A320 A321

