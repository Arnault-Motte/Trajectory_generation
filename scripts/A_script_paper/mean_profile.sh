# uv run data_orly/scripts/A_script_paper/mean_profile.py --onnx_cvae_dir "data_orly/models_paper/CVAE_A320_010" --data /data/data/arnault/data/final_data/TO_LFPO_test_final.pkl --plot_path data_orly/figures/paper/compare_profile/final_plots/compare_profiles_CVAE_010.png --typecodes B738 A321 A320 A319 A20N A21N
# uv run data_orly/scripts/A_script_paper/mean_profile.py --onnx_cvae_dir "data_orly/models_paper/CVAE_0005" --data /data/data/arnault/data/final_data/TO_LFPO_test_final.pkl --plot_path data_orly/figures/paper/compare_profile/final_plots/compare_profiles_CVAE_0005.png --typecodes B738 A321 A320 A319 A20N A21N




# typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
# list=()
# for elem in "${typecodes[@]}"; do 
#     list+=("data_orly/models_paper/VAE_${elem}_A320_010_3")
# done


# uv run data_orly/scripts/A_script_paper/mean_profile.py --onnx_vae_dir "${list[@]}"  --onnx_cvae_dir "data_orly/models_paper/CVAE_A320_010" --data /data/data/arnault/data/final_data/TO_LFPO_test_final.pkl --plot_path data_orly/figures/paper/compare_profile/final_plots/compare_profiles_CVAE_010_wvae.png --typecodes "${typecodes[@]}"

typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
typecodes_use=('B738' 'A321' 'A319' 'A20N' 'A21N')
list=("data_orly/models_paper/VAE_A320_0005")
for elem in "${typecodes_use[@]}"; do 
    list+=("data_orly/models_paper/VAE_${elem}_A320_010_3")
done


# uv run data_orly/scripts/A_script_paper/mean_profile.py --onnx_vae_dir "${list[@]}" --onnx_cvae_dir "data_orly/models_paper/CVAE_0005" --data /data/data/arnault/data/final_data/TO_LFPO_test_final.pkl --plot_path data_orly/figures/paper/compare_profile/final_plots/compare_profiles_CVAE_0005_wvae.png --typecodes "${typecodes[@]}"



uv run data_orly/scripts/A_script_paper/mean_profile.py --profile_path data_orly/figures/paper/compare_profile/final_plots/compare_profiles_CVAE_010_mean_profiles.csv --plot_path data_orly/figures/paper/compare_profile/final_plots/compare_profiles_CVAE_010.png 
uv run data_orly/scripts/A_script_paper/mean_profile.py  --profile_path data_orly/figures/paper/compare_profile/final_plots/compare_profiles_CVAE_0005_mean_profiles.csv --plot_path data_orly/figures/paper/compare_profile/final_plots/compare_profiles_CVAE_0005.png
uv run data_orly/scripts/A_script_paper/mean_profile.py --profile_path data_orly/figures/paper/compare_profile/final_plots/compare_profiles_CVAE_010_wvae_mean_profiles.csv --plot_path data_orly/figures/paper/compare_profile/final_plots/compare_profiles_CVAE_010_wvae.png
uv run data_orly/scripts/A_script_paper/mean_profile.py --profile_path  data_orly/figures/paper/compare_profile/final_plots/compare_profiles_CVAE_0005_wvae_mean_profiles.csv --plot_path data_orly/figures/paper/compare_profile/final_plots/compare_profiles_CVAE_0005.png



# typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
# typecodes_use=('B738' 'A321' 'A319' 'A20N' 'A21N')
# list=("data_orly/models_paper/VAE_A320_0005")
# for elem in "${typecodes_use[@]}"; do 
#     list+=("data_orly/models_paper/VAE_${elem}_A320_010_3")
# done


# uv run data_orly/scripts/A_script_paper/mean_profile.py  --onnx_cvae_dir "data_orly/models_paper/CVAE_full_A320_010_balanced" --data /data/data/arnault/data/final_data/TO_LFPO_test_final.pkl --plot_path data_orly/figures/paper/compare_profile/final_plots/compare_profiles_CVAE_010_balanced.png --typecodes "${typecodes[@]}"


# typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
# typecodes_use=('B738' 'A321' 'A319' 'A20N' 'A21N')
# list=("data_orly/models_paper/VAE_A320_0005")
# for elem in "${typecodes_use[@]}"; do 
#     list+=("data_orly/models_paper/VAE_${elem}_A320_010_3")
# done


# # uv run data_orly/scripts/A_script_paper/mean_profile.py  --spec 1 --onnx_cvae_dir "data_orly/models_paper/CVAE_full_A320__low_data_1" --data /data/data/arnault/data/final_data/TO_LFPO_test_final.pkl --plot_path data_orly/figures/paper/compare_profile/final_plots/compare_profiles_CVAE_balanced_old_spec.png --typecodes "${typecodes[@]}"




# # uv run data_orly/scripts/A_script_paper/mean_profile.py  --onnx_cvae_dir "data_orly/models_paper/CVAE_old_data_cond" --data /data/data/arnault/data/final_data/TO_LFPO_test_final.pkl --plot_path data_orly/figures/paper/compare_profile/final_plots/compare_profiles_CVAE_old_data_pseud_input_cond.png --typecodes "A320" "B738" --cond_pseudo 1


# uv run data_orly/scripts/A_script_paper/mean_profile.py  --onnx_cvae_dir data_orly/models_paper/CVAE_full_A320_0005_balanced --data /data/data/arnault/data/final_data/TO_LFPO_test_final.pkl --plot_path data_orly/figures/paper/compare_profile/final_plots/compare_profiles_CVAE_old_data_pseud_input_mean_weight.png --typecodes  "${typecodes[@]}" --cond_pseudo 0
