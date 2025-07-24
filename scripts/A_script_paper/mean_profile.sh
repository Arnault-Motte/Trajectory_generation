uv run scripts/A_script_paper/mean_profile.py --onnx_cvae_dir "models_paper/CVAE_A320_010" --data data/TO_LFPO_test_final_sub_samp3.pkl --plot_path figures/paper/compare_profile/final_plots/compare_profiles_CVAE_010.png --typecodes B738 A321 A320 A319 A20N A21N --n_f 2000
uv run scripts/A_script_paper/mean_profile.py --onnx_cvae_dir "models_paper/CVAE_0005" --data /data/data/arnault/data/final_data/TO_LFPO_test_final.pkl --plot_path figures/paper/compare_profile/final_plots/compare_profiles_CVAE_0005.png --typecodes B738 A321 A320 A319 A20N A21N




typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
list=()
for elem in "${typecodes[@]}"; do 
    list+=("models_paper/VAE_${elem}_A320_010_3")
done


uv run scripts/A_script_paper/mean_profile.py --onnx_vae_dir "${list[@]}"  --onnx_cvae_dir "models_paper/CVAE_A320_010" --data /data/data/arnault/data/final_data/TO_LFPO_test_final.pkl --plot_path figures/paper/compare_profile/final_plots/compare_profiles_CVAE_010_wvae.png --typecodes "${typecodes[@]}" 

typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
typecodes_use=('B738' 'A321' 'A319' 'A20N' 'A21N')
list=("models_paper/VAE_A320_0005")
for elem in "${typecodes_use[@]}"; do 
    list+=("models_paper/VAE_${elem}_A320_010_3")
done


uv run scripts/A_script_paper/mean_profile.py --onnx_vae_dir "${list[@]}" --onnx_cvae_dir "models_paper/CVAE_0005" --data /data/data/arnault/data/final_data/TO_LFPO_test_final.pkl --plot_path figures/paper/compare_profile/final_plots/compare_profiles_CVAE_0005_wvae.png --typecodes "${typecodes[@]}"



# uv run scripts/A_script_paper/mean_profile.py --profile_path figures/paper/compare_profile/final_plots/data/compare_profiles_CVAE_010_mean_profiles.csv --plot_path figures/paper/compare_profile/final_plots/compare_profiles_CVAE_010.png 
# uv run scripts/A_script_paper/mean_profile.py  --profile_path figures/paper/compare_profile/final_plots/data/compare_profiles_CVAE_0005_mean_profiles.csv --plot_path figures/paper/compare_profile/final_plots/compare_profiles_CVAE_0005.png
# uv run scripts/A_script_paper/mean_profile.py --profile_path figures/paper/compare_profile/final_plots/data/compare_profiles_CVAE_010_wvae_mean_profiles.csv --plot_path figures/paper/compare_profile/final_plots/compare_profiles_CVAE_010_wvae.png
# uv run scripts/A_script_paper/mean_profile.py --profile_path  figures/paper/compare_profile/final_plots/data/compare_profiles_CVAE_0005_wvae_mean_profiles.csv --plot_path figures/paper/compare_profile/final_plots/compare_profiles_CVAE_0005.png



uv run scripts/A_script_paper/mean_profile.py  --onnx_cvae_dir models_paper/CVAE_5_A320_0005_cond  --data data/TO_LFPO_test_final_sub_samp3.pkl --plot_path figures/paper/compare_profile/final_plots/compare_profiles_CVAE_pseudo_cond.png --typecodes  B738 A320 A321 A318 A21N --cond_pseudo 1 --n_f 2000
