typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
typecodes_use=('B738' 'A321' 'A319' 'A20N' 'A21N')
list=("models_paper/VAE_A320_0005")
for elem in "${typecodes_use[@]}"; do 
    list+=("models_paper/VAE_${elem}_A320_010_3")
done

# uv run scripts/A_script_paper/plot_latent_space.py --cond_pseudo 1  --typecodes "${typecodes[@]}" --onnx_cvae_dir models_paper/CVAE_old_data_cond --data /data/data/arnault/data/final_data/TO_LFPO_test_final.pkl --plot_path figures/paper/latent_space/CAVE_005_cond.png


# uv run scripts/A_script_paper/plot_latent_space.py --cond_pseudo 1  --typecodes B738 A320 A321 A318 A21N  --onnx_cvae_dir models_paper/CVAE_5_A320_0005_cond --data /data/data/arnault/data/final_data/TO_LFPO_test_final.pkl --plot_path figures/paper/latent_space/CAVE__full_005_cond.png

uv run scripts/A_script_paper/plot_latent_space.py --cond_pseudo 0  --typecodes B738 A320 A321 A318 A21N  --onnx_cvae_dir models_paper/CVAE_full_A320_0005_balanced --data /data/data/arnault/data/final_data/TO_LFPO_test_final.pkl --plot_path figures/paper/latent_space/CAVE_005_balanced.png

