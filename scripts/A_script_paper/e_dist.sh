typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
list=()
for elem in "${typecodes[@]}"; do 
    list+=("models_paper/VAE_${elem}_A320_010_3")
done
uv run scripts/A_script_paper/e_dist.py --typecodes "${typecodes[@]}" --data_og "/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl" --CVAE_ONNX models_paper/CVAE_minority --dist_path figures/paper/results/e_dist/distances_010.csv --VAEs_ONNX "${list[@]}" --n_f 3000

typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
typecodes_use=('B738' 'A321' 'A319' 'A20N' 'A21N')
list=("models_paper/VAE_A320_0005")
for elem in "${typecodes_use[@]}"; do 
    list+=("models_paper/VAE_${elem}_A320_010_3")
done

uv run scripts/A_script_paper/e_dist.py --typecodes "${typecodes[@]}" --data_og "/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl" --CVAE_ONNX models_paper/CVAE_0005 --dist_path figures/paper/results/e_dist/distances_0005.csv --VAEs_ONNX "${list[@]}" --n_f 3000
