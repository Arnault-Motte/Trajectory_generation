typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
list=()
for elem in "${typecodes[@]}"; do 
    list+=("models_paper/VAE_${elem}_A320_010_3")
done
uv run scripts/A_script_paper/vertical_rates_all_models.py --typecodes "${typecodes[@]}" --data_og "data/TO_LFPO_test_final_sub_samp3.pkl" --CVAE_ONNX models_paper/CVAE_0005 --plot_path figures/paper/vertical_rate/comparaison/vertical_rates_comp_testi.png --VAEs_ONNX "${list[@]}" --n_f 5000 


# typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
# typecodes_use=('B738' 'A321' 'A319' 'A20N' 'A21N')
# list=("models_paper/VAE_A320_0005")
# for elem in "${typecodes_use[@]}"; do 
#     list+=("models_paper/VAE_${elem}_A320_010_3")
# done

# uv run scripts/A_script_paper/vertical_rates_all_models.py --typecodes "${typecodes[@]}" --data_og "/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl" --CVAE_ONNX models_paper/CVAE_005 --plot_path figures/paper/vertical_rate/comparaison/vertical_rates_comp_A320_0005.png --VAEs_ONNX "${list[@]}" --n_f 5000




# typecodes=('A320' 'B738' 'A321' 'A318' 'A21N' )
# typecodes_use=('B738' 'A321' 'A318' 'A21N')
# list=("models_paper/VAE_A320_0005")
# for elem in "${typecodes_use[@]}"; do 
#     list+=("models_paper/VAE_${elem}_A320_010_3")
# done

# uv run scripts/A_script_paper/vertical_rates_all_models.py --typecodes "${typecodes[@]}" --data_og "/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl" --CVAE_ONNX models_paper/CVAE_5_A320_0005_cond --plot_path figures/paper/vertical_rate/comparaison/vertical_rates_comp_A320_0005_cond.png --VAEs_ONNX "${list[@]}" --n_f 5000 --cond_pseudo 1