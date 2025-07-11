# typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
# list=()
# for elem in "${typecodes[@]}"; do 
#     list+=("data_orly/models_paper/VAE_${elem}_A320_010_3")
# done
# uv run data_orly/scripts/A_script_paper/vertical_rates_all_models.py --typecodes "${typecodes[@]}" --data_og "/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl" --CVAE_ONNX data_orly/models_paper/CVAE_minority --plot_path data_orly/figures/paper/vertical_rates_comp.png --VAEs_ONNX "${list[@]}" --n_f 5000


typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
typecodes_use=('B738' 'A321' 'A319' 'A20N' 'A21N')
list=("data_orly/models_paper/VAE_A320_0005")
for elem in "${typecodes_use[@]}"; do 
    list+=("data_orly/models_paper/VAE_${elem}_A320_010_3")
done

uv run data_orly/scripts/A_script_paper/vertical_rates_all_models.py --typecodes "${typecodes[@]}" --data_og "/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl" --CVAE_ONNX data_orly/models_paper/CVAE_005 --plot_path data_orly/figures/paper/vertical_rates_comp_A320_0005.png --VAEs_ONNX "${list[@]}" --n_f 5000