typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
list=()
for elem in "${typecodes[@]}"; do 
    list+=("models_paper/VAE_${elem}_A320_010_3")
done

uv run scripts/A_script_paper/compute_loss.py --typecodes "${typecodes[@]}" --VAEs_ONNX "${list[@]}" --CVAE_ONNX models_paper/CVAE_A320_010 --loss_path "figures/paper/results/loss/loss_010.csv" --data_og "/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl"


typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
# typecodes_use=('B738' 'A321' 'A319' 'A20N' 'A21N')
# list=("models_paper/VAE_A320_0005")
# for elem in "${typecodes_use[@]}"; do 
#     list+=("models_paper/VAE_${elem}_A320_010_3")
# done

uv run scripts/A_script_paper/compute_loss.py --typecodes "${typecodes[@]}"  --CVAE_ONNX models_paper/CVAE_full_A320_010_balanced --loss_path "figures/paper/results/loss/loss_010_balanced.csv" --data_og "/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl"