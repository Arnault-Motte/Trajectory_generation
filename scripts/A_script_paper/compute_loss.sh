# uv run data_orly/scripts/A_script_paper/compute_loss.py --typecode "A320"   --loss_path "data_orly/figures/paper/results/loss/loss_010.csv" --data_og "/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl" --VAEs_ONNX data_orly/models_paper/VAE_A320_A320_010_3 #--CVAE_ONNX data_orly/models_paper/CVAE_minority2

typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
typecodes_use=('B738' 'A321' 'A319' 'A20N' 'A21N')
list=("data_orly/models_paper/VAE_A320_0005")
for elem in "${typecodes_use[@]}"; do 
    list+=("data_orly/models_paper/VAE_${elem}_A320_010_3")
done

uv run data_orly/scripts/A_script_paper/compute_loss.py --typecodes "${typecodes[@]}" --VAEs_ONNX "${list[@]}" --CVAE_ONNX data_orly/models_paper/CVAE_005 --loss_path "data_orly/figures/paper/results/loss/loss_005.csv" --data_og "/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl"