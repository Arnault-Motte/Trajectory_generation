# uv run data_orly/scripts/A_script_paper/compute_loss.py --typecode "A320"   --loss_path "data_orly/figures/paper/results/loss/loss_010.csv" --data_og "/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl" --VAEs_ONNX data_orly/models_paper/VAE_A320_A320_010_3 #--CVAE_ONNX data_orly/models_paper/CVAE_minority2

# typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
# list=()
# for elem in "${typecodes[@]}"; do 
#     list+=("data_orly/models_paper/VAE_${elem}_A320_010_3")
# done

# uv run data_orly/scripts/A_script_paper/compute_loss.py --typecodes "${typecodes[@]}" --VAEs_ONNX "${list[@]}" --CVAE_ONNX data_orly/models_paper/CVAE_A320_010 --loss_path "data_orly/figures/paper/results/loss/loss_010.csv" --data_og "/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl"


typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
# typecodes_use=('B738' 'A321' 'A319' 'A20N' 'A21N')
# list=("data_orly/models_paper/VAE_A320_0005")
# for elem in "${typecodes_use[@]}"; do 
#     list+=("data_orly/models_paper/VAE_${elem}_A320_010_3")
# done

# uv run data_orly/scripts/A_script_paper/compute_loss.py --typecodes "${typecodes[@]}"  --CVAE_ONNX data_orly/models_paper/CVAE_full_A320_010_balanced --loss_path "data_orly/figures/paper/results/loss/loss_010_balanced.csv" --data_og "/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl"



typecodes=('A320' 'B738' 'A321' 'A318' 'A21N' )
typecodes_use=('B738' 'A321' 'A318' 'A21N')
list=("data_orly/models_paper/VAE_A320_0005")
for elem in "${typecodes_use[@]}"; do 
    list+=("data_orly/models_paper/VAE_${elem}_A320_010_3")
done

uv run data_orly/scripts/A_script_paper/compute_loss.py  --cond_pseudo 1 --typecodes "${typecodes[@]}" --VAEs_ONNX "${list[@]}"   --CVAE_ONNX data_orly/models_paper/CVAE_5_A320_0005_cond --loss_path "data_orly/figures/paper/results/loss/loss_0005_cond.csv" --data_og "/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl"