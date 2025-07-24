# uv run scripts/A_script_paper/plot_traffic.py --onnx_dir models_paper/CVAE_minority --plot_path figures/paper/traffic/traff_new_A320.png --typecodes  A320


# uv run scripts/A_script_paper/plot_traffic.py --onnx_vaes "models_paper/VAE_A320_0005" --onnx_dir  "models_paper/CVAE_0005"  --plot_path figures/paper/traffic/traff_A320_5.png --typecodes  A320

typecodes=('A320' 'B738' 'A321' 'A319' 'A20N' 'A21N')
typecodes_use=('B738' 'A321' 'A319' 'A20N' 'A21N')
list=("models_paper/VAE_A320_0005")
for elem in "${typecodes_use[@]}"; do 
    list+=("models_paper/VAE_${elem}_A320_010_3")
done

uv run scripts/A_script_paper/plot_traffic.py --onnx_vaes "${list[@]}" --onnx_dir  "models_paper/CVAE_0005" --plot_per_typecode 1 --plot_path figures/paper/traffic/traff_0005_all_typecodes.png --typecodes  "${typecodes[@]}"