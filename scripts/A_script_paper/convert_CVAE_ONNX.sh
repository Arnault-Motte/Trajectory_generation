#allows to the CVAE model as an ONNX file, only works with unconditioned version of CVAE

uv run scripts/A_script_paper/save_ONNX_and_scaler.py --scale 1 --weight_file src/generation/models/saved_weights/full/CVAE_full_A320_010_3.pth --save_dir models_paper/CVAE_minority2 --nf 200 --cond 1  --l_dim 64 --pseudo_in 1000 --cuda 1
uv run scripts/A_script_paper/save_ONNX_and_scaler.py --scale 1 --weight_file src/generation/models/saved_weights/full/CVAE_full_A320_0005.pth --save_dir models_paper/CVAE_005 --nf 200 --cond 1 --l_dim 64 --pseudo_in 1000 --cuda 1



uv run scripts/A_script_paper/save_ONNX_and_scaler.py --scaler 1 --weight_file src/generation/models/saved_weights/full/CVAE_full_A320__low_cond.pth --save_dir models_paper/CVAE_old_data_cond --nf 200 --cond 1 --l_dim 64 --pseudo_in 1000 --cuda 1 --cond_pseudo 1
