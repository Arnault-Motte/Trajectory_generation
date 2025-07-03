import sys  # noqa: I001
import os

from tqdm import tqdm

CURRENT_PATH = os.getcwd()
sys.path.append(os.path.abspath(CURRENT_PATH))
print("Current working directory:", CURRENT_PATH)
print(os.path.dirname(__file__))


import argparse

import torch

from data_orly.src.generation.data_process import Data_cleaner,return_labels
from data_orly.src.generation.generation import Generator,ONNX_Generator
from data_orly.src.generation.models.CVAE_TCN_VampPrior import CVAE_TCN_Vamp
from data_orly.src.generation.models.CVAE_ONNX import CVAE_ONNX
from data_orly.src.generation.models.VAE_TCN_VampPrior import VAE_TCN_Vamp
from data_orly.src.generation.test_display import plot_traffic,vertical_rate_profile_2
from data_orly.src.simulation import Simulator
from traffic.core import Traffic


def main() -> None:
    parser = argparse.ArgumentParser(description="My script with arguments")

    parser.add_argument(
        "--weight_file",
        type=str,
        default="/home/arnault/traffic/data_orly/src/generation/models/saved_weights/limited_one_typecode/VAE_TCN_Vampprior_take_off_7_vr_direct_cond_A21N.pth",
        help="Path of the weight file",
    )
    parser.add_argument(
        "--scene_file",
        type=str,
        default="",
        help="Name of the scene file",
    )

    parser.add_argument(
        "--nf", type=int, default=2000, help="Number of flights to be generated"
    )
    parser.add_argument(
        "--typecodes",
        nargs="+",
        type=str,
        default=[],
        help="typecodes of interest",
    )
    parser.add_argument("--typecode_to_gen", type=str, default="")
    parser.add_argument(
        "--cond", type=int, default=0, help="Conditional model or not"
    )

    parser.add_argument(
        "--vertical_rate",
        type=int,
        default=0,
        help="Model trained with vertical rate",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data_orly/data/takeoffs_LFPO_07.pkl",
        help="path to the data",
    )

    parser.add_argument(
        "--cuda",
        type=int,
        default="0",
        help="Index of the cuda GPU to use",
    )

    parser.add_argument(
        "--l_dim",
        type=int,
        default=64,
        help="latent dim of the model",
    )

    parser.add_argument(
        "--pseudo_in",
        type=int,
        default=800,
        help="pseudo inputs num",
    )

    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    )  # noqa: F405
    print(device)
    print(args.typecodes)
    conditional = bool(args.cond)

    columns = ["track", "groundspeed", "timedelta"]
    columns += ["vertical_rate"] if args.vertical_rate else ["altitude"]
    # data_cleaner = Data_cleaner(
    #     args.data,
    #     columns=columns,
    #     chosen_typecodes=args.typecodes,
    #     airplane_types_num=10 if len(args.typecodes) ==0 else -1,
    # )
    data_cleaner = Data_cleaner(no_data=True, columns=columns)
    data_cleaner.load_scalers(args.weight_file.split(".")[0] + "_scalers.pkl")
    # print(data_cleaner.chosen_types)
    # data = data_cleaner.clean_data()
    print("start")

    seq_len = 200
    # in_channels = len(data_cleaner.columns)
    in_channels = 4
    output_channels = 64
    latent_dim = args.l_dim
    pooling_factor = 10
    stride = 1
    number_of_block = 4
    kernel_size = 16
    dilatation = 2
    dropout = 0.2
    pseudo_input_num = args.pseudo_in  # *1.5
    patience = 30
    min_delta = -100
    labels_latent = 16
    # labels = data_cleaner.return_labels()
    # labels_dim = labels.shape[1]
    # print("l ",labels_dim)
    labels_dim = 10
    # print(labels.shape)
    if not conditional:
        model = VAE_TCN_Vamp(
            in_channels,
            output_channels,
            latent_dim,
            kernel_size,
            stride,
            dilatation,
            dropout,
            number_of_block,
            pooling_factor,
            pooling_factor,
            seq_len,
            pseudo_input_num=pseudo_input_num,
            early_stopping=True,
            patience=patience,
            min_delta=min_delta,
            init_std=1,
        ).to(device)
    else:
        model = CVAE_TCN_Vamp(
            in_channels,
            output_channels,
            latent_dim,
            kernel_size,
            stride,
            dilatation,
            dropout,
            number_of_block,
            pooling_factor,
            pooling_factor,
            label_dim=labels_dim,
            label_latent=labels_latent,
            seq_len=seq_len,
            pseudo_input_num=pseudo_input_num,
            early_stopping=True,
            patience=patience,
            min_delta=min_delta,
            temp_save="best_model_1000.pth",
            conditioned_prior=True,
            num_worker=6,
        ).to(device)

    model.load_model(weight_file_path=args.weight_file)
    model.save_model_ONNX("test_ONNX")

    print("model_saved")

    data_cleaner.save_scalers(args.weight_file.split(".")[0] + "_scalers.pkl")

    t_test = Traffic.from_file("/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl")
    t_test = t_test.aircraft_data()
    t_test = t_test.query("typecode in ['B738', 'A320', 'A321', 'A319', 'A20N', 'A318', 'A21N', 'A359', 'E145', 'A333']")
    t_test= t_test.sample(1000)
    data_test = data_cleaner.clean_data_specific(t_test,False)
    l_test = return_labels(t_test,data_cleaner.one_hot) 

    data2 = torch.tensor(data_test, dtype=torch.float32)
    labels_tensor = torch.tensor(l_test, dtype=torch.float32) 
    print("comp loss")
    with torch.no_grad():
        model.eval()
        model_loss = model.compute_loss(data2,labels_tensor)
    print("loss: ",model_loss[0],"kl: ",model_loss[1],"recon ", model_loss[2])

    gen = Generator(model, data_cleaner)

    pseudo_in = model.get_pseudo_inputs_recons()
    traf = data_cleaner.output_converter(
                    pseudo_in, landing=True, lat_lon=True
    )



    plot_traffic(traf,"test_ONNX/pseudo_inputs_new_b_model.png")
    # chosen_label = args.typecode_to_gen
    # print(chosen_label)
    # # t = (
    # #     gen.generate_flight_for_label_vamp(chosen_label,vamp=2,n_points=500)
    # # )
    # t = gen.generate_n_flight_per_labels(labels=["B738"], n_points=500)
    # # if t.flight_ids is None:
    # #     t = t.assign_id().eval()

    # plot_traffic(t[0], "test_ONNX/traffic_old.png")

    # # print("end")

    # import numpy as np

    cvae_onnx = CVAE_ONNX("test_ONNX")
    # label = "B738"
    # labels_array = np.full(500, label).reshape(-1, 1)
    # transformed_label = data_cleaner.one_hot.transform(labels_array)

    # labels_final = torch.Tensor(transformed_label)
    # print(labels_final.shape)
    # # test gen
    # gen_traj = cvae_onnx.sample(500, 500, labels_final)

    # traf = data_cleaner.output_converter(gen_traj, landing=True, lat_lon=True)
    # plot_traffic(traf, "test_ONNX/traffic_new.png")

    # traj_recons = Traffic.from_file(
    #     "/data/data/arnault/data/final_data/TO_LFPO_test_final.pkl"
    # )
    
    # traj_recons = traj_recons.aircraft_data()
    # traj_recons=traj_recons.query("typecode == 'A320'")
    # traj_recons=traj_recons.sample(200)
    # traj_recons_a = data_cleaner.clean_data_specific(traj_recons,False)
    # labels = return_labels(traj_recons,data_cleaner.one_hot) 
    # tensor_reprod = cvae_onnx.reproduce_data(traj_recons_a,labels,500,1)[0]
    # traf_recons=data_cleaner.output_converter(tensor_reprod, landing=True, lat_lon=True)

    # plot_traffic(traf_recons,plot_path="test_ONNX/traffic_new_recons_A320.png")
    # plot_traffic(traj_recons,plot_path="test_ONNX/traffic_to_recons_A320.png")

    onnx_gen = ONNX_Generator(cvae_onnx,data_cleaner)


    
    


    import matplotlib.pyplot as plt
    from cartes.crs import Lambert93, PlateCarree
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    import numpy as np
    
    def turn_labels_tensor(list_labels:list,one_hot:OneHotEncoder)-> torch.Tensor:
        np_label = np.array(list_labels).reshape(-1,1)
        np_label   = one_hot.transform(np_label)
        tensor_label = torch.Tensor(np_label)
        return tensor_label

    p_w:torch.Tensor = model.prior_weights_layers.forward(turn_labels_tensor(['A320'],data_cleaner.one_hot).to(next(model.parameters()).device))
    print(p_w.tolist())
    print("list^")
    
    # label = turn_labels_tensor(['B738'], one_hot= data_cleaner.one_hot)
    # weights = cvae_onnx.get_prior_weight(label).flatten()

    # list_w = weights.tolist()

    # import heapq
    # top_n_index = [index for index, _ in heapq.nlargest(15, enumerate(list_w), key=lambda x: x[1])]
    # print('t',len(top_n_index))
    
    # with plt.style.context("traffic"):
    #     fig, axes = plt.subplots(3, 5, subplot_kw=dict(projection=Lambert93()))
    #     ax = axes.flatten()

    #     for i,traff in tqdm(enumerate(top_n_index)):
    #         traff_vamp = onnx_gen.generate_flight_for_label_vamp('B738',traff,100)
    #         traff_vamp.plot(ax[i], alpha=0.2, color="blue")
    #         ax[i].set_title(f"#{i} (vamp={traff})", fontsize=6)
    #     # plt.tight_layout()
    #     plt.savefig("test_ONNX/high_score_B738.png", dpi=3000)
    #     plt.show()
    
    #test with 112

    # with plt.style.context("traffic"):
    #     fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
    #     traff_vamp_112b = onnx_gen.generate_flight_for_label_vamp('B738',112,100)
    #     traff_vamp_112b.plot(ax, alpha=0.2, color="blue")
    #     plt.savefig("test_ONNX/vamp112_B738.png", dpi=3000)
    #     plt.show()

    # vertical_rate_profile_2(traff_vamp_112b,"test_ONNX/vamp112_B738_distance.png",distance=False)

    
    # with plt.style.context("traffic"):
    #     fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
    #     traff_vamp_112a = onnx_gen.generate_flight_for_label_vamp('A320',112,100)
    #     traff_vamp_112a.plot(ax, alpha=0.2, color="blue")
    #     vertical_rate_profile_2(traff_vamp_112a,"test_ONNX/vamp112_A320_distance.png",distance=False)
    #     plt.savefig("test_ONNX/vamp112_A320.png", dpi=3000)
    #     plt.show()
    
    traffs = onnx_gen.generate_n_flight_per_labels(["A320","A321","B738"],1000)
    other_f = gen.generate_n_flight_per_labels(labels=["B738"], n_points=500)
    plot_traffic(other_f[0],plot_path="test_ONNX/traffic_new_b_gen_B738.png")

    # vertical_rate_profile_2(traffs[0],"test_ONNX/A320_TO_profile.png",distance=False)
    # vertical_rate_profile_2(traffs[2],"test_ONNX/B738_TO_profile.png",distance=False)
    # vertical_rate_profile_2(traffs[1],"test_ONNX/A321_TO_profile.png",distance=False)

    # plot_traffic(traffs[2],plot_path="test_ONNX/traffic_new_gen_B738.png")
    # plot_traffic(traffs[0],plot_path="test_ONNX/traffic_new_gen_A320.png")
    # plot_traffic(traffs[1],plot_path="test_ONNX/traffic_new_gen_A321.png")

    #og data 

    data_t = Traffic.from_file(args.data)
    if 'typecode' not in data_t.data.columns:
        data_t = data_t.aircraft_data()


    vertical_rate_profile_2(data_t.query('typecode == "A320"').sample(1000),"test_ONNX/A320_TO_profile_data_t.png",distance=False)
    vertical_rate_profile_2(data_t.query('typecode == "B738"').sample(1000),"test_ONNX/B738_TO_profile.png",distance=False)
    vertical_rate_profile_2(data_t.query('typecode == "B738"').sample(1000),"test_ONNX/A321_TO_profile.png",distance=False)




    

    
    


    
    








if __name__ == "__main__":
    main()
