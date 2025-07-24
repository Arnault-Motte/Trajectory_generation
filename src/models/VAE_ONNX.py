import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.distributions as distrib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.core.early_stop import Early_stopping
from src.core.loss import *
from src.core.networks import *
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from traffic.core import tqdm


## Encoder
class TCN_encoder(nn.Module):
    def __init__(
        self,
        inital_channels: int,
        latent_channels: int,
        latent_dim: int,
        kernel_size: int,
        stride: int,
        dilatation: int,
        dropout: float,
        nb_blocks: int,
        avgpoolnum: int,
        seq_len: int,
    ):
        super(TCN_encoder, self).__init__()
        self.tcn = TCN(
            inital_channels,
            latent_channels,
            latent_channels,
            kernel_size,
            stride,
            dilatation,
            dropout,
            nb_blocks,
        )
        self.avgpool = nn.AvgPool1d(avgpoolnum)
        self.dense_dim = latent_channels * seq_len // avgpoolnum
        self.mu_layer = nn.Linear(self.dense_dim, latent_dim)
        self.logvar_layer = nn.Linear(self.dense_dim, latent_dim)
        self.flaten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)
        x = self.avgpool(x)
        x = self.flaten(x)
        mu = self.mu_layer(x)
        log_var = self.logvar_layer(x)
        return mu, log_var


## Decoder
class TCN_decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        kernel_size: int,
        stride: int,
        dilatation: int,
        dropout: float,
        nb_blocks: int,
        upsamplenum: int,
        seq_len: int,
    ):
        super(TCN_decoder, self).__init__()
        self.seq_len = seq_len
        self.first_dense = nn.Linear(
            latent_dim, in_channels * (seq_len // upsamplenum)
        )
        self.upsample = nn.Upsample(scale_factor=upsamplenum)
        self.tcn = TCN(
            in_channels,
            latent_dim,
            out_channels,
            kernel_size,
            stride,
            dilatation,
            dropout,
            nb_blocks,
        )
        self.sampling_factor = upsamplenum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_dense(x)
        b, _ = x.size()
        x = x.view(b, -1, int(self.seq_len / self.sampling_factor))
        x = self.upsample(x)
        x = self.tcn(x)
        return x


class VAE_ONNX:
    """
    Class used to load the VAE ONNX files
    """

    def __init__(self, onnx_dir: str, device: str = "cpu") -> None:
        self.device = device
        self.onnx_dir = onnx_dir

        # Load ONNX sessions
        self.encoder_sess = ort.InferenceSession(
            f"{onnx_dir}/encoder.onnx", providers=["CPUExecutionProvider"]
        )
        self.decoder_sess = ort.InferenceSession(
            f"{onnx_dir}/decoder.onnx", providers=["CPUExecutionProvider"]
        )
        self.pseudo_inputs_sess = ort.InferenceSession(
            f"{onnx_dir}/pseudo_inputs_layer.onnx",
            providers=["CPUExecutionProvider"],
        )

        self.prior_weights: torch.Tensor = torch.load(f"{onnx_dir}/prior_w.pt",map_location=torch.device('cpu'))

        encoder_input = self.encoder_sess.get_inputs()
        self.seq_len = encoder_input[0].shape[2]
        self.log_std : torch.Tensor = torch.load(f"{onnx_dir}/log_std.pt", map_location=torch.device('cpu'))
        print(f"Seq_len : {self.seq_len}")

    def encode(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_np = x.cpu().numpy().astype(np.float32)
        mu, log_var = self.encoder_sess.run(None, {"input_0": x_np})
        return torch.tensor(mu), torch.tensor(log_var)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z_np = z.cpu().numpy().astype(np.float32)
        x_recon = self.decoder_sess.run(None, {"input_0": z_np})[0]
        return torch.tensor(x_recon)

    def pseudo_inputs_latent(self) -> tuple[torch.Tensor, torch.Tensor]:
        pseudo_inputs = self.pseudo_inputs_sess.run(None, {})[0]

        pseudo_inputs_tensor = torch.tensor(pseudo_inputs)
        return self.encode(pseudo_inputs_tensor)  # wierd no view()

    def reparametrize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x = x.permute(0, 2, 1)
        mu_pseudo, log_var_pseudo = self.pseudo_inputs_latent()
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)  # no permute for x
        x_recon = self.decode(z)
        return x_recon, z, mu, log_var, mu_pseudo, log_var_pseudo

    def reproduce(self, x: torch.Tensor) -> torch.Tensor:
        x_recon, *_ = self.forward(x)
        return x_recon.permute(0, 2, 1)

    def reproduce_data(
        self,
        data: np.ndarray,
        batch_size: int,
        n_batch: int,
    ) -> torch.Tensor:
        """
        Generate the reproducted trajectories for every
        trajectory of data.
        Also returns the data_loader used
        """
        data1, _ = get_data_loader(
            data,
            labels=None,
            batch_size=batch_size,
            train_split=0.9,
            shuffle=False,
            num_worker=4,
        )
        i = 0
        batches_reconstructed = []
        for batch, labels_batch in data1:
            if i == n_batch:
                break
            b = batch
            labels_batch = labels_batch
            x_recon = self.reproduce(b)
            batches_reconstructed.append(x_recon)
            i += 1
        reproduced_data = torch.cat(batches_reconstructed, dim=0)
        return reproduced_data

    def sample_from_prior(
        self, num_sample: int
    ) -> torch.Tensor:  # modify sampling and test
        """ """
        mu, log_var = self.pseudo_inputs_latent()
        prior_weights = self.prior_weights.squeeze(0).cpu()
        # weights = prior_weights.squeeze().cpu().numpy()
        # weights /= weights.sum()

        mu = mu.cpu()
        log_var = log_var.cpu()

        distrib = create_mixture(mu, log_var, prior_weights)
        samples = distrib.sample((num_sample,))
        return samples

    def sample(
        self, num_samples: int, batch_size: int
    ) -> torch.Tensor:
        samples = []
        for _ in range(0, num_samples, batch_size):
            cur_batch = min(batch_size, num_samples - len(samples))

            z = self.sample_from_prior(cur_batch)

            x_gen = self.decode(z)
            samples.append(x_gen)
        return torch.cat(samples).permute(0, 2, 1)

    def get_pseudo_inputs_recons(self) -> torch.Tensor:
        pseudo_inputs = self.pseudo_inputs_sess.run(None, {})[0]
        pseudo_inputs_tensor: torch.Tensor = torch.tensor(
            pseudo_inputs
        ).permute(0, 2, 1)
        return self.reproduce(pseudo_inputs_tensor)

    def generate_from_specific_vamp_prior(
        self, vamp_index: int, num_traj: int
    ) -> torch.Tensor:
        mu, log_var = self.pseudo_inputs_latent()
        mean = mu[vamp_index]
        scale = torch.exp(0.5 * log_var[vamp_index])
        dist = distrib.Independent(distrib.Normal(mean, scale), 1)
        samples = dist.sample(torch.Size([num_traj]))
        x_gen = self.decode(samples)
        return x_gen.permute(0, 2, 1)

    def compute_loss(self,data:torch.Tensor):
            x_batch = data
            x_recon, z, mu, log_var, pseudo_mu, pseudo_log_var = self.forward(
                x_batch
            )

            loss = VAE_vamp_prior_loss(
                x_batch.permute(0, 2, 1),
                x_recon,
                z,
                mu,
                log_var,
                pseudo_mu,
                pseudo_log_var,
                scale=self.log_std.cpu(),
                vamp_weight=self.prior_weights.cpu(),
            )
            kl = vamp_prior_kl_loss(
                z,
                mu,
                log_var,
                pseudo_mu,
                pseudo_log_var,
                vamp_weight=self.prior_weights.cpu(),
            )
            mse = reconstruction_loss(x_batch.permute(0, 2, 1), x_recon)
            cur_size = x_batch.size(0)
            total_mse = mse.item() / cur_size
            kl_div = kl.mean().item()
            return loss.item(), kl_div, total_mse
