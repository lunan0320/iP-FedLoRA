from opacus.accountants.utils import get_noise_multiplier
import numpy as np



def compute_noise_multiplier(target_epsilon, target_delta, global_epoch, local_epoch, batch_size, client_data_sizes):
    total_dataset_size = sum(client_data_sizes)
    sample_rate = batch_size / total_dataset_size 
    total_steps = (sum([global_epoch * local_epoch * (client_data_size / batch_size) for client_data_size in client_data_sizes]))
    #total_steps = (sum([global_epoch * local_epoch * 50 for client_data_size in client_data_sizes]))

    noise_multiplier = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        steps=total_steps, 
        accountant="rdp"
    )

    return noise_multiplier


def cal_sensitivity(clip, dataset_size):
    return 2 * clip / dataset_size

def Laplace(epsilon):
    return 1 / epsilon


def Gaussian_Simple(epsilon, delta):
    return np.sqrt(2 * np.log(1.25 / delta)) / epsilon

def calculate_noise_scale(mechanism, epsilon, delta, times):
    if mechanism == 'Laplace':
        epsilon_single_query = epsilon / times
        return Laplace(epsilon=epsilon_single_query)
    elif mechanism == 'Gaussian':
        epsilon_single_query = epsilon / times
        delta_single_query = delta / times
        return Gaussian_Simple(epsilon=epsilon_single_query, delta=delta_single_query)