import math
import numpy as np

def dirichlet_quantity_process(label_vocab, label_assignment, client_num, alpha, data_length):
    label_indices = {label: np.where(label_assignment == label)[0] for label in label_vocab}
    for indices in label_indices.values():
        np.random.shuffle(indices)  

    partition_result = [[] for _ in range(client_num)]

    for label, indices in label_indices.items():
        label_length = len(indices)
        proportions = np.random.dirichlet([alpha] * client_num)
        label_counts = np.round(proportions * label_length).astype(int)
        
        if label_counts.sum() != label_length:
            diff = label_length - label_counts.sum()
            label_counts[np.argmax(label_counts)] += diff

        start = 0
        for i in range(client_num):
            end = start + label_counts[i]
            partition_result[i].extend(indices[start:end])
            start = end

    for client_data in partition_result:
        np.random.shuffle(client_data)

    return partition_result


def dirichlet_label_process(label_vocab, label_assignment, client_num, alpha, data_length):

    label_indices = {label: np.where(label_assignment == label)[0] for label in label_vocab}
    for indices in label_indices.values():
        np.random.shuffle(indices)  

    partition_result = [[] for _ in range(client_num)]
    

    label_distribution = np.random.dirichlet([alpha] * client_num, size=len(label_vocab)) 
    print(f'label distribution:{label_distribution}')
    for label_idx, (label, indices) in enumerate(label_indices.items()):
        label_count = len(indices)  
        client_proportions = label_distribution[label_idx]  
        
        
        client_sample_counts = np.round(client_proportions * label_count).astype(int)
    
        total_assigned = client_sample_counts.sum()
        if total_assigned != label_count:
            diff = label_count - total_assigned
            client_sample_counts[np.argmax(client_sample_counts)] += diff

        start = 0
        for i in range(client_num):
            end = start + client_sample_counts[i]
            partition_result[i].extend(indices[start:end])
            start = end


    for client_data in partition_result:
        np.random.shuffle(client_data)

    return partition_result


