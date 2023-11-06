import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from scipy.signal import find_peaks, welch

from model import ConvNet
from loss import coral_loss
from dataset import levels_from_labelbatch
from dataset import proba_to_label


def hrv_feature_extractor(signals):
    fs = 512
    hrv_feature = []
    for signal in signals:
        peaks, _ = find_peaks(signal, distance=300, height=500)

        rr_intervals = np.diff(peaks)
        
        mean_rr = np.mean(rr_intervals)
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))

        diff_rr = np.diff(rr_intervals)

        # Poincaré plot
        sd1 = np.std(diff_rr) / np.sqrt(2)
        sd2 = np.std(rr_intervals[:-1] - rr_intervals[1:]) / np.sqrt(2)
        
        hrv_feature.append([mean_rr, sdnn, rmssd, sd1, sd2])
        
    return hrv_feature

def new_coral_training(dl_train, num_classes, scale_factor_list, important_weight_type, n_epochs=20):
    CORAL_Net = ConvNet(num_classes) 
    optimizer = optim.Adam(CORAL_Net.parameters())

    # Get the tensor shape of the model parameters
    parameter_tensor_shapes = [param.shape for param in CORAL_Net.parameters()]

    for epoch in range(n_epochs):
        CORAL_Net = CORAL_Net.train()
        for subject_id, batch_data in dl_train:
            inputs = [item["signal"] for item in batch_data]
            labels = [item["label"] for item in batch_data]

            # Create weight adjustment factor tensor
            weight_adjustment_factor = [torch.ones(shape) for shape in parameter_tensor_shapes]

            inputs = torch.stack(inputs, dim=0)
            labels = torch.tensor(labels)
            
            inputs = inputs.unsqueeze(1)
            # Prediction on source data
            logits, probas = CORAL_Net(inputs)
            
            # Convert class labels for CORAL
            levels = levels_from_labelbatch(labels, num_classes=5)

            ##hard weighting
            if (important_weight_type == "random"):
                importance_weights = F.softmax(torch.randn(4),dim=-1)
            elif (important_weight_type == "hard1"):
                importance_weights =  torch.tensor([1., 1., 1., 1.], dtype=torch.float32)
            elif (important_weight_type == "hard2"):
                importance_weights =  torch.tensor([0.3, 0.2, 0.2, 0.3], dtype=torch.float32)
            elif (important_weight_type == "hard3"):
                importance_weights =  torch.tensor([0.4, 0.1, 0.1, 0.4], dtype=torch.float32)
            elif (important_weight_type == "hard4"):
                importance_weights =  torch.tensor([0.2, 0.3, 0.3, 0.2], dtype=torch.float32)
            elif (important_weight_type == "hard5"):
                importance_weights =  torch.tensor([0.1, 0.4, 0.4, 0.1], dtype=torch.float32)
            elif (important_weight_type == "hard6"):
                importance_weights =  torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
            
            # Compute loss
            regression_loss = coral_loss(logits, levels)

            scaling_factor = scale_factor_list[subject_id]
            
            # Adjustment of weight adjustment factors
            for factor in weight_adjustment_factor:
                factor *= scaling_factor
        
            # Apply weight adjustment factors to the gradient
            for param, factor in zip(CORAL_Net.parameters(), weight_adjustment_factor):
                param.grad *= factor
            
            
            optimizer.zero_grad()
            regression_loss.backward()
            optimizer.step()

            #print(f"Epoch [{epoch}/{n_epochs}], Regression Loss: {regression_loss.item():.4f}")
        
    return CORAL_Net

def new_compute_mae_and_mse(model, data_loader):
    # Set the model to evaluation mode
    with torch.no_grad():
        mae, mse, num_examples = 0., 0., 0., 0

        for subject_id, batch_data in data_loader:
            inputs = [item["signal"] for item in batch_data]
            labels = [item["label"] for item in batch_data]

            inputs = torch.stack(inputs, dim=0)
            labels = torch.tensor(labels)
            
            inputs = inputs.unsqueeze(1)
            # Prediction on source data
            logits, probas = model(inputs)
            
            predicted_labels = proba_to_label(probas).float()
                        
            num_examples += labels.size(0)

            mae += torch.sum(torch.abs(predicted_labels - labels))
            mse += torch.sum((predicted_labels - labels)**2)

        mae = mae / num_examples
        mse = mse / num_examples
    return mae, mse



