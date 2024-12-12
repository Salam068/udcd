# Printing out all outputs
from IPython.core.interactiveshell import InteractiveShell

from helper.CRA import *
InteractiveShell.ast_node_interactivity = 'all'

# PyTorch
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch import optim, cuda, tensor
from torch.utils.data import DataLoader
from torch.serialization import SourceChangeWarning

# Warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
for warning in [UserWarning, SourceChangeWarning, Warning]:
    warnings.filterwarnings("ignore", category=warning)

# Data science tools
import os
import numpy as np
from os import path
from importlib import import_module

# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

# Customized functions 
from utils import *
from models import *

# Additional imports for dual-mode training
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command line arguments
fname = "config_train.py"
configuration = import_module(fname.split(".")[0])
config = configuration.config

# Define necessary functions for dual-mode training

def get_accuracy(y_pred, y_actual):
    """Calculates the accuracy (0 to 1)

    Args:
    + y_pred (tensor): output from the model
    + y_actual (tensor): ground truth 

    Returns:
    + float: a value between 0 to 1
    """
    y_pred = torch.argmax(y_pred, axis=1)
    y_actual = torch.argmax(y_actual, axis=1)
    return (1/len(y_actual)) * torch.sum(y_pred == y_actual).float()

def update_teacher(student, teacher, alpha, global_step):
    """EMA update for teacher model."""
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def evaluate(model, dataloader, criterion, device="cuda"):
    """Evaluate the model on validation or test data."""
    model.eval()
    loss, acc = 0.0, 0.0
    actuals = []
    predictions = []
    
    with torch.no_grad():
        for data, target, *rest in dataloader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            outputs = model(data)
            loss += criterion(outputs, target).item() * data.size(0)
            acc += get_accuracy(outputs, target).item() * data.size(0)
            
            # For metrics
            preds = torch.argmax(outputs, dim=1)
            actuals.extend(torch.argmax(target, dim=1).cpu().numpy())
            predictions.extend(preds.cpu().numpy())
    
    loss /= len(dataloader.dataset)
    acc /= len(dataloader.dataset)
    
    # Compute additional metrics
    pre = precision_score(actuals, predictions, average='macro', zero_division=0)
    rec = recall_score(actuals, predictions, average='macro', zero_division=0)
    f1 = f1_score(actuals, predictions, average='macro', zero_division=0)
    
    return loss, acc, pre, rec, f1

def train_dual_mode(
    student,
    teacher,
    train_loader,
    test_loader,
    weights,
    class_index,
    logger,
    args,
    device="cuda",
):
    """Train models in dual-mode (student-teacher)"""
    print("\nStarting Dual-Mode Training...\n")
    student = student.to(device)
    teacher = teacher.to(device)
    
    # Set optimizer
    if args['optimizer'] == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, student.parameters()), 
            lr=args['lr'], 
            weight_decay=1e-5
        )
    elif args['optimizer'] == "amsgrad":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, student.parameters()), 
            lr=args['lr'], 
            weight_decay=1e-5, 
            amsgrad=True
        )
    elif args['optimizer'] == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, student.parameters()), 
            lr=args['lr'], 
            momentum=0.9, 
            weight_decay=1e-5
        )
    else:
        raise ValueError("Unsupported optimizer type.")
    
    # Set scheduler
    if args['scheduler'] == 'StepLR':
        sch = optim.lr_scheduler.StepLR(
            optimizer, step_size=args['epochs'] // 3, gamma=0.1
        )
    elif args['scheduler'] == 'CosineAnnealingLR':
        sch = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args['epochs']
        )
    elif args['scheduler'] == 'OneCycleLR':
        sch = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=3e-4, epochs=args['epochs'],
            steps_per_epoch=len(train_loader)
        )
    elif args['scheduler'] == 'CosineAnnealingWarmRestarts':
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5
        )
    else:
        raise ValueError("Unsupported scheduler type.")
    
    # Loss Functions
    weights = torch.Tensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    CRA = CRALoss(args).to(device)
    
    # Initialize teacher's weights as student's
    teacher.load_state_dict(student.state_dict())
    
    # Set teacher to eval mode and disable gradients
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    student.train()
    
    iter_num = 0
    
    acc_all = np.zeros(10)
    pre_all = np.zeros(10)
    rec_all = np.zeros(10)
    f1_all = np.zeros(10)
    
    for epoch in range(args['epochs']):
        train_running_loss = 0.0
        train_running_acc = 0.0
        
        tqdm_train_iterator = tqdm(
            enumerate(train_loader),
            desc=f"[train]{epoch+1}/{args['epochs']}",
            total=len(train_loader),
            ascii=True,
            leave=True,
            colour="green",
            position=0,
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
            mininterval=10
        )
        
        for batch_idx, (images, target) in tqdm_train_iterator:
    # Now it can unpack 3 values (source images, target images, target labels)
            # print(f"Batch {batch_idx}")
            # print(f"Input s_images: {s_images.shape if isinstance(s_images, torch.Tensor) else type(s_images)}")
            # print(f"Input t_images: {t_images.shape if isinstance(t_images, torch.Tensor) else type(t_images)}")
            # print(f"Target: {target.shape if isinstance(target, torch.Tensor) else type(target)}")
            # print(f"Path: {path}")
            
            s_images = images.to(device, non_blocking=True)
            t_images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass for student
            print(student(s_images))
            s_ftrs, s_logits = student(s_images)
            # Forward pass for teacher
            with torch.no_grad():
                t_ftrs, t_logits = teacher(t_images)
            
            # Supervised loss
            loss = criterion(s_logits, target)
            
            # Add consistency loss after n_distill epochs
            if epoch >= args['n_distill']:
                consistency_weight = args['consistency'] * sigmoid_rampup(epoch, args['consistency_rampup'])
                consistency_dist = uncertainity_loss(t_logits, s_logits)
                consistency_loss = consistency_weight * consistency_dist
                
                # Add CRA losses
                ccd_loss, relation_loss = CRA(s_ftrs, t_ftrs, index.cuda(), target, class_index, args['nce_p'], sample_idx.cuda())
                loss += consistency_loss + args['ccd_weight'] * ccd_loss + args['rel_weight'] * relation_loss
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            sch.step()
            
            # Update the teacher
            update_teacher(student, teacher, args['t_decay'], iter_num)
            iter_num += 1
            
            # Update running loss and accuracy
            train_running_loss += loss.item() * s_images.size(0)
            train_running_acc += get_accuracy(s_logits.detach(), target) * s_images.size(0)
            
            # Update progress bar
            tqdm_train_iterator.set_postfix(
                avg_train_acc=f"{train_running_acc / ((batch_idx + 1) * batch_size):0.4f}",
                avg_train_loss=f"{(train_running_loss / ((batch_idx + 1) * batch_size)):0.4f}"
            )
        
        # Evaluate on test data
        val_loss, val_acc, val_pre, val_rec, val_f1 = evaluate(student, val_loader, criterion, device=device)
        test_loss, test_acc, test_pre, test_rec, test_f1 = evaluate(student, test_loader, criterion, device=device)
        
        logger.info(f"Epoch {epoch+1}/{args['epochs']}: "
                    f"Train Loss: {train_running_loss / len(train_loader.dataset):.4f}, "
                    f"Train Acc: {train_running_acc / len(train_loader.dataset):.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                    f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Collect metrics for last 10 epochs
        if epoch >= args['epochs'] - 10:
            acc_all[epoch - (args['epochs'] - 10)] = test_acc
            pre_all[epoch - (args['epochs'] - 10)] = test_pre
            rec_all[epoch - (args['epochs'] - 10)] = test_rec
            f1_all[epoch - (args['epochs'] - 10)] = test_f1
        
        # Early stopping can be implemented here based on validation loss or accuracy
        # (Optional: Not implemented in this example)
    
    # Log average performance of the last 10 epochs
    logger.info("\nAverage performance of the last 10 epochs:")
    logger.info(f"Accuracy: {np.mean(acc_all):.6f}, "
                f"Precision: {np.mean(pre_all):.6f}, "
                f"Recall: {np.mean(rec_all):.6f}, "
                f"F1: {np.mean(f1_all):.6f}")
    
    student.eval()
    logger.info(" ** Dual-Mode Training complete **")
    print(" ** Dual-Mode Training complete **")

# Define the main training function for normal training
def train_normal(
    model_to_load,
    model,
    stop_criteria,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    test_loader,
    save_file_name,
    train_on_gpu,
    aux_logits,
    history=[],
    max_epochs_stop=10,
    n_epochs=30,
    epochs_prev=0,
    print_every=2
):
    """Train a PyTorch Model (Normal Mode)"""
    # Early stopping initialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf
    valid_best_acc = 0
    best_epoch = 0
    epochs = epochs_prev
    print(f'Starting Training...\n')
    # Set Timer
    overall_start = timer()
    # Main loop
    for epoch in range(n_epochs):
        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0
        test_loss = 0.0
        train_acc = 0
        valid_acc = 0
        test_acc = 0
        # Set to training
        model.train()
        start = timer() 
        # Training loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for ii, (data, target) in enumerate(pbar):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.to('cuda', non_blocking=True), target.to('cuda', non_blocking=True)
            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)
            # Loss and backpropagation of gradients 
            if aux_logits == False:
                loss = criterion(output, target)
            elif aux_logits == True:
                loss = 0
                for k in range(0, len(output)):
                    loss += (criterion(output[k], target)) / (2**k)
            loss.backward() 
            # Update the parameters
            optimizer.step()
            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)
            # Calculate accuracy by finding max log probability
            if aux_logits == False:
                _, pred = torch.max(torch.exp(output), dim=1)
            elif aux_logits == True:
                _, pred = torch.max(torch.exp(output[0]), dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item())
            # Release memory
            del output, data, target 
            del loss, accuracy, pred, correct_tensor 
        # After training loops ends, start validation
        epochs += 1
        # Set to evaluation mode
        model.eval()
        # Don't need to keep track of gradients
        with torch.no_grad():
            # Validation loop
            for data, target in val_loader:
                # Tensors to gpu
                if train_on_gpu:
                    data, target = data.to('cuda', non_blocking=True), target.to('cuda', non_blocking=True)
                # Forward pass
                output = model(data)
                # Validation loss
                if aux_logits == False or model_to_load == 'inception_v3':
                    loss = criterion(output, target)
                elif aux_logits == True and model_to_load != 'inception_v3':
                    loss = 0
                    for k in range(0, len(output)):
                        loss += (criterion(output[k], target)) / (2**k)
                # Multiply average loss times the number of examples in batch
                valid_loss += loss.item() * data.size(0)
                # Calculate validation accuracy
                if aux_logits == False or model_to_load == 'inception_v3':
                    _, pred = torch.max(torch.exp(output), dim=1)
                elif aux_logits == True and model_to_load != 'inception_v3':
                    _, pred = torch.max(torch.exp(output[0]), dim=1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                # Multiply average accuracy times the number of examples
                valid_acc += accuracy.item() * data.size(0)
            # Release memory
            del output, data, target
            del loss, accuracy, pred, correct_tensor
            # Test loop
            for data, target, *rest in test_loader:
                # Tensors to gpu
                if train_on_gpu:
                    data, target = data.to('cuda', non_blocking=True), target.to('cuda', non_blocking=True)
                # Forward pass
                out = model(data)
                if aux_logits == False or model_to_load == 'inception_v3':
                    loss = criterion(out, target)
                elif aux_logits == True and model_to_load != 'inception_v3':
                    loss = 0
                    for k in range(0,len(out)):
                        loss += (criterion(out[k], target))/(2**k)
                test_loss += loss.item() * data.size(0)
                if aux_logits == False or model_to_load == 'inception_v3':
                    output = torch.exp(out)
                elif aux_logits == True and model_to_load != 'inception_v3':
                    output = torch.exp(out[0])
                # For metrics
                _, temp_label = torch.max(output.cpu(), dim=1)
                correct_tensor = temp_label.eq(target.data.view_as(temp_label))
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                test_acc += accuracy.item() * data.size(0)
                del output, data, target
                del loss, accuracy, pred, correct_tensor 
            # Calculate average losses
            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(val_loader.dataset)
            test_loss = test_loss / len(test_loader.dataset)
            # Calculate average accuracy
            train_acc = train_acc / len(train_loader.dataset)
            valid_acc = valid_acc / len(val_loader.dataset)
            test_acc = test_acc / len(test_loader.dataset)
            #
            history.append([train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc]) 
            # Print training and validation results
            if (epoch + 1) % print_every == 0:
                print(f'Training Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f} \tTest Loss: {test_loss:.4f}')
                print(f'Training Accuracy: {100 * train_acc:.2f}% \tValidation Accuracy: {100 * valid_acc:.2f}% \tTest Accuracy: {100 * test_acc:.2f}%\n')
            # Early Stopping
            if stop_criteria == 'loss': 
                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    checkpoint = { 
                        'epoch': epochs,
                        'loss': train_loss,
                        'accuracy': train_acc,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }
                    torch.save(checkpoint, save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch
                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(f'\nEarly Stopping! Total epochs: {epoch + 1}. Best epoch: {best_epoch + 1} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%')
                        total_time = timer() - overall_start
                        print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch + 1):.2f} seconds per epoch.')
                        # Load the best state dict
                        checkpoint = torch.load(save_file_name)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        # Attach the optimizer
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        # Format history
                        history = pd.DataFrame(history, columns=['train_loss', 'val_loss', 'test_loss', 'train_acc','val_acc', 'test_acc'])
                        return model, history
            elif stop_criteria == 'accuracy': 
                if valid_acc > valid_best_acc:
                    # Save model
                    checkpoint = { 
                        'epoch': epochs,
                        'loss': train_loss,
                        'accuracy': train_acc,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }
                    torch.save(checkpoint, save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch
                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(f'\nEarly Stopping! Total epochs: {epoch + 1}. Best epoch: {best_epoch + 1} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%')
                        total_time = timer() - overall_start
                        print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch + 1):.2f} seconds per epoch.')
                        # Load the best state dict
                        checkpoint = torch.load(save_file_name)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        # Attach the optimizer
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        # Format history
                        history = pd.DataFrame(history, columns=['train_loss', 'val_loss', 'test_loss', 'train_acc', 'val_acc', 'test_acc'])
                        return model, history
        # Update Scheduler
        scheduler.step(valid_loss)
                
    # Load the best state dict
    model.load_state_dict(torch.load(save_file_name)['model_state_dict']) 
    # Attach the optimizer 
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print('==============================================================')
    print(f'Best epoch: {best_epoch + 1} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%')
    print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch + 1):.2f} seconds per epoch.')
    # Format history
    history = pd.DataFrame(history, columns=['train_loss', 'val_loss', 'test_loss', 'train_acc', 'val_acc', 'test_acc'])
    return model, history

# Define the main execution
if __name__ == '__main__':
    ################## Network hyper-parameters 
    parentdir = config['parentdir']                     # root directory
    isPretrained = config['ImageNet']                   # set to 'True' to use pretrained weights or set to 'False' to train from scratch
    model_mode = config['model_mode']                   # 'custom_CNN' | 'custom_ONN' | 'import_Torch' | 'import_TIMM'
    q_order = config['q_order']                         # qth order Maclaurin approximation, common values: {1,3,5,7,9}. q=1 is equivalent to CNN
    ONN = config['ONN']                                 # set to 'True' if you are using ONN
    isDual = config['train_dual']                       # set to 'True' for dual-mode training
    input_ch = config['input_ch']                       # 1 for grayscale images, 3 for RGB images, 4 for RGBA images with an Alpha channel
    batch_size = config['batch_size']                   # Batch size, Change to fit hardware, common values: {4,8,16} for 2D datasets
    input_mean = config['input_mean']                   # Dataset mean
    input_std = config['input_std']                     # Dataset std
    loss_func = config['loss_func']                     # 'MSELoss', 'CrossEntropyLoss', etc. (https://pytorch.org/docs/stable/nn.html)
    optim_fc = config['optim_fc']                       # 'Adam', 'SGD', etc. (https://pytorch.org/docs/stable/optim.html)
    # optim_scheduler = config['optim_scheduler']         # 'ReduceLROnPlateau', etc. (https://pytorch.org/docs/stable/optim.html)
    final_activation_func = config['final_activation_func']  # 'Sigmoid', 'Softmax', etc. (https://pytorch.org/docs/stable/nn.html)
    lr = config['lr']                                   # Learning rate
    stop_criteria = config['stop_criteria']             # Stopping criteria: 'loss' or 'Accuracy'
    n_epochs = config['n_epochs']                       # Number of training epochs
    epochs_patience = config['epochs_patience']         # If val loss did not decrease for a number of epochs then decrease learning rate by a factor of lr_factor
    lr_factor = config['lr_factor']                     # Learning factor
    max_epochs_stop = config['max_epochs_stop']         # Maximum number of epochs with no improvement in validation metric for early stopping
    num_folds = config['num_folds']                     # Number of cross-validation folds
    Resize_h = config['Resize_h']                       # Network input (Image) height
    Resize_w = config['Resize_w']                       # Network input (Image) width
    load_weights = config['load_weights']               # Specify path of pretrained model weights or set to False to train from scratch
    model_to_load = config['model_to_load']             # Choose one of the models specified in config file
    model_name = config['model_name']                   # Choose a unique name for result folder
    aux_logits = config['aux_logits']                   # Required for models with auxiliary outputs (e.g., InceptionV3)
    fold_to_run = config['fold_to_run']                 # Define as [] to loop through all folds, or specify start and end folds i.e. [3, 5]
    encoder = config['encoder']                         # Set to 'True' if you retrain Seg. model encoder as a classifier
    outdir = config['outdir']                           # The destination directory for saving the pipeline outputs (models, results, plots, etc.)
    
    # Additional parameters for dual-mode training
    args = {
        # # Paths and logging
        # 'root_path': '../Datasets/APTOS/APTOS_images/train_images',
        # 'csv_file_path': '../CSVs/',
        # 'logdir': './logs/aptos/',
        # 'dataset': 'aptos',
        # 'split': 'split1',

        # Distillation parameters
        'n_distill': 20,  # Start using the KLD loss
        'consistency': 1.0,  # Weight for consistency loss
        'consistency_rampup': 30,  # Consistency ramp-up duration

        # NCE (Noise Contrastive Estimation) parameters
        'nce_p': 1,  # Number of positive samples for NCE
        'nce_k': 4096,  # Number of negative samples for NCE
        'nce_t': 0.07,  # Temperature parameter for softmax
        'nce_m': 0.5,  # Momentum for non-parametric updates

        # CCD (Contrastive Center Distillation) and Relation Loss
        'CCD_mode': 'sup',  # Supervised or unsupervised CCD
        'rel_weight': 25.0,  # Weight for relation loss
        'ccd_weight': 0.1,  # Weight for CCD loss

        # Anchor parameters
        'anchor_type': 'center',  # Anchor type: 'center' or 'class'
        'class_anchor': 30,  # Number of anchors in each class

        # Model dimensions
        'feat_dim': 128,  # Reduced feature dimension
        's_dim': 128,  # Feature dimension of the student model
        't_dim': 128,  # Feature dimension of the EMA teacher

        # Training data and EMA decay
        'n_data': 3662,  # Total number of training samples
        't_decay': 0.99,  # EMA decay for the teacher model

        # Training parameters
        'epochs': 80,  # Maximum number of training epochs
        'batch_size': 64,  # Batch size per GPU
        'drop_rate': 0.0,  # Dropout rate
        'lr': 1e-4,  # Learning rate
        'seed': 2024,  # Random seed

        # Optimizer and scheduler
        'optimizer': 'adam',  # Optimizer type
        'scheduler': 'OneCycleLR',  # Scheduler type

        # Device
        'device': 'cuda:0',  # Device to use for training
    }
    
    # Whether to train on a gpu
    train_on_gpu = cuda.is_available()
    print(f'Train on GPU: {train_on_gpu}')
    
    device = 'cuda' if train_on_gpu else 'cpu'
    
    if train_on_gpu:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} GPUs detected.')
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False 
    else:
        multi_gpu = False

    
    if isDual:
        print("Using dual training")
    else:
        print("Using normal training")
    
    # Proceed only if not cancelled
    # Remove or comment out the following line to enable training
    # raise ValueError("cancelled training")
    
    if (model_mode == 'custom_CNN') or (model_mode == 'import_Torch') or (model_mode == 'import_TIMM'):
        ONN = False
        q_order = 1
    
    if ((model_mode == 'import_Torch') or (model_mode == 'import_TIMM')) and (isPretrained == True):
        input_ch = 3
    
    traindir = os.path.join(parentdir, 'Data', 'Train')
    testdir = os.path.join(parentdir, 'Data', 'Test')
    valdir = os.path.join(parentdir, 'Data', 'Val')
    
    Results_path = os.path.join(outdir, 'Results')
    # Create Results Directory 
    os.makedirs(Results_path, exist_ok=True)
    
    # Loop through folds
    if not fold_to_run:
        loop_start, loop_end = 1, num_folds + 1
    else:
        loop_start, loop_end = fold_to_run[0], fold_to_run[1] + 1
    
    for fold_idx in range(loop_start, loop_end):
        if fold_idx == loop_start:
            print(f'Training using {model_to_load} network')
        print(f'Starting Fold {fold_idx}...')
        
        # Create Save Directory
        save_path = os.path.join(Results_path, model_name, f'fold_{fold_idx}')
        os.makedirs(save_path, exist_ok=True)
        save_file_name = os.path.join(save_path, f'{model_name}_fold_{fold_idx}.pt')
        checkpoint_name = os.path.join(save_path, f'checkpoint_{fold_idx}.pt')
        
        traindir_fold = os.path.join(traindir, f'fold_{fold_idx}')
        testdir_fold = os.path.join(testdir, f'fold_{fold_idx}')
        valdir_fold = os.path.join(valdir, f'fold_{fold_idx}')
        
        # Image Transformation
        if ONN:
            if input_ch == 3:
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(), 
                    transforms.Resize((Resize_h, Resize_w), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor()
                ])
                my_test_transforms = my_transforms  
            elif input_ch == 1:
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(), 
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((Resize_h, Resize_w), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor()
                ])
                my_test_transforms = my_transforms
        else:
            if input_ch == 1 and len(input_mean) == 3: 
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(), 
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((Resize_h, Resize_w), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=input_mean, std=input_std)  # gray
                ])
                my_test_transforms = my_transforms
            elif input_ch == 1:
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(), 
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((Resize_h, Resize_w), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=input_mean, std=input_std)  # gray
                ])
                my_test_transforms = my_transforms
            else: 
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(), 
                    transforms.Resize((Resize_h, Resize_w), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=input_mean, std=input_std)  # 3 channel
                ])
                my_test_transforms = my_transforms
        
        # Create labels
        categories, n_Class_train, img_names_train, labels_train, class_to_idx, idx_to_class = Createlabels(traindir_fold)
        labels_train = torch.from_numpy(labels_train).to(torch.int64)
        class_num = len(categories)
        _, n_Class_val, img_names_val, labels_val, _, _ = Createlabels(valdir_fold)
        labels_val = torch.from_numpy(labels_val).to(torch.int64)
        _, n_Class_test, img_names_test, labels_test, _, _ = Createlabels(testdir_fold)
        labels_test = torch.from_numpy(labels_test).to(torch.int64)
        
        # print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@ {len(categories)}")
        num_classes = len(categories)
        config2 = {
        # ... existing configurations ...
        'train_dual': True,
        'optimizer': 'adam',
        'scheduler': 'StepLR',
        'n_distill': 10,
        'consistency': 0.1,
        'consistency_rampup': 5,
        'ccd_weight': 1.0,
        'rel_weight': 1.0,
        't_decay': 0.99,
        'nce_p': 1,
        'weights': [1.0] * num_classes,  # Adjust as per class imbalance
        'class_index': list(range(num_classes)),  # Adjust as needed
        # ... other configurations ...
        }

        # Dataloaders
        # Train Dataloader 
        train_ds = MyData(
            root_dir=traindir_fold, 
            categories=categories, 
            img_names=img_names_train, 
            target=labels_train, 
            my_transforms=my_transforms, 
            return_path=False, 
            ONN=ONN, 
            mean=input_mean, 
            std=input_std
        )
        if (len(train_ds) / batch_size) == 0:
            train_dl = DataLoader(
                train_ds, 
                batch_size=batch_size, 
                shuffle=True, 
                pin_memory=True, 
                num_workers=1
            ) 
        else:
            train_dl = DataLoader(
                train_ds, 
                batch_size=batch_size, 
                shuffle=True, 
                pin_memory=True, 
                num_workers=1, 
                drop_last=True
            )  
        # Validation Dataloader 
        val_ds = MyData(
            root_dir=valdir_fold, 
            categories=categories, 
            img_names=img_names_val, 
            target=labels_val, 
            my_transforms=my_test_transforms, 
            return_path=False, 
            ONN=ONN, 
            mean=input_mean, 
            std=input_std
        )
        val_dl = DataLoader(
            val_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=1
        )
        # Test Dataloader
        test_ds = MyData(
            root_dir=testdir_fold, 
            categories=categories, 
            img_names=img_names_test, 
            target=labels_test, 
            my_transforms=my_test_transforms, 
            return_path=True, 
            ONN=ONN, 
            mean=input_mean, 
            std=input_std
        )
        test_dl = DataLoader(
            test_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=1
        )
    
        # Release Memory (delete variables)
        del n_Class_train, img_names_train, labels_train
        del n_Class_val, img_names_val, labels_val 
    
        # Load Model(s)
        if isDual:
            # Dual-mode: Initialize student and teacher models
            try:
                print('Loading previously trained student model weights from local directory...')
                student = get_pretrained_model(
                    parentdir, model_to_load, model_mode, isPretrained, 
                    input_ch, class_num, final_activation_func, 
                    train_on_gpu, multi_gpu, q_order
                )
                # checkpoint = torch.load(load_weights)
                # student.load_state_dict(checkpoint['model_state_dict'])
                # epochs_prev = checkpoint['epoch']
                # print(f'Student model had been trained previously for {epochs_prev} epochs.\n')
                if encoder:
                    student = EncoderModel(student.encoder, class_num)  # Adjust as per your EncoderModel definition
                    student = student.to(device)
            except:
                raise ValueError("The shape of the loaded weights do not exactly match with the student model framework.")
            
            # Initialize teacher model
            teacher = get_pretrained_model(
                parentdir, model_to_load, model_mode, isPretrained, 
                input_ch, class_num, final_activation_func, 
                train_on_gpu, multi_gpu, q_order
            )
            teacher.load_state_dict(student.state_dict())  # Initialize teacher with student weights
            for param in teacher.parameters():
                param.requires_grad = False
            teacher = teacher.to(device)
        else:
            # Normal training
            if load_weights: 
                try:
                    print('Loading previously trained model weights from local directory...')
                    model = get_pretrained_model(
                        parentdir, model_to_load, model_mode, isPretrained, 
                        input_ch, class_num, final_activation_func, 
                        train_on_gpu, multi_gpu, q_order
                    )
                    checkpoint = torch.load(load_weights)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    epochs_prev = checkpoint['epoch']
                    print(f'Model had been trained previously for {epochs_prev} epochs.\n')
                    if encoder:
                        model = EncoderModel(model.encoder, class_num)  # Adjust as per your EncoderModel definition
                        model = model.to(device)
                except:
                    raise ValueError("The shape of the loaded weights do not exactly match with the model framework.")
            else: 
                model = get_pretrained_model(
                    parentdir, model_to_load, model_mode, isPretrained, 
                    input_ch, class_num, final_activation_func, 
                    train_on_gpu, multi_gpu, q_order
                )
                epochs_prev = 0
            # Check if model on cuda
            if next(model.parameters()).is_cuda:
                print('Model device: CUDA')
                print('==============================================================')
        
        # Define Loss Function
        if not isDual:
            if loss_func == 'CrossEntropyLoss':
                criterion = nn.CrossEntropyLoss(weight=None, reduction='mean', label_smoothing=0.0)
            elif loss_func == 'NLLLoss':
                criterion = nn.NLLLoss(weight=None, reduction='mean', ignore_index=-100)
            elif loss_func == 'MultiMarginLoss':
                criterion = nn.MultiMarginLoss(p=1, margin=1.0, weight=None, reduction='mean')
            else:
                raise ValueError('Choose a valid loss function from here: https://pytorch.org/docs/stable/nn.html#loss-functions')
            
            if train_on_gpu:
                criterion = criterion.to(device)
        
            # Define Optimizer
            if optim_fc == 'Adagrad':  
                optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10, foreach=None, maximize=False)
            elif optim_fc == 'Adam':  
                optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=False)
            elif optim_fc == 'AdamW':  
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, maximize=False, foreach=None, capturable=False)
            elif optim_fc == 'Adamax':  
                optimizer = torch.optim.Adamax(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, foreach=None, maximize=False)
            elif optim_fc == 'NAdam':  
                optimizer = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004, foreach=None)
            elif optim_fc == 'RAdam':  
                optimizer = torch.optim.RAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, foreach=None)
            elif optim_fc == 'RMSprop':  
                optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, foreach=None, maximize=False, differentiable=False)
            elif optim_fc == 'Rprop':  
                optimizer = torch.optim.Rprop(model.parameters(), lr=lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50), foreach=None, maximize=False)
            elif optim_fc == 'SGD': 
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0.0001, nesterov=False, maximize=False, foreach=None, differentiable=False)
            else:
                raise ValueError('The pipeline does not support this optimizer. Choose a valid optimizer function from here: https://pytorch.org/docs/stable/optim.html')
        
            # Define Scheduler
            if config.get('optim_scheduler') == 'ReduceLROnPlateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=lr_factor, patience=epochs_patience, verbose=True, 
                    threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
                )
            else:
                raise ValueError('The pipeline does not support this scheduler. Choose a valid scheduler from here: https://pytorch.org/docs/stable/optim.html')
        
        # Training
        if isDual:
            # Dual-mode training requires additional arguments
            train_dual_mode(
                student=student,
                teacher=teacher,
                train_loader=train_dl,
                test_loader=test_dl,
                weights=config2['weights'],
                class_index=config2['class_index'],
                logger=logger,
                args=args,
                device='cuda' if train_on_gpu else 'cpu',
            )
        else:
            # Normal training
            model, history = train_normal(
                model_to_load=model_to_load,
                model=model,
                stop_criteria=stop_criteria,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_dl,
                val_loader=val_dl,
                test_loader=test_dl,
                save_file_name=checkpoint_name,
                train_on_gpu=train_on_gpu,
                aux_logits=aux_logits,
                history=[],
                max_epochs_stop=max_epochs_stop,
                n_epochs=n_epochs,
                epochs_prev=epochs_prev,
                print_every=1
            )
    
            # Saving TrainModel
            TrainChPoint = {} 
            TrainChPoint['model'] = model                              
            TrainChPoint['history'] = history
            TrainChPoint['categories'] = categories
            TrainChPoint['class_to_idx'] = class_to_idx
            TrainChPoint['idx_to_class'] = idx_to_class
            torch.save(TrainChPoint, save_file_name) 
    
            # Training Results
            # Plot loss
            plt.figure(figsize=(8, 6))
            for c in ['train_loss', 'val_loss', 'test_loss']:
                plt.plot(history[c], label=c) 
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(save_path, f'LossPerEpoch_fold_{fold_idx}.png'))
            # plt.show()
            # Plot accuracy
            plt.figure(figsize=(8, 6))
            for c in ['train_acc', 'val_acc', 'test_acc']:
                plt.plot(100 * history[c], label=c)
            plt.legend()
            plt.xlabel('Epoch') 
            plt.ylabel('Accuracy') 
            plt.savefig(os.path.join(save_path, f'AccuracyPerEpoch_fold_{fold_idx}.png'))
            # plt.show()
    
            # Release memory
            del my_transforms, optimizer, scheduler
            del train_ds, train_dl, val_ds, val_dl
            del img_names_test, labels_test 
            del TrainChPoint
            torch.cuda.empty_cache()
    
            # Compute cumulative confusion matrix
            all_paths = list()
            test_acc = 0.0
            test_loss = 0.0
            i = 0
            model.eval() 
            all_targets = []
            pred_probs = []
            pred_label = []
            for data, targets, im_path in test_dl:
                # Tensors to gpu
                if train_on_gpu:
                    data, targets = data.to('cuda', non_blocking=True), targets.to('cuda', non_blocking=True)
                # Raw model output
                out = model(data)
                if aux_logits == False or model_to_load == 'inception_v3':
                    loss = criterion(out, targets)
                elif aux_logits == True and model_to_load != 'inception_v3':
                    loss = 0
                    for k in range(0,len(out)):
                        loss += (criterion(out[k], targets))/(2**k)
                test_loss += loss.item() * data.size(0)
                if aux_logits == False or model_to_load == 'inception_v3':
                    output = torch.exp(out)
                elif aux_logits == True and model_to_load != 'inception_v3':
                    output = torch.exp(out[0])
                # For metrics
                all_paths.extend(im_path)
                targets = targets.cpu()
                if i == 0:
                    all_targets = targets.numpy()
                    pred_probs = output.cpu().detach().numpy()
                else:
                    all_targets = np.concatenate((all_targets, targets.numpy()))
                    pred_probs = np.concatenate((pred_probs, output.cpu().detach().numpy()))
                _, temp_label = torch.max(output.cpu(), dim=1)
                pred_label.extend(temp_label.numpy())
                i += 1
                # Release memory
                del out, data, targets, loss, output, temp_label
            test_loss = test_loss / len(test_dl.dataset)
            test_loss = round(test_loss, 4)
            test_acc = accuracy_score(all_targets, pred_label)
            test_acc = round(test_acc * 100, 2)
            print(f'Test Loss: {test_loss},  Test Accuracy: {test_acc}%')
    
            # Confusion Matrix
            cm = confusion_matrix(all_targets, pred_label)
            cm_per_class = confusion_matrix(all_targets, pred_label, labels=range(class_num))
            # Saving Test Results
            save_file_name = os.path.join(save_path, f'{model_name}_test_fold_{fold_idx}.pt')
            TestChPoint = {} 
            TestChPoint['categories'] = categories
            TestChPoint['class_to_idx'] = class_to_idx
            TestChPoint['idx_to_class'] = idx_to_class
            TestChPoint['Train_history'] = history 
            TestChPoint['n_Class_test'] = n_Class_test
            TestChPoint['targets'] = all_targets
            TestChPoint['prediction_label'] = pred_label
            TestChPoint['prediction_probs'] = pred_probs
            TestChPoint['image_names'] = all_paths 
            TestChPoint['cm'] = cm
            TestChPoint['cm_per_class'] = cm_per_class
            torch.save(TestChPoint, save_file_name)
    
            # Cumulative Confusion Matrix
            if fold_idx == loop_start:
                cumulative_cm = cm
            else:
                cumulative_cm += cm
    
            # Release memory
            del model, criterion, history, test_ds, test_dl
            del data, targets, out, output, temp_label
            del test_acc, test_loss, loss
            del pred_probs, pred_label, all_targets, all_paths
            del cm, cm_per_class, TestChPoint
            torch.cuda.empty_cache()
    
            # Delete checkpoint (optional)
            # os.remove(checkpoint_name)
            # print("Checkpoint File Removed!")
    
            print(f'Completed fold {fold_idx}')
    
    print('==============================================================')
    Overall_Accuracy = np.sum(np.diagonal(cumulative_cm)) / np.sum(cumulative_cm)
    Overall_Accuracy = round(Overall_Accuracy * 100, 2)
    print('Cumulative Confusion Matrix')
    print(cumulative_cm)
    print(f'Overall Test Accuracy: {Overall_Accuracy}')
    print('==============================================================')
