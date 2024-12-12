import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import torch
import torch.nn as nn
import torch.optim as optim

from utils.losses import KLD, uncertainity_loss
from utils.CRA import CRALoss
#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# def get_accuracy(y_pred, y_actual):
#     """Calculates the accuracy (0 to 1)
#
#     Args:
#     + y_pred (tensor): output from the model (logits)
#     + y_actual (tensor): ground truth labels (class indices)
#
#     Returns:
#     + float: a value between 0 to 1
#     """
#     # Use argmax to get predicted class index from logits
#     y_pred = torch.argmax(y_pred, axis=1)
#
#     # No need to do argmax on y_actual since it's already class indices
#     return (y_pred == y_actual).float().mean().item()
#
#
# # def update_teacher(student, teacher, alpha, global_step):
# #     alpha = min(1 - 1 / (global_step + 1), alpha)
# #     for ema_param, param in zip(teacher.parameters(), student.parameters()):
# #         ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
#
#
#
# # def update_teacher(student, teacher, t_decay, iter_num):
# #     alpha = min(1 - 1 / (iter_num + 1), t_decay)  # Exponential decay factor
# #     for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
# #         # Check if shapes match
# #         if student_param.shape == teacher_param.shape:
# #             teacher_param.data.mul_(alpha).add_(1 - alpha, student_param.data)
# #         # else:
# #         #     print(f"Skipping update for layer with mismatch: Student {student_param.shape}, Teacher {teacher_param.shape}")
# #
#
# def resize_tensor(src, target_shape):
#     """
#     Resizes the source tensor to match the target shape across all dimensions.
#     """
#     if src.dim() == 2:  # Linear layer (e.g., fully connected)
#         in_features = src.shape[1]
#         out_features = target_shape[1]
#         linear = nn.Linear(in_features, out_features, bias=False).to(src.device)
#         return linear(src)
#     elif src.dim() == 4:  # Convolutional layer (e.g., feature maps)
#         in_channels = src.shape[1]
#         out_channels = target_shape[1]
#         conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False).to(src.device)
#         return conv(src)
#     else:
#         raise ValueError(f"Unsupported tensor dimensions: {src.dim()}. Cannot resize.")
#
# def update_teacher(student, teacher, t_decay, iter_num):
#     """
#     Dynamically update the teacher model parameters from the student model.
#     """
#     alpha = min(1 - 1 / (iter_num + 1), t_decay)  # Exponential decay factor
#     for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
#         if student_param.shape == teacher_param.shape:
#             # Update teacher with EMA when shapes match
#             teacher_param.data.mul_(alpha).add_(1 - alpha, student_param.data)
#         else:
#             # Handle mismatched layers dynamically
#             try:
#                 resized_student = resize_tensor(student_param.data, teacher_param.data.shape)
#                 teacher_param.data.mul_(alpha).add_(1 - alpha, resized_student)
#                 print(f"Resized {student_param.shape} to {teacher_param.shape} for EMA.")
#             except ValueError as e:
#                 print(f"Skipping mismatched layer: {e}")
#
#
#
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

#
# # def fit(
# #         student,
# #         teacher,
# #         train_dl,
# #         test_dl,
# #         weights,
# #         class_index,
# #         args,
# #         device="cpu",
# # ):
# #     print()
# #     student = student.to(device)
# #
# #     teacher = teacher.to(device)
# #
# #     # Set optimizer
# #     if args.optimizer == "adam":
# #         optimizer = optim.Adam(filter(
# #             lambda p: p.requires_grad, student.parameters()), lr=args.lr, weight_decay=1e-5)
# #     elif args.optimizer == "amsgrad":
# #         optimizer = optim.Adam(
# #             filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr, weight_decay=1e-5, amsgrad=True
# #         )
# #     elif args.optimizer == "sgd":
# #         optimizer = optim.SGD(
# #             filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-5
# #         )
# #
# #     # Set scheduler
# #     if args.scheduler == 'StepLR':
# #         sch = optim.lr_scheduler.StepLR(
# #             optimizer, step_size=args.epochs // 3, gamma=0.1)
# #     elif args.scheduler == 'CosineAnnealingLR':
# #         sch = optim.lr_scheduler.CosineAnnealingLR(
# #             optimizer, T_max=args.epochs)
# #     elif args.scheduler == 'OneCycleLR':
# #         sch = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, epochs=args.epochs,
# #                                             steps_per_epoch=len(train_dl))
# #     elif args.scheduler == 'CosineAnnealingWarmRestarts':
# #         sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
# #
# #     # Loss Functions
# #     weights = torch.Tensor(weights).to(device)
# #     criterion = nn.CrossEntropyLoss(weight=weights)
# #     CRA = CRALoss(args).to(device)
# #     # kld = KLD()
# #
# #     student.train()
# #     iter_num = 0
# #
# #     # early_stop = 0
# #     # best_loss = 1000.0
# #
# #     acc_all = np.zeros(10)
# #     pre_all = np.zeros(10)
# #     rec_all = np.zeros(10)
# #     f1_all = np.zeros(10)
# #
# #     for epoch in range(args.epochs):
# #         print("going to train epoch " , epoch)
# #         train_running_loss = 0
# #         train_running_acc = 0
# #
# #         tqdm_train_iterator = tqdm(train_dl, desc='Training')
# #         for batch_idx, (images, target) in enumerate(tqdm_train_iterator):
# #
# #             s_images = images.to(device)
# #             t_images = s_images
# #             target = target.to(device)
# #             optimizer.zero_grad()
# #
# #             s_ftrs, s_logits = student(s_images)
# #             t_ftrs, t_logits = teacher(t_images)
# #
# #             loss = criterion(s_logits, target)
# #
# #             if epoch >= args.n_distill:
# #                 consistency_weight = args.consistency * \
# #                     sigmoid_rampup(epoch, args.consistency_rampup)
# #                 consistency_dist = uncertainity_loss(t_logits, s_logits)
# #                 consistency_loss = consistency_weight * consistency_dist
# #
# #                 ccd_loss, relation_loss = CRA(s_ftrs, t_ftrs, index.cuda(
# #                 ), target, class_index, args.nce_p, sample_idx.cuda())
# #                 loss += consistency_loss
# #                 loss += args.ccd_weight * ccd_loss
# #                 loss += args.rel_weight * relation_loss
# #
# #             loss.backward()
# #
# #             optimizer.step()
# #             sch.step()
# #             update_teacher(student, teacher, args.t_decay, iter_num)
# #             iter_num += 1
# #
# #             train_running_loss += loss.item()
# #
# #             train_running_acc += get_accuracy(s_logits.detach(), target)
# #
# #             tqdm_train_iterator.set_postfix(avg_train_acc=f"{train_running_acc/(batch_idx+1):0.4f}",
# #                                             avg_train_loss=f"{(train_running_loss/(batch_idx+1)):0.4f}")
# #
# #         print('')
# #         print(f"Epoch: {epoch}")
# #         acc, pre, rec, f1 = test(
# #             student, test_dl, verbose=True, device=device)
# #
# #         if epoch >= args.epochs - 10:
# #             acc_all[epoch - (args.epochs - 10)] = acc
# #             pre_all[epoch - (args.epochs - 10)] = pre
# #             rec_all[epoch - (args.epochs - 10)] = rec
# #             f1_all[epoch - (args.epochs - 10)] = f1
# #
# #     print("\nAverage performance of the last 10 epochs:")
# #     print("\nAccuracy: {:6f}, Precision: {:6f}, Balanced Accuracy: {:6f}, F1: {:6f}"
# #                 .format(np.mean(acc_all), np.mean(pre_all), np.mean(rec_all), np.mean(f1_all)))
# #     student.eval()
# #     print(" ** Training complete **")
# #     print(" ** Training complete **")
#
#
# def fit(
#         student,
#         teacher,
#         train_dl,
#         test_dl,
#         weights,
#         class_index,
#         args,
#         device="cpu",
# ):
#     print()
#     student = student.to(device)
#     teacher = teacher.to(device)
#
#     # Set optimizer
#     if args.optimizer == "adam":
#         optimizer = optim.Adam(filter(
#             lambda p: p.requires_grad, student.parameters()), lr=args.lr, weight_decay=1e-5)
#     elif args.optimizer == "amsgrad":
#         optimizer = optim.Adam(
#             filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr, weight_decay=1e-5, amsgrad=True
#         )
#     elif args.optimizer == "sgd":
#         optimizer = optim.SGD(
#             filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-5
#         )
#
#     # Set scheduler
#     if args.scheduler == 'StepLR':
#         sch = optim.lr_scheduler.StepLR(
#             optimizer, step_size=args.epochs // 3, gamma=0.1)
#     elif args.scheduler == 'CosineAnnealingLR':
#         sch = optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=args.epochs)
#     elif args.scheduler == 'OneCycleLR':
#         sch = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, epochs=args.epochs,
#                                             steps_per_epoch=len(train_dl))
#     elif args.scheduler == 'CosineAnnealingWarmRestarts':
#         sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
#
#     # Loss Functions
#     weights = torch.Tensor(weights).to(device)
#     criterion = nn.CrossEntropyLoss(weight=weights)
#     CRA = CRALoss(args).to(device)
#
#     student.train()
#     iter_num = 0
#
#     acc_all = np.zeros(10)
#     pre_all = np.zeros(10)
#     rec_all = np.zeros(10)
#     f1_all = np.zeros(10)
#
#     for epoch in range(args.epochs):
#         print("going to train epoch " , epoch)
#         train_running_loss = 0
#         train_running_acc = 0
#
#         tqdm_train_iterator = tqdm(train_dl, desc='Training')
#         for batch_idx, (images, target) in enumerate(tqdm_train_iterator):
#             s_images = images.to(device)
#             t_images = s_images
#             target = target.to(device)
#             optimizer.zero_grad()
#
#             s_ftrs, s_logits = student(s_images)
#             t_ftrs, t_logits = teacher(t_images)
#
#             loss = criterion(s_logits, target)
#
#             if epoch >= args.n_distill:
#                 consistency_weight = args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)
#                 consistency_dist = uncertainity_loss(t_logits, s_logits)
#                 consistency_loss = consistency_weight * consistency_dist
#
#                 ccd_loss, relation_loss = CRA(s_ftrs, t_ftrs, target, class_index, args.nce_p)
#                 loss += consistency_loss
#                 loss += args.ccd_weight * ccd_loss
#                 loss += args.rel_weight * relation_loss
#
#             loss.backward()
#
#             optimizer.step()
#             sch.step()
#             update_teacher(student, teacher, args.t_decay, iter_num)
#             iter_num += 1
#
#             train_running_loss += loss.item()
#             train_running_acc += get_accuracy(s_logits.detach(), target)
#
#             tqdm_train_iterator.set_postfix(avg_train_acc=f"{train_running_acc/(batch_idx+1):0.4f}",
#                                             avg_train_loss=f"{(train_running_loss/(batch_idx+1)):0.4f}")
#
#         print('')
#         print(f"Epoch: {epoch}")
#         acc, pre, rec, f1 = test(
#             student, test_dl, verbose=True, device=device)
#
#         if epoch >= args.epochs - 10:
#             acc_all[epoch - (args.epochs - 10)] = acc
#             pre_all[epoch - (args.epochs - 10)] = pre
#             rec_all[epoch - (args.epochs - 10)] = rec
#             f1_all[epoch - (args.epochs - 10)] = f1
#
#     print("\nAverage performance of the last 10 epochs:")
#     print("\nAccuracy: {:6f}, Precision: {:6f}, Balanced Accuracy: {:6f}, F1: {:6f}"
#                 .format(np.mean(acc_all), np.mean(pre_all), np.mean(rec_all), np.mean(f1_all)))
#     student.eval()
#     print(" ** Training complete **")
#     print(" ** Training complete **")
#
#
#
# # Test the model
#
#
# def test(
#     net,
#     test_dl,
#     verbose=True,
#     device="cpu"
# ):
#     net = net.to(device)
#     net.eval()
#     criterion = nn.CrossEntropyLoss()
#
#     tqdm_test_iterator = tqdm(enumerate(test_dl),
#                                desc="[TEST]",
#                                total=len(test_dl),
#                                ascii=True, leave=True,
#                                colour="green", position=0,
#                                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
#                                mininterval=10)
#
#     test_running_loss = 0
#     test_running_acc = 0
#
#     actuals = []
#     predictions = []
#
#     for idx, (data, target) in tqdm_test_iterator:
#         data = data.to(device)
#         target = target.to(device)
#
#         # Get predictions from the model
#         _, y_pred = net(data)
#
#         # Compute loss
#         loss = criterion(y_pred, target)
#         test_running_loss += loss.item()
#
#         # Collect actual and predicted values
#         actuals.extend(target.to(device).numpy())  # target already contains class indices
#         predictions.extend(y_pred.argmax(dim=1).cpu().numpy())  # argmax over logits to get predicted class indices
#
#         # Calculate accuracy
#         test_running_acc += get_accuracy(y_pred.detach(), target)
#
#         tqdm_test_iterator.set_postfix(avg_test_acc=f"{test_running_acc/(idx+1):0.4f}",
#                                        avg_test_loss=f"{(test_running_loss/(idx+1)):0.4f}")
#
#     print("Test Loss: ", test_running_loss/len(test_dl))
#     actuals = np.array(actuals)
#     predictions = np.array(predictions)
#
#     # Calculate metrics
#     acc = accuracy_score(actuals, predictions)
#     pre = precision_score(actuals, predictions, average='macro')
#     rec = recall_score(actuals, predictions, average='macro')
#     f1 = f1_score(actuals, predictions, average='macro')
#
#     if verbose:
#         print("Accuracy: %6f, Precision: %6f, Recall: %6f, F1: %6f \n" %
#                     (acc, pre, rec, f1))
#
#     return acc, pre, rec, f1


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.CRA import CRALoss

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to calculate accuracy
def get_accuracy(y_pred, y_actual):
    y_pred = torch.argmax(y_pred, axis=1)
    return (y_pred == y_actual).float().mean().item()


# Function to update teacher model using EMA
def update_teacher(student, teacher, t_decay, iter_num):
    alpha = min(1 - 1 / (iter_num + 1), t_decay)
    for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
        if student_param.shape == teacher_param.shape:
            teacher_param.data.mul_(alpha).add_(1 - alpha, student_param.data)
        else:
            resized_student = resize_tensor(student_param.data, teacher_param.data.shape)
            teacher_param.data.mul_(alpha).add_(1 - alpha, resized_student)


# Resizing tensor in case of shape mismatch
def resize_tensor(src, target_shape):
    if src.dim() == 2:
        in_features = src.shape[1]
        out_features = target_shape[1]
        linear = nn.Linear(in_features, out_features, bias=False).to(src.device)
        return linear(src)
    elif src.dim() == 4:
        in_channels = src.shape[1]
        out_channels = target_shape[1]
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False).to(src.device)
        return conv(src)
    else:
        raise ValueError(f"Unsupported tensor dimensions: {src.dim()}. Cannot resize.")


def fit(
        student,
        teacher,
        train_dl,
        test_dl,
        weights,
        class_index,
        args,
        device=device,
):
    print()
    student = student.to(device)
    teacher = teacher.to(device)

    # Set optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, student.parameters()), lr=args.lr, weight_decay=1e-5)
    elif args.optimizer == "amsgrad":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr, weight_decay=1e-5, amsgrad=True
        )
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-5
        )

    # Set scheduler
    if args.scheduler == 'StepLR':
        sch = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.epochs // 3, gamma=0.1)
    elif args.scheduler == 'CosineAnnealingLR':
        sch = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    elif args.scheduler == 'OneCycleLR':
        sch = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, epochs=args.epochs,
                                            steps_per_epoch=len(train_dl))
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)

    # Loss Functions
    weights = torch.Tensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    CRA = CRALoss(args).to(device)

    student.train()
    iter_num = 0

    # Initialize lists for tracking metrics
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(args.epochs):
        print("going to train epoch ", epoch)
        train_running_loss = 0
        train_running_acc = 0

        tqdm_train_iterator = tqdm(train_dl, desc='Training')
        for batch_idx, (images, target) in enumerate(tqdm_train_iterator):
            s_images = images.to(device)
            t_images = s_images
            target = target.to(device)
            optimizer.zero_grad()

            s_ftrs, s_logits = student(s_images)
            t_ftrs, t_logits = teacher(t_images)

            loss = criterion(s_logits, target)

            if epoch >= args.n_distill:
                consistency_weight = args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)
                consistency_dist = uncertainity_loss(t_logits, s_logits)
                consistency_loss = consistency_weight * consistency_dist

                ccd_loss, relation_loss = CRA(s_ftrs, t_ftrs, target, class_index, args.nce_p)
                loss += consistency_loss
                loss += args.ccd_weight * ccd_loss
                loss += args.rel_weight * relation_loss

            loss.backward()

            optimizer.step()
            sch.step()
            update_teacher(student, teacher, args.t_decay, iter_num)
            iter_num += 1

            train_running_loss += loss.item()
            train_running_acc += get_accuracy(s_logits.detach(), target)

            tqdm_train_iterator.set_postfix(avg_train_acc=f"{train_running_acc/(batch_idx+1):0.4f}",
                                            avg_train_loss=f"{(train_running_loss/(batch_idx+1)):0.4f}")

        # Save train loss and accuracy
        train_losses.append(train_running_loss / len(train_dl))
        train_accuracies.append(train_running_acc / len(train_dl))

        print('')
        print(f"Epoch: {epoch}")
        # Call the test function after each epoch
        acc, pre, rec, f1, test_loss = test(
            student, test_dl, verbose=True, device=device)

        # Save test loss and accuracy
        test_losses.append(test_loss)
        test_accuracies.append(acc)

    # Save the training history as a .pt file
    history = {
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'test_loss': test_losses,
        'test_acc': test_accuracies
    }
    torch.save(history, 'training_history.pt')

    # Plot and save the loss and accuracy graphs
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(args.epochs), train_losses, label='Train Loss')
    plt.plot(range(args.epochs), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(args.epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(args.epochs), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_progress.png')
    # plt.show()

    # Save the trained model
    torch.save(student.state_dict(), 'trained_student_model.pt')
    student.eval()
    print(" ** Training complete **")
    print(" ** Training complete **")


# Test function to evaluate the model
def test(
        net,
        test_dl,
        verbose=True,
        device="cpu"
):
    net = net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()

    tqdm_test_iterator = tqdm(enumerate(test_dl),
                              desc="[TEST]",
                              total=len(test_dl),
                              ascii=True, leave=True,
                              colour="green", position=0,
                              bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                              mininterval=10)

    test_running_loss = 0
    test_running_acc = 0

    actuals = []
    predictions = []

    for idx, (data, target) in tqdm_test_iterator:
        data = data.to(device)
        target = target.to(device)

        # Get predictions from the model
        _, y_pred = net(data)

        # Compute loss
        loss = criterion(y_pred, target)
        test_running_loss += loss.item()

        # Collect actual and predicted values
        actuals.extend(target.to(device).numpy())  # target already contains class indices
        predictions.extend(y_pred.argmax(dim=1).cpu().numpy())  # argmax over logits to get predicted class indices

        # Calculate accuracy
        test_running_acc += get_accuracy(y_pred.detach(), target)

        tqdm_test_iterator.set_postfix(avg_test_acc=f"{test_running_acc / (idx + 1):0.4f}",
                                       avg_test_loss=f"{(test_running_loss / (idx + 1)):0.4f}")

    print("Test Loss: ", test_running_loss / len(test_dl))
    actuals = np.array(actuals)
    predictions = np.array(predictions)

    # Calculate metrics
    acc = accuracy_score(actuals, predictions)
    pre = precision_score(actuals, predictions, average='macro')
    rec = recall_score(actuals, predictions, average='macro')
    f1 = f1_score(actuals, predictions, average='macro')

    if verbose:
        print("Accuracy: %6f, Precision: %6f, Recall: %6f, F1: %6f \n" %
              (acc, pre, rec, f1))

    return acc, pre, rec, f1, test_running_loss / len(test_dl)
