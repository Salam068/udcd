##### DO NOT EDIT THESE LINES #####
config = {}
###################################


#### START EDITING FROM HERE ######
config['parentdir'] = '/media/salam/Salam/MSc/DMSA3/'                       # main directory
config['ImageNet'] = True  ##                           # set to 'True' to use ImageNet weights or set to 'False' to train from scratch
config['q_order'] = 3                                   # qth order Maclaurin approximation, common values: {1,3,5,7,9}. q=1 is equivalent to conventional CNN
config['model_mode'] = 'import_Torch'                  # 'custom_CNN' | 'custom_ONN' | 'import_Torch' | 'import_TIMM'
config['ONN'] = False                                   # set to 'True' if you are using ONN
config['input_ch'] = 3                                  # 1 for gray scale x-rays, and 3 for RGB (3channel) x-rays
config['batch_size']  = 4                              # batch size, Change to fit hardware
config['train_dual'] = False                            # set to 'True' to train dual model in the form or densenet201*densenet201
# config['input_mean'] = [0.0256]                       # Dataset mean per channel, b/w [0,1]
# config['input_std'] = [0.4472]                        # Dataset std per channel,  b/w [0,1]
config['input_mean'] = [0.6493, 0.6307, 0.6029]                      # Dataset mean per channel, b/w [0,1]
config['input_std'] = [0.3096, 0.3068, 0.3255]                       # Dataset std per channel,  b/w [0,1]
config['loss_func'] = 'CrossEntropyLoss'                   # 'MSELoss', 'CrossEntropyLoss', etc. (https://pytorch.org/docs/stable/nn.html)
config['optim_fc'] = 'Adam'                             # 'Adam' or 'SGD'
config['optim_scheduler'] = 'ReduceLROnPlateau'        # 'ReduceLROnPlateau', etc. (https://pytorch.org/docs/stable/optim.html)
config['final_activation_func'] = 'Softmax'  # 'Sigmoid', 'Softmax', etc. (https://pytorch.org/docs/stable/nn.html)
config['lr'] = 0.0001                                    # learning rate
config['stop_criteria'] = 'loss'                        # Stopping criteria: 'loss' or 'accuracy'
config['n_epochs']  = 60                               # number of training epochs
config['epochs_patience'] = 30                          # if val loss did not decrease for a number of epochs then decrease learning rate by a factor of lr_factor
config['lr_factor'] = 0.1
config['max_epochs_stop'] = 30                          # maximum number of epochs with no improvement in validation loss for early stopping
config['num_folds']  = 5                                # number of cross validation folds
config['Resize_h'] = 224                                # network input size
config['Resize_w'] = config['Resize_h']
#config['load_model'] = config['parentdir'] + 'drive/MyDrive/ppg/2D pytorch/Results/densenet121_training_100/checkpoint_1.pt' # specify path of pretrained model wieghts or set to False to train from scratch
config['aux_logits'] = False                   # Required for models with auxilliary outputs (e.g., InceptionV3)
config['load_weights'] = False                                                       # specify path of pretrained model wieghts or set to False to train from scratch
config['model_to_load'] = 'densenet201'
config['model_name'] = 'densenet201_training'                                # choose a unique name for result folder
config['RHFlip'] = 0
config['RotaionDegree'] = 0
config['P_padding'] = 0
config['P_fill'] = 0
config['P_padding_mode'] = 'constant'                                               # chosse one of the following models:
#  'squeezenet1_0', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet50', 'resnet101', 'resnet152', 'inception_v3', 'inceptionresnetv2'
#  'xception', 'chexnet', 'nasnetalarge', 'pnasnet5large', 'densenet121', 'densenet161', 'densenet201', 'shufflenet' , 'googlenet', 'mobilenet_v2', 'nasnetamobile', 'alexnet'
config['encoder'] = False  # set to 'True' if you retrain Seg. model encoder as a classifer
config['fold_to_run'] = [2,2] # define as [3,5] to loop through all folds, or specify start and end folds i.e. [3, 5] or [5, 5]
##################

##################
config['drive_folder'] = '/media/salam/Salam/MSc/DMSA3' +'/'
config['Results_path'] = config['drive_folder'] + 'Results'                             # main results file
config['outdir'] = config['Results_path'] +'/'+ config['model_name']              # save path
##################
