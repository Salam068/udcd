# CNN test configuration file 
config = {}
config['parentdir'] =  '/content/'                    # Root directory
config['ONN'] = False                                 # Set to 'True' if you are using ONN
config['input_ch'] = 3                                # 1 for gray scale x-rays, and 3 for RGB (3channel) x-rays  
config['batch_size'] = 8                              # Batch size, Change to fit hardware
config['num_folds'] = 1                               # Number of cross-validation folds
config['CI']  = 0.9                                   # Confidence interval (missied cases with probability>=CI will be reported in excel file)
# config['input_mean'] = [0.3680]                     # Dataset mean per channel, b/w [0,1]
# config['input_std'] = [0.1023]                      # Dataset std per channel,  b/w [0,1]
config['input_mean'] = [0.3680,0.3680,0.3680]         # Dataset mean per channel, RGB or RGBA [0,1]
config['input_std'] = [0.1023,0.1023,0.1023]          # Dataset std per channel,  RGB or RGBA [0,1]
config['loss_func'] = 'NLLLoss'                       # 'MSELoss', 'CrossEntropyLoss', etc. (https://pytorch.org/docs/stable/nn.html)
config['Resize_h'] = 224                              # Network input (Image) height
config['Resize_w'] = config['Resize_h']               # Network input (Image) width
# config['load_model'] ='/content/gdrive/MyDrive/EthnicData/Results/mobilenet_v2/mobilenet_v2_fold_1.pt'    # specify full path of pretrained model pt file 
config['load_model'] = False                          # Specify path of pretrained model wieghts or set to False to train from scratch 
config['labeled_Data'] = True                         # Set to true if you have the labeled test set
config['aux_logits'] = False                          # Required for models with auxilliary outputs (e.g., InceptionV3)
config['model_name'] = 'DenseNet201_Tumor_Classification'  # Name of trained model .pt file, same name used in train code
config['N_steps'] = 1000                              # Number of steps for inference
config['fold_to_run'] = [1,1]                         # Define as [] to loop through all folds, or specify start and end folds i.e. [3, 5] or [5, 5]
config['outdir'] = '/content/gdrive/MyDrive/Colab_Notebooks/Research/KiTS21_Kidney_Segmentation/Kidney_Classification/Results'  # The destination directory for saving the pipeline outputs (models, results, plots, etc.)
