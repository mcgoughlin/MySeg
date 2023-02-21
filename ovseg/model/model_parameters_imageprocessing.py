

def get_model_params_2d_imageprocessing(image_folder='images_win_norm',
                                        n_stages=4):
    model_parameters = {}
    batch_size = 12
    trn_dl_params = {'batch_size': batch_size, 'epoch_len': 250, 'store_data_in_ram': True}
    val_dl_params = trn_dl_params.copy()
    val_dl_params['epoch_len'] = 25
    val_dl_params['store_data_in_ram'] = True
    val_dl_params['n_max_volumes'] = 50
    keys = ['image']
    folders = [image_folder]
    data_params = {'n_folds': 4, 'fixed_shuffle': True,
                   'trn_dl_params': trn_dl_params,
                   'val_dl_params': val_dl_params,
                   'keys': keys, 'folders': folders}
    model_parameters['data'] = data_params
    model_parameters['preprocessing'] = {'window': [-1024, 1024]}
    # now finally the training!
    opt_params = {'lr': 10**-4, 'betas': (0.9, 0.999)}
    lr_params = {'beta': 0.9, 'lr_min': 0.5*10**-4}
    training_params = {'num_epochs': 100, 'opt_params': opt_params,
                       'lr_params': lr_params, 'nu_ema_trn': 0.99,
                       'nu_ema_val': 0.7, 'fp32': False,
                       'p_plot_list': [1, 0.5, 0.2], 'opt_name': 'ADAM'}
    model_parameters['training'] = training_params

    network_parameters = {'in_channels': 1, 'out_channels': 1,
                          'kernel_sizes': n_stages*[3],
                          'is_2d': True, 'filters': 8,
                          'filters_max': 384, 'n_pyramid_scales': None}
    model_parameters['network'] = network_parameters
    return model_parameters
