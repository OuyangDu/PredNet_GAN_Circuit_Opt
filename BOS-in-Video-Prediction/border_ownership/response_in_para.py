# convert neural responses to different parameterization (natural or rf)
import numpy as np

def response_in_natural_paramemterization(output_dark, output_light, angle):
    '''
    compute the natural parameterization for merged output. The format of merged output is (c, a, t, width, height, chs). Here natural_para[i, j] returns a 2d array which indicates the c (False is dark, True is light square) value and a (angle) value of the corresponding output_np[i, j]
    c_np and angle_np make up the natural_para. They also indicates possible c and angles that will be useful in constructing the rf_para
    '''
    # obtain parameter table
    c_np = np.array([False, True]) # False means dark. np means natural parameterization
    a_np = angle
    c_grid_np, a_grid_np = np.meshgrid(c_np, a_np, indexing='ij')
    natural_para = np.stack( (c_grid_np, a_grid_np), axis=-1)

    # merge dark and light
    output_np = {}
    for key in output_dark:
        output_np[key] = np.array( ( output_dark[key], output_light[key]) )

    return output_np, natural_para, c_np, a_np

def compute_rf_para(c_np, a_np):
    '''
    the neural response under RF parameterization is (alpha, beta, gamma, t, width, height, chs). Here we use rf_para, whose rf_para[i, j, k] returns a 3d tuple indicating the alpha, beta, and gamma value for responses in output merge[i,j,k] under rf parameterization
    '''
    alpha_rf = np.unique(a_np % 180)
    beta_rf = np.array([False, True])
    gamma_rf = np.array([False, True])
    alpha_grid_rf, beta_grid_rf, gamma_grid_rf = np.meshgrid(alpha_rf, beta_rf, gamma_rf, indexing='ij')
    rf_para = np.stack( (alpha_grid_rf, beta_grid_rf, gamma_grid_rf), axis=-1)
    return rf_para, alpha_rf, beta_rf, gamma_rf

def match_index(i, j, natural_para, rf_para):
    '''
    given i, j in natural parameterization, what is the corresponding index in rf_parameterization
    '''
    alpha = natural_para[i, j, 1] % 180
    beta = natural_para[i, j, 1] < 180
    gamma = np.logical_xor( natural_para[i, j, 0], beta )
    gamma = np.logical_not(gamma)

    alpha_match = rf_para[:, :, :, 0] == alpha
    beta_match = rf_para[:, :, :, 1] == beta
    gamma_match = rf_para[:, :, :, 2] == gamma

    idx_tuple = np.nonzero(alpha_match & beta_match & gamma_match)
    return idx_tuple[0][0], idx_tuple[1][0], idx_tuple[2][0]

def response_in_rf_paramemterization(output_dark, output_light, angle):
    '''
    the neural response under RF parameterization is (alpha, beta, gamma, t, width, height, chs). Here we use rf_para, whose rf_para[i, j, k] returns a 3d tuple indicating the alpha, beta, and gamma value for responses in output merge[i,j,k] under rf parameterization
    '''
    # compute rf para table
    output_np, natural_para, c_np, a_np = response_in_natural_paramemterization(output_dark, output_light, angle)
    rf_para, alpha_rf, beta_rf, gamma_rf = compute_rf_para(c_np, a_np)

    output_rf = {}
    for key in output_np:
        shape = *rf_para.shape[:-1], *output_np[key].shape[2:] # shape of output_np is (angle, color, time, width, height, chs)
        output_rf[key] = np.zeros(shape)
        for i in range( output_np[key].shape[0] ):
            for j in range( output_np[key].shape[1] ):
                idx = match_index(i, j, natural_para, rf_para)
                output_rf[key][idx] = output_np[key][i, j]
    return output_rf, rf_para, alpha_rf, beta_rf, gamma_rf

def keep_central_neuron(output):
    '''
    output can be output_rf or output_np
    '''
    output_center = {}
    for key in output:
        if key == 'X':
            output_center[key] = output[key]
        else:
            center_x, center_y = output[key].shape[-3] // 2, output[key].shape[-2] // 2
            output_center[key] = output[key][..., center_x, center_y, :]
    return output_center

