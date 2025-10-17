import numpy as np

def ori_tuning(output_module):
    '''
    input:
      the output_module should be in the rf_parametereization (alpha, beta, gamma, t, chs), where chs also means neurons
    output:
      tuning: (alpha, chs)
    '''
    tuning = np.mean(output_module, axis=(1, 2, 3)) - np.mean(output_module, axis=(0, 1, 2, 3))[np.newaxis] # substracted by baseline

    ## This is for quick comparison if you wanna compute directional tuning of a white square
    #tuning = np.mean(output_module[:, 1, 1], axis=(1)) - np.mean(output_module[:, 1, 1], axis=(0, 1))[np.newaxis] # substracted by baseline
    #tuning = np.mean(output_module[:, 0, 0], axis=(1)) - np.mean(output_module[:, 0, 0], axis=(0, 1))[np.newaxis] # substracted by baseline
    #print(tuning_1 == tuning_2)
    return tuning
