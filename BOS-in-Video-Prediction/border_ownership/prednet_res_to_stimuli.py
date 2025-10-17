import border_ownership.response_in_para as rip
from border_ownership.square_stimuli_generator import Square_Generator
from border_ownership.agent import Agent
from kitti_settings import *
import matplotlib.pyplot as plt

def get_neural_response_to_para(edge_ori, background_grey, square_grey, beta, para_list, n_frames=20, pixel_format='1', output_mode='R2', prednet_json_file=None, prednet_weights_file=None, mode='shift', verbose=False):
    '''
    mode (str): shift or size
    verbose (bool): output the video as well
    '''
    if prednet_json_file is None:
        prednet_json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
    if prednet_weights_file is None:
        prednet_weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/' + 'prednet_kitti_weights.hdf5')

    rsg = Square_Generator(background_grey=background_grey, square_grey=square_grey)

    if mode == 'shift':
        edge_ori, para_list, video_batch = rsg.generate_shift_square_video_list(edge_ori=edge_ori, beta=beta, shift_dis_list=para_list, n_frames=n_frames, pixel_format=pixel_format)
    elif mode == 'size':
        edge_ori, para_list, video_batch = rsg.generate_size_square_video_list(edge_ori=edge_ori, beta=beta, size_list=para_list, n_frames=n_frames, pixel_format=pixel_format)
    elif mode == 'strip':
        edge_ori, para_list, video_batch = rsg.generate_strip_square_video_list(edge_ori=edge_ori, beta=beta, strip_id_list=para_list, n_frames=n_frames, pixel_format=pixel_format)

    #for video in video_batch:
    #    plt.figure()
    #    plt.imshow(video[0])
    #    plt.show()
    #exit()

    sub = Agent()
    sub.read_from_json(prednet_json_file, prednet_weights_file)
    output_square = sub.output(video_batch, output_mode=output_mode, batch_size=32, is_upscaled=False) # is_upscaled false assumes that input pixel ranges from 0 to 1
    output_square = {output_mode: output_square}
    output_square_center = rip.keep_central_neuron(output_square)
    neural_res = output_square_center[output_mode]
    if verbose:
        return neural_res, edge_ori, para_list, video_batch
    else:
        return neural_res, edge_ori, para_list
