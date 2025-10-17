import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt
from drawing_square_image import *
from drawing_pacman import *
from border_ownership.agent import Agent

def normalize_neuron_responses(bo_info, outputs_dict):
    """
    Normalize neuron responses across all frames and videos.

    Output Format:
    normalized_responses is a dictionary where:
        - each key is the name of the video (e.g., 'norm_rsp_cir', 'norm_rsp_non_ksqr', etc.)
        - each value is a list of arrays, one per neuron
            - each array contains normalized responses over time (frames) for that neuron

    Example:
    {
        'norm_rsp_cir': [neuron0_responses, neuron1_responses, ...],
        'norm_rsp_non_ksqr': [...],
        ...
    }
    where each neuron response is a 1D array of shape (time_steps,)
    """
    outputs = [
        outputs_dict.get('output_cir'),
        outputs_dict.get('output_ksqr'),
        outputs_dict.get('output_non_ksqr'),
        outputs_dict.get('output_ksqr_line'),
        outputs_dict.get('output_Lsqr'),
        outputs_dict.get('output_sqr'),
        outputs_dict.get('output_sqr_fliped'),
    ]

    max_response_per_neuron = {}
    for neuron_idx, neuron_id in enumerate(bo_info['E2']['neuron_id']):
        channel, x, y = neuron_id
        max_val = -np.inf
        for out in outputs:
            if out and 'E2' in out:
                o = out['E2']
                neuron_responses = o[0, :, channel, x, y]
                max_in_output = np.max(neuron_responses)
                if max_in_output > max_val:
                    max_val = max_in_output
        max_response_per_neuron[neuron_idx] = max_val

    def normalize_output(output, mod='E2'):
        norm_rsp = []
        for neuron_idx, neuron_id in enumerate(bo_info['E2']['neuron_id']):
            channel, x, y = neuron_id
            neuron_response = output[mod][0, :, channel, x, y]
            normalized = neuron_response / max_response_per_neuron[neuron_idx] if max_response_per_neuron[neuron_idx] else neuron_response
            norm_rsp.append(normalized)
        return norm_rsp

    normalized_responses = {}
    normalized_responses['norm_rsp_cir'] = normalize_output(outputs_dict.get('output_cir'))
    normalized_responses['norm_rsp_ksqr'] = normalize_output(outputs_dict.get('output_ksqr'))
    normalized_responses['norm_rsp_non_ksqr'] = normalize_output(outputs_dict.get('output_non_ksqr'))
    normalized_responses['norm_rsp_ksqr_line'] = normalize_output(outputs_dict.get('output_ksqr_line'))
    normalized_responses['norm_rsp_Lsqr'] = normalize_output(outputs_dict.get('output_Lsqr'))
    normalized_responses['norm_rsp_sqr'] = normalize_output(outputs_dict.get('output_sqr'))
    normalized_responses['norm_rsp_sqr_fliped'] = normalize_output(outputs_dict.get('output_sqr_fliped'))
    return normalized_responses

if __name__ == "__main__":
    with open('center_neuron_info_radius10.pkl', 'rb') as file:
        data = pkl.load(file)

    width_ = 48
    r_ = 12
    light_grey_value = 255 * 2 // 3
    dark_grey_value = 255 // 3
    dark_grey = (dark_grey_value, dark_grey_value, dark_grey_value)
    light_grey = (light_grey_value, light_grey_value, light_grey_value)

    direction_labels = ['up', 'right', 'down', 'left']

    for ori in range(4):
        print(f"Processing orientation {ori}...")
        ori_d = ori * 90

        img_cir = circle_sqr(r=r_, width=width_, orientation=ori, circle_color=light_grey, background_color=dark_grey)
        img_ksqr = border_kaniza_sqr(r=r_, width=width_, orientation=ori, pacman_color=light_grey, background_color=dark_grey)
        img_n_ksqr = non_kaniza_sqr(r=r_, width=width_, orientation=ori, pacman_color=light_grey, background_color=dark_grey)
        img_ksqr_line = border_kaniza_sqr_with_square(r=r_, width=width_, orientation=ori, pacman_color=light_grey, background_color=dark_grey, square_line_color=light_grey)
        img_Lsqr = line_border_sqr(width=width_, orientation=ori, background_color=dark_grey, square_line_color=light_grey)
        img_sqr_1 = square_generator(width=width_, orientation=ori, background_color=dark_grey, square_fill_color=light_grey)
        img_sqr_2 = square_generator(width=width_, orientation=ori, background_color=light_grey, square_fill_color=dark_grey)

        video_cir = create_static_video_from_two_images(img_cir, img_cir)
        video_ksqr = create_static_video_from_two_images(img_cir, img_ksqr)
        video_non_ksqr = create_static_video_from_two_images(img_cir, img_n_ksqr)
        video_ksqr_line = create_static_video_from_two_images(img_cir, img_ksqr_line)
        video_Lsqr = create_static_video_from_two_images(img_cir, img_Lsqr)
        video_sqr = create_static_video_from_two_images(img_cir, img_sqr_1)
        video_sqr_fliped = create_static_video_from_two_images(img_cir, img_sqr_2)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        WEIGHTS_DIR = '../model_data_keras2/'
        json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
        weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')

        output_mode_ = ['prediction', 'E2']
        agent = Agent()
        agent.read_from_json(json_file, weights_file)

        output_cir = agent.output_multiple(video_cir, output_mode=output_mode_, is_upscaled=False)
        output_ksqr = agent.output_multiple(video_ksqr, output_mode=output_mode_, is_upscaled=False)
        output_non_ksqr = agent.output_multiple(video_non_ksqr, output_mode=output_mode_, is_upscaled=False)
        output_ksqr_line = agent.output_multiple(video_ksqr_line, output_mode=output_mode_, is_upscaled=False)
        output_Lsqr = agent.output_multiple(video_Lsqr, output_mode=output_mode_, is_upscaled=False)
        output_sqr = agent.output_multiple(video_sqr, output_mode=output_mode_, is_upscaled=False)
        output_sqr_fliped= agent.output_multiple(video_sqr_fliped, output_mode=output_mode_, is_upscaled=False)

        outputs_dict = {
            'output_cir': output_cir,
            'output_ksqr': output_ksqr,
            'output_non_ksqr': output_non_ksqr,
            'output_ksqr_line': output_ksqr_line,
            'output_Lsqr': output_Lsqr,
            'output_sqr': output_sqr,
            'output_sqr_fliped': output_sqr_fliped,

        }

        normalized_responses = normalize_neuron_responses(data['bo_info'], outputs_dict)

        legend_labels = {
            'norm_rsp_cir': 'no change',
            'norm_rsp_ksqr': 'kanisa square',
            'norm_rsp_non_ksqr': 'non-kanisa',
            'norm_rsp_ksqr_line': 'kanisa square with line',
            'norm_rsp_Lsqr': 'line square',
            'norm_rsp_sqr': 'filled square',
            'norm_rsp_sqr_fliped': 'filled square fliped'
        }

        plt.figure(figsize=(10, 6))
        for video_key, neuron_responses in normalized_responses.items():
            summed_response = np.sum(np.array(neuron_responses), axis=0)
            label = legend_labels.get(video_key, video_key)
            plt.plot(np.arange(len(summed_response)), summed_response, label=label)
        plt.xlabel("Time Frames")
        plt.ylabel("Summed Normalized Response")
        plt.title(f"Summed Normalized Responses per Video (Direction: {direction_labels[ori]})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"summed_normalized_response_{direction_labels[ori]}.png")
        plt.close()
