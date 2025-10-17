#####################################################
#This file creates videos of 2 pictures and looks at the 
#  averages neuron resposes of prednet.
# Here by average responses we mean the average of
# only the canidate neurons with the reciptive field of radius less than 10 pixels
#####################################################

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
from PIL import Image, ImageDraw
from drawing_square_image import *
from drawing_pacman import *
with open('center_neuron_info_radius10.pkl', 'rb') as file:
    data = pkl.load(file)

# Get the directory of the current script
current_dir = os.path.dirname(__file__)
# Get the common parent directory (one level up)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add the parent directory to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from border_ownership.full_square_response import square_generator_bo

# main



if __name__ == "__main__":
    ################################################ 
    # creating different images
    ################################################
    width_=48
    r_=12
    ori=3
    out=['E0','E1','E2','E3']
    ori_d=0

    light_grey_value=255 * 2 // 3
    dark_grey_value=255//3
    dark_grey=(dark_grey_value,dark_grey_value,dark_grey_value)
    light_grey=(light_grey_value,light_grey_value,light_grey_value)
    ###################################################

    img_cir=circle_sqr(r=r_,width=width_,orientation=ori,circle_color=dark_grey,background_color=light_grey)
    img_ksqr=border_kaniza_sqr(r=r_,width=width_,orientation=ori,pacman_color=dark_grey,background_color=light_grey)
    img_n_ksqr=non_kaniza_sqr(r=r_,width=width_,orientation=ori,pacman_color=dark_grey,background_color=light_grey)
    img_ksqr_line=border_kaniza_sqr_with_square(r=r_,width=width_,orientation=ori,pacman_color=dark_grey,background_color=light_grey,square_line_color=dark_grey)
    img_Lsqr=line_border_sqr(width=width_,orientation=ori,background_color=light_grey,square_line_color=dark_grey)
    img_sqr_1=square_generator_bo(beta=False,orientation=ori_d,gamma=True,size=width_)
    img_sqr_2=square_generator_bo(beta=True,orientation=0,gamma=False,size=width_)
    
    """
    # draw circle of radius 10 to the center image
    # make sure there is no overlap between inducers and the recptive field 
    draw = ImageDraw.Draw(img_cir)
    left_up_point = (80 - 10, 64 - 10)
    right_down_point = (80 + 10, 64 + 10)
    draw.ellipse([left_up_point, right_down_point], fill=dark_grey, outline="red")

    # show each image to make sure for testing purposes
    img_cir.show()
    img_ksqr.show()
    img_n_ksqr.show()
    img_ksqr_line.show()
    img_Lsqr.show()
    img_sqr_1.show()
    img_sqr_2.show()
    """
    img_cir.show()
    img_ksqr.show()
    img_n_ksqr.show()
    img_ksqr_line.show()
    img_Lsqr.show()
    img_sqr_1.show()

   
    #################################################################
    # Generating videos
    ################################################################


    # generate 2 picture images that 5 frame for each image

    # circle to circle
    video_cir= create_static_video_from_two_images(img_cir,img_cir)
    #print(cir_video.shape)

    #circle to kanisa sqr
    video_ksqr= create_static_video_from_two_images(img_cir,img_ksqr)
    
    """
    Testing  if video frames is indexed as expected
    plt.figure()
    plt.imshow(video_ksqr[0, 4])
    plt.show()
    plt.figure()
    plt.imshow(video_ksqr[0, 5])
    plt.show()
    """


    #circle to non kanisa sqr (pac-man rotated outward)
    video_non_ksqr= create_static_video_from_two_images(img_cir,img_n_ksqr)


    # circle to kanisa sqr with real contour
    video_ksqr_line= create_static_video_from_two_images(img_cir,img_ksqr_line)
    
    
    # circle to border of square
    video_Lsqr=create_static_video_from_two_images(img_cir,img_Lsqr)

    #circle to filled square
    video_sqr=create_static_video_from_two_images(img_cir,img_sqr_1)


    ################################################################
    # Generating Prednet Response
    ###############################################################    
    #### PredNet demo. You need to put border_ownership.agent.py file in the correct path so that import works.
    from border_ownership.agent import Agent

    ## turn off CUDA if your GPU is not available or too new > 3060
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    WEIGHTS_DIR = '../model_data_keras2/'
    json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
    weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')

    output_mode = ['prediction', 'E0','E1','E2','E3'] # a list of output modes. Can be 'prediction', 'E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3', 'A0', 'A1', 'A2', 'A3', 'Ahat0', 'Ahat1', 'Ahat2', 'Ahat3'. If prediction, output prediction; if others, output the units responses
    agent = Agent()
    agent.read_from_json(json_file, weights_file)

    ###############################################################################
    # Getting Outputs 
    # read prednet resposes to the output generated by the videos created above
    #
    ###################
    output_cir = agent.output_multiple(video_cir, output_mode=output_mode, is_upscaled=False) # the output is a dictionary with keys as output_mode. is_upscaled=False means the input frame ranges from 0 to 1; is_upscaled=True means the input frame ranges from 0 to 255.
    output_ksqr = agent.output_multiple(video_ksqr, output_mode=output_mode, is_upscaled=False)
    output_non_ksqr= agent.output_multiple(video_non_ksqr, output_mode=output_mode, is_upscaled=False)
    output_ksqr_line= agent.output_multiple(video_ksqr_line, output_mode=output_mode, is_upscaled=False)
    output_Lsqr= agent.output_multiple(video_Lsqr, output_mode=output_mode, is_upscaled=False)
    output_sqr= agent.output_multiple(video_sqr, output_mode=output_mode, is_upscaled=False)
    #################################################################################

    #print("output_cir E1 shape:")
    #print(output_cir['E1'].shape)

    #################################################################################
    # read candidate iD for reciptive within R<10
    ###########################
    for mod in out:
        sums=np.zeros(10)
        for i in range(10):
            for id in data['bo_info'][mod]['neuron_id']:
                sums[i]+=output_cir[mod][0,i,id[0],id[1],id[2]]

        sums2=np.zeros(10)
        for i in range(10):
            for id in data['bo_info'][mod]['neuron_id']:
                sums2[i]+=output_ksqr[mod][0,i,id[0],id[1],id[2]]

    
        sums3=np.zeros(10)
        for i in range(10):
            for id in data['bo_info'][mod]['neuron_id']:
                sums3[i]+=output_non_ksqr[mod][0,i,id[0],id[1],id[2]]
    
        sums4=np.zeros(10)
        for i in range(10):
            for id in data['bo_info'][mod]['neuron_id']:
                sums4[i]+=output_ksqr_line[mod][0,i,id[0],id[1],id[2]]
    
        sums5=np.zeros(10)
        for i in range(10):
            for id in data['bo_info'][mod]['neuron_id']:
                sums5[i]+=output_Lsqr[mod][0,i,id[0],id[1],id[2]]
    
        sums6=np.zeros(10)
        for i in range(10):
            for id in data['bo_info'][mod]['neuron_id']:
                sums6[i]+=output_sqr[mod][0,i,id[0],id[1],id[2]]

        plt.figure()
        plt.plot(sums,label="No change", marker='o')
        plt.plot(sums2,label="kanisa square", marker='o')
        plt.plot(sums3,label="non kanisa square", marker='o')
        plt.plot(sums4,label="k square and line", marker='o')
        plt.plot(sums5,label="line square", marker='o')
        plt.plot(sums6,label="block square", marker='o')
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("PredNet Response"+mod)
        plt.legend()
        plt.show()
    
