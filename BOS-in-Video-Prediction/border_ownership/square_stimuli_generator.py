import numpy as np
from PIL import Image, ImageChops
from PIL import ImageDraw

def convert_strip_id_by_beta(strip_id, beta):
    convert_map = {1: 7, 7: 1, 2: 6, 6: 2, 3: 5, 5: 3, 0: 0, 4: 4, None: None}
    if not beta:
        strip_id_beta = [convert_map[si] for si in strip_id]
        return strip_id_beta
    else:
        return strip_id.copy()

def add_strip_square(im, background_color, strip_id, center, length_square, beta, delta=0.):
    '''
    im (pillow image): with a empty square on the right side of the center
    background_color (int): 0 to 255 indicate the background color of the im
    strip_id (list of int): from 0 to 7 indicating which part the square want to strip. Or contains None means no strip.
    center (tuple of 2 ints): center coordinate of the image
    length_square (int): length of the square in im
    beta (bool): the input strip_id assumes beta = True. If false, some strip_id should be converted to the oppsite side.
    delta (float) from 0 to 1. strip square length will be 2 * (0.25 + delta)
    '''
    hb = int( length_square * (0.25 + delta) )
    strip_square = [
        [(center[0] - hb, center[1] - hb), (center[0] + hb, center[1] + hb)], # upper left and bottom right corner of the strip square
        [(center[0] - hb, center[1] - 3 * hb), (center[0] + hb, center[1] - hb)],
        [(center[0] + hb, center[1] - 3 * hb), (center[0] + 3 * hb, center[1] - hb)],
        [(center[0] + 3 * hb, center[1] - 3 * hb), (center[0] + 5 * hb, center[1] - hb)],
        [(center[0] + 3 * hb, center[1] - hb), (center[0] + 5 * hb, center[1] + hb)],
        [(center[0] + 3 * hb, center[1] + hb), (center[0] + 5 * hb, center[1] + 3 * hb)],
        [(center[0] + hb, center[1] + hb), (center[0] + 3 * hb, center[1] + 3 * hb)],
        [(center[0] - hb, center[1] + hb), (center[0] + hb, center[1] + 3 * hb)],
    ]
    strip_id_beta = convert_strip_id_by_beta(strip_id, beta) # the input strip_id assumes beta = True. If false, some strip_id should be converted to the oppsite side.
    for si in strip_id_beta:
        if si is None:
            continue
        else:
            upper_left, bottom_right = strip_square[si][0], strip_square[si][1]
            drawer = ImageDraw.Draw(im)
            drawer.rectangle([tuple(upper_left), tuple(bottom_right)], fill=(background_color))
    return im

class Square_Generator():
    def __init__(self, background_grey, square_grey, length_square=50, width=160, height=128):
        self.background_grey = background_grey
        self.square_grey = square_grey
        self.width = width
        self.height = height
        self.length_square = length_square

    def generate_grey_square_img(self):
        grey_img = Image.new(mode='L', size=(self.width, self.height), color=self.background_grey)

        center = np.array( (self.width//2, self.height//2) )
        self.rotate_center = tuple(center + 0.5)

        upper_left = (center[0], center[1] - self.length_square // 2)
        bottom_right = (center[0] + self.length_square, center[1] + self.length_square // 2)
        drawer = ImageDraw.Draw(grey_img)
        drawer.rectangle( [tuple(upper_left), tuple(bottom_right)], fill=(self.square_grey))
        return grey_img

    def _check_img_pixel_format(self, img):
        '''
        check the image pixel format.
        input:
          img: PIL Image class. Grey image
        '''
        pixel_val = img.getpixel( (0, 0) )
        if type(pixel_val) == int:
            pixel_format = '255'
        else:
            pixel_format = '1'
        return pixel_format

    def _convert_grey_img_to_rgb_video_list(self, img_list, n_frames=5, pixel_format='1'):
        '''
        repeat each image for n_times. We expect the range of pixel either 0 to 1, or 255
        input:
          img_list: a list of PIL image object
          pixel_format: convert to the corresponding range. This can only be '1' or '255'
        output:
          video tensor: np.array. shape is (n_image, n_time, width, height, chs=3)
        '''
        input_pixel_format = self._check_img_pixel_format(img_list[0]) # check the input image pixel 255 or 1

        # convert to np array
        img_np_list = [np.asarray(img) for img in img_list]
        img_np = np.array(img_np_list)

        # convert to target pixel format
        if pixel_format == '1' and input_pixel_format == '255':
            img_np = img_np / 255
        elif pixel_format == '255' and input_pixel_format == '1':
            img_np = int(img_np * 255)

        # make it as a rgb video batch
        video_batch = np.repeat( img_np[:, np.newaxis], n_frames, axis=1 ) # repeat frames
        video_batch = np.repeat( video_batch[..., np.newaxis], 3, axis=-1 ) # repeat rgb
        return video_batch

    def generate_rotated_square_img_list(self, n_direction_line):
        grey_img = self.generate_grey_square_img()

        n_angles = 2 * n_direction_line
        angle_list = np.linspace(0, 360, n_angles, endpoint=False)

        edge_dir_list = angle_list % 180

        img_list = []
        for angle in angle_list:
            img_rotated = grey_img.rotate(angle, fillcolor=self.background_grey, center=self.rotate_center, resample=Image.BICUBIC)
            img_list.append(img_rotated)
        return edge_dir_list, angle_list, img_list

    def generate_strip_square_img_list(self, edge_ori=0, beta=True, strip_id_list=[None]):
        if beta: angle = edge_ori
        else: angle = (edge_ori + 180) % 360
        square_center = np.array( (self.width//2, self.height//2) )
        img_list = []
        for sil in strip_id_list:
            grey_img = self.generate_grey_square_img()
            grey_img_striped = add_strip_square(grey_img, self.background_grey, sil, square_center, self.length_square, beta)
            img_rotated = grey_img_striped.rotate(angle, fillcolor=self.background_grey, center=self.rotate_center, resample=Image.BICUBIC)
            img_list.append(img_rotated)

        return edge_ori, strip_id_list, img_list


    def _idx_within_im(self, idx, length):
        return (idx < length) and (idx > 0)

    def generate_shift_square_img_list(self, edge_ori=0, beta=True, shift_dis_list=[0]):
        '''
        input:
          edge_ori (float): the angle between the edge to the positive y direction. Should be a period variable between 0 to 180
          beta (bool): true means the square is in the upper side, False otherwise. See your note (parameter gamma) for more detail.
          shift_dis_list (list of int): number of pixels shifted along the edge_ori vector (vector whose angle is edge_ori to the positive x direction)
        '''
        if beta: angle = edge_ori
        else: angle = (edge_ori + 180) % 360
        edge_ori_rad = np.deg2rad(edge_ori)

        img_list = []
        grey_img = self.generate_grey_square_img()
        for sd in shift_dis_list:
            img_rotated = grey_img.rotate(angle, fillcolor=self.background_grey, center=self.rotate_center, resample=Image.BICUBIC)
            x_sd, y_sd = int(sd * np.cos(edge_ori_rad)), -int(sd * np.sin(edge_ori_rad)) # y_sd points down, x_sd points to right. Shift direction is fixed for both beta cases

            im_np = np.asarray(img_rotated)
            im_shift = np.ones(im_np.shape) * self.background_grey
            for i in range(im_np.shape[0]):
                for j in range(im_np.shape[1]):
                    idx, idy = i - x_sd, j - y_sd
                    if self._idx_within_im(idx, self.height) and self._idx_within_im(idy, self.width): im_shift[idx, idy] = im_np[i, j]
            img_shifted = Image.fromarray(np.uint8(im_shift))

            img_list.append(img_shifted)

        return edge_ori, shift_dis_list, img_list

    def generate_size_square_img_list(self, edge_ori=0, beta=True, size_list=[0]):
        '''
        input:
          edge_ori (float): the angle between the edge to the positive y direction. Should be a period variable between 0 to 180
          beta (bool): true means the square is in the upper side, False otherwise. See your note (parameter gamma) for more detail.
          size_list (list of int): square length
        '''
        if beta: angle = edge_ori
        else: angle = (edge_ori + 180) % 360
        edge_ori_rad = np.deg2rad(edge_ori)

        img_list = []
        length_square_temp = self.length_square # keep it temporarily

        for s in size_list:
            self.length_square = s
            grey_img = self.generate_grey_square_img()
            img_rotated = grey_img.rotate(angle, fillcolor=self.background_grey, center=self.rotate_center, resample=Image.BICUBIC)

            img_list.append(img_rotated)

        self.length_square = length_square_temp # change it back to default

        return edge_ori, size_list, img_list

    def generate_rotated_square_video_list(self, n_direction_line, n_frames=5, pixel_format='1'):
        edge_dir_list, angle_list, img_list = self.generate_rotated_square_img_list(n_direction_line)
        video_batch = self._convert_grey_img_to_rgb_video_list(img_list, n_frames, pixel_format)
        return edge_dir_list, angle_list, video_batch

    def generate_strip_square_video_list(self, edge_ori, beta=True, strip_id_list=[None], n_frames=5, pixel_format='1'):
        '''
        stripe_id_list (list of list): each sublist contains ints from 0 to 7. None means no striping, 0 to 7 means stripe corresponding edge. For example, strip_id_list = [[1, 2, 3], [1]] means generate a square without 1, 2, 3 edges and another square without edge 1.
        '''
        edge_ori, strip_id_list, img_list = self.generate_strip_square_img_list(edge_ori=edge_ori, beta=beta, strip_id_list=strip_id_list)
        video_batch = self._convert_grey_img_to_rgb_video_list(img_list, n_frames, pixel_format)
        return edge_ori, strip_id_list, video_batch

    def generate_shift_square_video_list(self, edge_ori=0, beta=True, shift_dis_list=[0], n_frames=5, pixel_format='1'):
        '''
        input:
          n_frames (int): number of frames you wanna create
          pixel_format (str, '1' or '255'): image grey value scale
        '''
        edge_ori, shift_dis_list, img_list = self.generate_shift_square_img_list(edge_ori=edge_ori, beta=beta, shift_dis_list=shift_dis_list)
        video_batch = self._convert_grey_img_to_rgb_video_list(img_list, n_frames, pixel_format)
        return edge_ori, shift_dis_list, video_batch

    def generate_size_square_video_list(self, edge_ori=0, beta=True, size_list=[0], n_frames=5, pixel_format='1'):
        '''
        this is similar to shift_square. They should be merged into one function, including generate_rotated_square_video_list. However, generate_rotate... has a bit different interface so I keep these functions seperated.
        input:
          n_frames (int): number of frames you wanna create
          pixel_format (str, '1' or '255'): image grey value scale
        '''
        edge_ori, size_list, img_list = self.generate_size_square_img_list(edge_ori=edge_ori, beta=beta, size_list=size_list)
        video_batch = self._convert_grey_img_to_rgb_video_list(img_list, n_frames, pixel_format)
        return edge_ori, size_list, video_batch


