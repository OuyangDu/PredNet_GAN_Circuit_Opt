"""
Draw kaniza squares

    add_pacman() adds pacman shape to an existing figure

    add_full_circles() adds circles to an existing image

    def add_cross() adds crosses to an existing image
    
    border_kaniza_sqr() draws kaniza squares with pacman

    kaniza_cross_sqr() draws kaniza square with crosses

    border_kaniza_rec() draws kaniza rectangle

    border_kaniza_sqr_with_square() draws kaniza squares with a line square connecting the pacmans

    line_border_sqr(): draw the borders of a square

    non_kaniza_sqr():  have packman facing outward

    non_kaniza_rec():  kanisza retangle have packman facing outward

    circle_sqr(): draw 4 circles so the circles form the corners of a square
    
"""
import numpy as np
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt



light_grey_value=255 * 2 // 3
light_grey=(light_grey_value,light_grey_value,light_grey_value)

dark_grey_value=255//3
dark_grey=(dark_grey_value,dark_grey_value,dark_grey_value)

image_size=(160,128)

def add_pacman(image, center, radius=13, mouth_angle=90, orientation=0, pacman_color=dark_grey):
    """
    Draws a Pac-Man–like figure onto an existing PIL image.
    
    Parameters:
        image (PIL.Image): The image to modify.
        center (tuple): The (x, y) coordinates for the center of Pac-Man.
        radius (float): The radius of Pac-Man.
        mouth_angle (float): The total opening angle (in degrees) for the mouth.
        orientation (int): 0,1,2,3;  
            0:(right_down facing);  1:(right_up facing); 2:(left_up facing); 3:(left_down facing)
        pacman_color (tuple): The fill color for Pac-Man (default is yellow).
        
    Returns:
        PIL.Image: The image with the Pac-Man figure drawn on it.
    """
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    
    # re-asign orentation so the angle is in degrees: 0 is right facing
    orientation= 45+ orientation*90


    # Calculate the arc angles (in degrees) for the body (a circle minus a wedge for the mouth)
    start_angle = orientation + mouth_angle / 2
    end_angle = orientation - mouth_angle / 2 + 360  # ensure proper ordering
    
    # Generate points along the arc using NumPy
    theta = np.linspace(np.deg2rad(start_angle), np.deg2rad(end_angle), num=100)
    arc_points = [(center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)) for t in theta]
    
    # Create the polygon points: start at the center, then follow the arc
    polygon_points = [center] + arc_points
    
    # Draw the filled Pac-Man polygon
    draw.polygon(polygon_points, fill=pacman_color)
    
    return image

def add_cross(image, center, size=26, width=3, color=dark_grey):
    """
    Draws a centered cross (“+”) onto a PIL image.
    
    Parameters:
        image (PIL.Image): The image to modify.
        center (tuple): (x, y) center of the cross.
        size (int): Total length of each arm of the cross (so each half-arm is size/2).
        width (int): Thickness of the lines.
        color (tuple): RGB color of the cross.
    Returns:
        PIL.Image: The image with the cross drawn.
    """
    draw = ImageDraw.Draw(image)
    x, y = center
    half = size / 2

    # horizontal line
    draw.line(
        [(x - half, y), (x + half, y)],
        fill=color,
        width=width
    )
    # vertical line
    draw.line(
        [(x, y - half), (x, y + half)],
        fill=color,
        width=width
    )
    return image

def add_full_circle(image, center, radius=13, circle_color=dark_grey):
    """
    Draws a full circle onto an existing PIL image.
    
    Parameters:
        image (PIL.Image): The image to modify.
        center (tuple): The (x, y) coordinates for the center of the circle.
        radius (float): The radius of the circle.
        circle_color (tuple): The fill color for the circle (default is dark grey).
        
    Returns:
        PIL.Image: The image with the full circle drawn on it.
    """
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    
    # Calculate the bounding box for the circle.
    # The bounding box is defined by the top-left and bottom-right points.
    left = center[0] - radius
    top = center[1] - radius
    right = center[0] + radius
    bottom = center[1] + radius
    
    # Draw the circle (ellipse with a square bounding box)
    draw.ellipse([left, top, right, bottom], fill=circle_color)
    
    return image




def border_kaniza_sqr (image_size=(160,128), orientation=0, width=52, pacman_color=light_grey, background_color=dark_grey,r=13):
    """
    make an image with kaniza square, with one of the imaginary edges crossing the center
    of the image

    the kaniza square can either be up, down, right, or, left. 
    to avoid diagnol (step like diagonal edges) 

    Pramenters:
        image_size(width,height): the size of the image
        orientation(int): 0 1 2 3
            0(up) 1(right) 2(down) 3(left)
        pacman_color: RGB colors (,,) of kaniza inducers
        background_color: background color of the entire image
        r: radius of the pacman inducers


    """

    # create new image of image_size
    img = Image.new("RGB", image_size, background_color)
    mouth_angle=90
    
    #up orientation
    if orientation==0:
        bot_y=image_size[1]//2
        top_y=image_size[1]//2-width
        left_x=image_size[0]//2 -width//2
        right_x=image_size[0]//2 +width//2


        #top_left pacman
        img = add_pacman(img, (left_x,top_y), r, mouth_angle, 0,pacman_color)
        #top_right pacman
        img = add_pacman(img, (right_x,top_y), r, mouth_angle, 1,pacman_color)
        #bottom_left pacman
        img = add_pacman(img, (left_x,bot_y), r, mouth_angle, 3,pacman_color)
        #bottom_right pacman
        img = add_pacman(img, (right_x,bot_y), r, mouth_angle, 2, pacman_color)
   
    #right orentation
    elif orientation==1:
        bot_y=image_size[1]//2+width//2
        top_y=image_size[1]//2-width//2
        left_x=image_size[0]//2 
        right_x=image_size[0]//2 +width
        #top_left pacman
        img = add_pacman(img, (left_x,top_y), r, mouth_angle, 0,pacman_color)
        #top_right pacman
        img = add_pacman(img, (right_x,top_y), r, mouth_angle, 1,pacman_color)
        #bottom_left pacman
        img = add_pacman(img, (left_x,bot_y), r, mouth_angle, 3,pacman_color)
        #bottom_right pacman
        img = add_pacman(img, (right_x,bot_y), r, mouth_angle, 2, pacman_color)

    #down orientation
    elif orientation==2:
        bot_y=image_size[1]//2+width
        top_y=image_size[1]//2
        left_x=image_size[0]//2 -width//2
        right_x=image_size[0]//2 +width//2


        #top_left pacman
        img = add_pacman(img, (left_x,top_y), r, mouth_angle, 0,pacman_color)
        #top_right pacman
        img = add_pacman(img, (right_x,top_y), r, mouth_angle, 1,pacman_color)
        #bottom_left pacman
        img = add_pacman(img, (left_x,bot_y), r, mouth_angle, 3,pacman_color)
        #bottom_right pacman
        img = add_pacman(img, (right_x,bot_y), r, mouth_angle, 2, pacman_color)
    
    #left orientation
    elif orientation==3:
        bot_y=image_size[1]//2+width//2
        top_y=image_size[1]//2-width//2
        left_x=image_size[0]//2 - width
        right_x=image_size[0]//2
        #top_left pacman
        img = add_pacman(img, (left_x,top_y), r, mouth_angle, 0,pacman_color)
        #top_right pacman
        img = add_pacman(img, (right_x,top_y), r, mouth_angle, 1,pacman_color)
        #bottom_left pacman
        img = add_pacman(img, (left_x,bot_y), r, mouth_angle, 3,pacman_color)
        #bottom_right pacman
        img = add_pacman(img, (right_x,bot_y), r, mouth_angle, 2, pacman_color)
    return img

def kaniza_cross_sqr(
    image_size=(160, 128),
    orientation=0,
    width=52,
    cross_color=light_grey,
    background_color=dark_grey,
    r=13,
    thickness_divisor=4):
    """
    Make a Kanizsa-square image using inward-facing crosses.
    Parameters:
        image_size: (width, height) of the image
        orientation: 0=up, 1=right, 2=down, 3=left
        width: side length of the square
        cross_color: RGB tuple for the crosses
        background_color: RGB tuple for the background
        r: controls cross size and inward shift
        thickness_divisor: the number r is divided by to compute stroke thickness
    """
    img = Image.new("RGB", image_size, background_color)

    # cross dimensions
    cross_size = 2 * r
    cross_width = max(1, r // thickness_divisor)
    o = cross_width // 2  # inward shift distance

    cx = image_size[0] // 2
    cy = image_size[1] // 2
    half_w = width // 2

    # compute raw square corner centers
    if orientation == 0:      # up
        y1, y2 = cy - width, cy
        x1, x2 = cx - half_w, cx + half_w
    elif orientation == 1:    # right
        y1, y2 = cy - half_w, cy + half_w
        x1, x2 = cx, cx + width
    elif orientation == 2:    # down
        y1, y2 = cy, cy + width
        x1, x2 = cx - half_w, cx + half_w
    else:                     # left
        y1, y2 = cy - half_w, cy + half_w
        x1, x2 = cx - width, cx

    # shifts to move each cross outward
    shifts = [
        (-o, -o),  # top-left
        (+o, -o),  # top-right
        (-o, +o),  # bottom-left
        (+o, +o),  # bottom-right
    ]
    centers = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

    # draw the four outward-shifted crosses
    for (raw_x, raw_y), (dx, dy) in zip(centers, shifts):
        img = add_cross(
            img,
            (raw_x + dx, raw_y + dy),
            size=cross_size,
            width=cross_width,
            color=cross_color
        )

    return img


def border_kaniza_rec (image_size=(160,128), orientation=0, width=52, height=52 ,pacman_color=light_grey, background_color=dark_grey,r=13):
    """
    make an image with kaniza rectangle, with one of the imaginary edges crossing the center
    of the image

    the kaniza square can either be up, down, right, or, left. 
    to avoid diagnol (step like diagonal edges) 

    Pramenters:
        image_size(width,height): the size of the image
        orientation(int): 0 1 2 3
            0(up) 1(right) 2(down) 3(left)
        pacman_color: RGB colors (,,) of kaniza inducers
        background_color: background color of the entire image
        r: radius of the pacman inducers


    """

    # create new image of image_size
    img = Image.new("RGB", image_size, background_color)
    mouth_angle=90
    
    #up orientation
    if orientation==0:
        bot_y=image_size[1]//2
        top_y=image_size[1]//2-height
        left_x=image_size[0]//2 -width//2
        right_x=image_size[0]//2 +width//2


        #top_left pacman
        img = add_pacman(img, (left_x,top_y), r, mouth_angle, 0,pacman_color)
        #top_right pacman
        img = add_pacman(img, (right_x,top_y), r, mouth_angle, 1,pacman_color)
        #bottom_left pacman
        img = add_pacman(img, (left_x,bot_y), r, mouth_angle, 3,pacman_color)
        #bottom_right pacman
        img = add_pacman(img, (right_x,bot_y), r, mouth_angle, 2, pacman_color)
   
    #right orentation
    elif orientation==1:
        bot_y=image_size[1]//2+height//2
        top_y=image_size[1]//2-height//2
        left_x=image_size[0]//2 
        right_x=image_size[0]//2 +width
        #top_left pacman
        img = add_pacman(img, (left_x,top_y), r, mouth_angle, 0,pacman_color)
        #top_right pacman
        img = add_pacman(img, (right_x,top_y), r, mouth_angle, 1,pacman_color)
        #bottom_left pacman
        img = add_pacman(img, (left_x,bot_y), r, mouth_angle, 3,pacman_color)
        #bottom_right pacman
        img = add_pacman(img, (right_x,bot_y), r, mouth_angle, 2, pacman_color)

    #down orientation
    elif orientation==2:
        bot_y=image_size[1]//2+height
        top_y=image_size[1]//2
        left_x=image_size[0]//2 -width//2
        right_x=image_size[0]//2 +width//2


        #top_left pacman
        img = add_pacman(img, (left_x,top_y), r, mouth_angle, 0,pacman_color)
        #top_right pacman
        img = add_pacman(img, (right_x,top_y), r, mouth_angle, 1,pacman_color)
        #bottom_left pacman
        img = add_pacman(img, (left_x,bot_y), r, mouth_angle, 3,pacman_color)
        #bottom_right pacman
        img = add_pacman(img, (right_x,bot_y), r, mouth_angle, 2, pacman_color)
    
    #left orientation
    elif orientation==3:
        bot_y=image_size[1]//2+height//2
        top_y=image_size[1]//2-height//2
        left_x=image_size[0]//2 - width
        right_x=image_size[0]//2
        #top_left pacman
        img = add_pacman(img, (left_x,top_y), r, mouth_angle, 0,pacman_color)
        #top_right pacman
        img = add_pacman(img, (right_x,top_y), r, mouth_angle, 1,pacman_color)
        #bottom_left pacman
        img = add_pacman(img, (left_x,bot_y), r, mouth_angle, 3,pacman_color)
        #bottom_right pacman
        img = add_pacman(img, (right_x,bot_y), r, mouth_angle, 2, pacman_color)
    return img

def kaniza_cross_rec(
    image_size=(160, 128),
    orientation=0,
    width=52,
    height=52,
    cross_color=light_grey,
    background_color=dark_grey,
    r=13,
    thickness_divisor=4):
    """
    Make a Kanizsa-rectangle image using inward-facing crosses.
    Parameters:
        image_size: (width, height) of the image
        orientation: 0=up, 1=right, 2=down, 3=left
        width, height: dimensions of the rectangle
        cross_color: RGB tuple for the crosses
        background_color: RGB tuple for the background
        r: controls cross size and inward shift
        thickness_divisor: the number r is divided by to compute stroke thickness
    """
    img = Image.new("RGB", image_size, background_color)

    # cross dimensions
    cross_size = 2 * r
    cross_width = max(1, r // thickness_divisor)
    o = cross_width // 2  # outward shift distance

    cx = image_size[0] // 2
    cy = image_size[1] // 2
    half_w = width // 2
    half_h = height // 2

    # compute raw rectangle corner centers
    if orientation == 0:      # up
        y1, y2 = cy - height, cy
        x1, x2 = cx - half_w, cx + half_w
    elif orientation == 1:    # right
        y1, y2 = cy - half_h, cy + half_h
        x1, x2 = cx, cx + width
    elif orientation == 2:    # down
        y1, y2 = cy, cy + height
        x1, x2 = cx - half_w, cx + half_w
    else:                     # left
        y1, y2 = cy - half_h, cy + half_h
        x1, x2 = cx - width, cx

    # shifts to move each cross outward
    shifts = [
        (-o, -o),  # top-left
        (+o, -o),  # top-right
        (-o, +o),  # bottom-left
        (o, o),  # bottom-right
    ]
    centers = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

    # draw the four outward-shifted crosses
    for (raw_x, raw_y), (dx, dy) in zip(centers, shifts):
        img = add_cross(
            img,
            (raw_x + dx, raw_y + dy),
            size=cross_size,
            width=cross_width,
            color=cross_color
        )

    return img

def border_kaniza_sqr_with_square(image_size=(160,128), orientation=0, width=52, 
                                  pacman_color=light_grey, background_color=dark_grey, r=13,
                                  square_line_color=light_grey, square_line_width=1):
    """
    Creates an image with a Kaniza square by drawing four Pac-Man–like inducers using border_kaniza_sqr,
    then draws a square (i.e. lines connecting the centers of the four inducers) over the image.
    
    Parameters:
        image_size (tuple): The size of the image (width, height).
        orientation (int): Orientation of the Kaniza square:
                           0: up, 1: right, 2: down, 3: left.
        width (int): The width (or height) of the Kaniza square element.
        pacman_color (tuple): The RGB color for the Kaniza inducers.
        background_color (tuple): The RGB background color for the image.
        r (int): The radius of the inducers.
        square_line_color (tuple): The RGB color for the connecting square (default black).
        square_line_width (int): The width of the square line.
    
    Returns:
        PIL.Image: The generated image with the inducers and the connecting square.
    """

    # First, generate the Kaniza square image with Pac-Man–like inducers.
    img = border_kaniza_sqr(image_size=image_size, orientation=orientation, width=width,
                            pacman_color=pacman_color, background_color=background_color, r=r)
    
    # Now determine the center coordinates used to position the inducers.
    # These centers correspond to the coordinates used in border_kaniza_sqr.
    if orientation == 0:
        # "Up" orientation:
        # The imaginary square is above the center (one edge crosses the center).
        bot_y = image_size[1] // 2
        top_y = image_size[1] // 2 - width
        left_x = image_size[0] // 2 - width // 2
        right_x = image_size[0] // 2 + width // 2
    elif orientation == 1:
        # "Right" orientation:
        bot_y = image_size[1] // 2 + width // 2
        top_y = image_size[1] // 2 - width // 2
        left_x = image_size[0] // 2
        right_x = image_size[0] // 2 + width
    elif orientation == 2:
        # "Down" orientation:
        bot_y = image_size[1] // 2 + width
        top_y = image_size[1] // 2
        left_x = image_size[0] // 2 - width // 2
        right_x = image_size[0] // 2 + width // 2
    elif orientation == 3:
        # "Left" orientation:
        bot_y = image_size[1] // 2 + width // 2
        top_y = image_size[1] // 2 - width // 2
        left_x = image_size[0] // 2 - width
        right_x = image_size[0] // 2
    else:
        raise ValueError("Orientation must be 0, 1, 2, or 3.")
    
    # The centers for the four inducers drawn in border_kaniza_sqr (they are passed to add_pacman)
    # are:
    top_left = (left_x, top_y)
    top_right = (right_x, top_y)
    bottom_left = (left_x, bot_y)
    bottom_right = (right_x, bot_y)
    
    # Now draw a square connecting these four centers.
    draw = ImageDraw.Draw(img)
    # Define a closed polygon by listing the points in order and closing the loop.
    square_points = [top_left, top_right, bottom_right, bottom_left, top_left]
    
    # Draw the connecting line. (Alternatively, draw.polygon(square_points, outline=square_line_color)
    # if you prefer.)
    draw.line(square_points, fill=square_line_color, width=square_line_width)
    
    return img


def line_border_sqr(image_size=(160,128), orientation=0, width=52, 
                                background_color=dark_grey,
                                  square_line_color=light_grey, square_line_width=1):
    """
    Creates an image with borders of a square
    
    Parameters:
        image_size (tuple): The size of the image (width, height).
        orientation (int): Orientation of the Kaniza square:
                           0: up, 1: right, 2: down, 3: left.
        width (int): The width (or height) of the square element.
        
        background_color (tuple): The RGB background color for the image.
        square_line_color (tuple): The RGB color for the connecting square (default black).
        square_line_width (int): The width of the square line.
    
    Returns:
        PIL.Image: The generated image with the inducers and the connecting square.
    """

    # First, generate the Kaniza square image with Pac-Man–like inducers.
    img = Image.new("RGB", image_size, background_color)
    
    # Now determine the center coordinates used to position the inducers.
    # These centers correspond to the coordinates used in border_kaniza_sqr.
    if orientation == 0:
        # "Up" orientation:
        # The imaginary square is above the center (one edge crosses the center).
        bot_y = image_size[1] // 2
        top_y = image_size[1] // 2 - width
        left_x = image_size[0] // 2 - width // 2
        right_x = image_size[0] // 2 + width // 2
    elif orientation == 1:
        # "Right" orientation:
        bot_y = image_size[1] // 2 + width // 2
        top_y = image_size[1] // 2 - width // 2
        left_x = image_size[0] // 2
        right_x = image_size[0] // 2 + width
    elif orientation == 2:
        # "Down" orientation:
        bot_y = image_size[1] // 2 + width
        top_y = image_size[1] // 2
        left_x = image_size[0] // 2 - width // 2
        right_x = image_size[0] // 2 + width // 2
    elif orientation == 3:
        # "Left" orientation:
        bot_y = image_size[1] // 2 + width // 2
        top_y = image_size[1] // 2 - width // 2
        left_x = image_size[0] // 2 - width
        right_x = image_size[0] // 2
    else:
        raise ValueError("Orientation must be 0, 1, 2, or 3.")
    
    # The centers for the four inducers drawn in border_kaniza_sqr (they are passed to add_pacman)
    # are:
    top_left = (left_x, top_y)
    top_right = (right_x, top_y)
    bottom_left = (left_x, bot_y)
    bottom_right = (right_x, bot_y)
    
    # Now draw a square connecting these four centers.
    draw = ImageDraw.Draw(img)
    # Define a closed polygon by listing the points in order and closing the loop.
    square_points = [top_left, top_right, bottom_right, bottom_left, top_left]
    
    # Draw the connecting line. (Alternatively, draw.polygon(square_points, outline=square_line_color)
    # if you prefer.)
    draw.line(square_points, fill=square_line_color, width=square_line_width)
    
    return img

def square_generator(image_size=(160, 128), orientation=0, width=52, 
                     background_color=(50, 50, 50), square_fill_color=(200, 200, 200)):
    """
    Creates an image with a filled square.

    Parameters:
        image_size (tuple): The size of the image (width, height).
        orientation (int): Orientation of the Kanizsa square:
                           0: up, 1: right, 2: down, 3: left.
        width (int): The width (or height) of the square element.
        background_color (tuple): The RGB background color for the image.
        square_fill_color (tuple): The RGB fill color for the square.

    Returns:
        PIL.Image: The generated image with the filled square.
    """
    # Create a blank image with background color
    img = Image.new("RGB", image_size, background_color)
    draw = ImageDraw.Draw(img)

    # Calculate the four corners of the square
    if orientation == 0:
        bot_y = image_size[1] // 2
        top_y = image_size[1] // 2 - width
        left_x = image_size[0] // 2 - width // 2
        right_x = image_size[0] // 2 + width // 2
    elif orientation == 1:
        bot_y = image_size[1] // 2 + width // 2
        top_y = image_size[1] // 2 - width // 2
        left_x = image_size[0] // 2
        right_x = image_size[0] // 2 + width
    elif orientation == 2:
        bot_y = image_size[1] // 2 + width
        top_y = image_size[1] // 2
        left_x = image_size[0] // 2 - width // 2
        right_x = image_size[0] // 2 + width // 2
    elif orientation == 3:
        bot_y = image_size[1] // 2 + width // 2
        top_y = image_size[1] // 2 - width // 2
        left_x = image_size[0] // 2 - width
        right_x = image_size[0] // 2
    else:
        raise ValueError("Orientation must be 0, 1, 2, or 3.")

    # Define corners of the square
    square_points = [(left_x, top_y), (right_x, top_y),
                     (right_x, bot_y), (left_x, bot_y)]

    # Draw the filled square
    draw.polygon(square_points, fill=square_fill_color)

    return img


def rec_generator(image_size=(160, 128), orientation=0, width=52, height=30,
                  background_color=(50, 50, 50), rect_fill_color=(200, 200, 200)):
    """
    Creates an image with a filled rectangle.

    Parameters:
        image_size (tuple): The size of the image (width, height).
        orientation (int): Orientation of the rectangle:
                           0: up, 1: right, 2: down, 3: left.
        width (int): The width of the rectangle.
        height (int): The height of the rectangle.
        background_color (tuple): The RGB background color.
        rect_fill_color (tuple): The RGB fill color for the rectangle.

    Returns:
        PIL.Image: The generated image with the filled rectangle.
    """
    img = Image.new("RGB", image_size, background_color)
    draw = ImageDraw.Draw(img)
    cx, cy = image_size[0] // 2, image_size[1] // 2

    if orientation == 0:  # extend upward
        left_x = cx - width // 2
        right_x = cx + width // 2
        top_y = cy - height
        bot_y = cy
    elif orientation == 1:  # extend rightward
        left_x = cx
        right_x = cx + width
        top_y = cy - height // 2
        bot_y = cy + height // 2
    elif orientation == 2:  # extend downward
        left_x = cx - width // 2
        right_x = cx + width // 2
        top_y = cy
        bot_y = cy + height
    elif orientation == 3:  # extend leftward
        left_x = cx - width
        right_x = cx
        top_y = cy - height // 2
        bot_y = cy + height // 2
    else:
        raise ValueError("Orientation must be 0, 1, 2, or 3.")

    rect_points = [(left_x, top_y), (right_x, top_y),
                   (right_x, bot_y), (left_x, bot_y)]
    draw.polygon(rect_points, fill=rect_fill_color)
    return img

def non_kaniza_sqr (image_size=(160,128), orientation=0, width=52, pacman_color=light_grey, background_color=dark_grey,r=13):
    """
    make an image with kaniza square, with one of the imaginary edges crossing the center
    of the image

    the kaniza square can either be up, down, right, or, left. 
    to avoid diagnol (step like diagonal edges) 

    Pramenters:
        image_size(width,height): the size of the image
        orientation(int): 0 1 2 3
            0(up) 1(right) 2(down) 3(left)
        pacman_color: RGB colors (,,) of kaniza inducers
        background_color: background color of the entire image
        r: radius of the pacman inducers


    """

    # create new image of image_size
    img = Image.new("RGB", image_size, background_color)
    mouth_angle=90
    
    #up orientation
    if orientation==0:
        bot_y=image_size[1]//2
        top_y=image_size[1]//2-width
        left_x=image_size[0]//2 -width//2
        right_x=image_size[0]//2 +width//2


        #top_left pacman
        img = add_pacman(img, (left_x,top_y), r, mouth_angle, 2,pacman_color)
        #top_right pacman
        img = add_pacman(img, (right_x,top_y), r, mouth_angle, 3,pacman_color)
        #bottom_left pacman
        img = add_pacman(img, (left_x,bot_y), r, mouth_angle, 1,pacman_color)
        #bottom_right pacman
        img = add_pacman(img, (right_x,bot_y), r, mouth_angle, 0, pacman_color)
   
    #right orentation
    elif orientation==1:
        bot_y=image_size[1]//2+width//2
        top_y=image_size[1]//2-width//2
        left_x=image_size[0]//2 
        right_x=image_size[0]//2 +width
        #top_left pacman
        img = add_pacman(img, (left_x,top_y), r, mouth_angle, 2,pacman_color)
        #top_right pacman
        img = add_pacman(img, (right_x,top_y), r, mouth_angle, 3,pacman_color)
        #bottom_left pacman
        img = add_pacman(img, (left_x,bot_y), r, mouth_angle, 1,pacman_color)
        #bottom_right pacman
        img = add_pacman(img, (right_x,bot_y), r, mouth_angle, 0, pacman_color)

    #down orientation
    elif orientation==2:
        bot_y=image_size[1]//2+width
        top_y=image_size[1]//2
        left_x=image_size[0]//2 -width//2
        right_x=image_size[0]//2 +width//2


        #top_left pacman
        img = add_pacman(img, (left_x,top_y), r, mouth_angle, 2,pacman_color)
        #top_right pacman
        img = add_pacman(img, (right_x,top_y), r, mouth_angle, 3,pacman_color)
        #bottom_left pacman
        img = add_pacman(img, (left_x,bot_y), r, mouth_angle, 1,pacman_color)
        #bottom_right pacman
        img = add_pacman(img, (right_x,bot_y), r, mouth_angle, 0, pacman_color)
    
    #left orientation
    elif orientation==3:
        bot_y=image_size[1]//2+width//2
        top_y=image_size[1]//2-width//2
        left_x=image_size[0]//2 - width
        right_x=image_size[0]//2
        #top_left pacman
        img = add_pacman(img, (left_x,top_y), r, mouth_angle, 2,pacman_color)
        #top_right pacman
        img = add_pacman(img, (right_x,top_y), r, mouth_angle, 3,pacman_color)
        #bottom_left pacman
        img = add_pacman(img, (left_x,bot_y), r, mouth_angle, 1,pacman_color)
        #bottom_right pacman
        img = add_pacman(img, (right_x,bot_y), r, mouth_angle, 0, pacman_color)
    return img

def non_kaniza_rec (image_size=(160,128), orientation=0, width=52, height=52 ,pacman_color=light_grey, background_color=dark_grey,r=13):
    """
    make an image with kaniza rectangle, with one of the imaginary edges crossing the center
    of the image

    the kaniza rectangle can either be up, down, right, or, left. 
    to avoid diagnol (step like diagonal edges) 

    Pramenters:
        image_size(width,height): the size of the image
        orientation(int): 0 1 2 3
            0(up) 1(right) 2(down) 3(left)
        pacman_color: RGB colors (,,) of kaniza inducers
        background_color: background color of the entire image
        r: radius of the pacman inducers


    """

    # create new image of image_size
    img = Image.new("RGB", image_size, background_color)
    mouth_angle=90
    
    #up orientation
    if orientation==0:
        bot_y=image_size[1]//2
        top_y=image_size[1]//2-height
        left_x=image_size[0]//2 -width//2
        right_x=image_size[0]//2 +width//2


        #top_left pacman
        img = add_pacman(img, (left_x,top_y), r, mouth_angle, 2,pacman_color)
        #top_right pacman
        img = add_pacman(img, (right_x,top_y), r, mouth_angle, 3,pacman_color)
        #bottom_left pacman
        img = add_pacman(img, (left_x,bot_y), r, mouth_angle, 1,pacman_color)
        #bottom_right pacman
        img = add_pacman(img, (right_x,bot_y), r, mouth_angle, 0, pacman_color)
   
    #right orentation
    elif orientation==1:
        bot_y=image_size[1]//2+height//2
        top_y=image_size[1]//2-height//2
        left_x=image_size[0]//2 
        right_x=image_size[0]//2 +width
        #top_left pacman
        img = add_pacman(img, (left_x,top_y), r, mouth_angle, 2,pacman_color)
        #top_right pacman
        img = add_pacman(img, (right_x,top_y), r, mouth_angle, 3,pacman_color)
        #bottom_left pacman
        img = add_pacman(img, (left_x,bot_y), r, mouth_angle, 1,pacman_color)
        #bottom_right pacman
        img = add_pacman(img, (right_x,bot_y), r, mouth_angle, 0, pacman_color)

    #down orientation
    elif orientation==2:
        bot_y=image_size[1]//2+height
        top_y=image_size[1]//2
        left_x=image_size[0]//2 -width//2
        right_x=image_size[0]//2 +width//2


        #top_left pacman
        img = add_pacman(img, (left_x,top_y), r, mouth_angle, 2,pacman_color)
        #top_right pacman
        img = add_pacman(img, (right_x,top_y), r, mouth_angle, 3,pacman_color)
        #bottom_left pacman
        img = add_pacman(img, (left_x,bot_y), r, mouth_angle, 1,pacman_color)
        #bottom_right pacman
        img = add_pacman(img, (right_x,bot_y), r, mouth_angle, 0, pacman_color)
    
    #left orientation
    elif orientation==3:
        bot_y=image_size[1]//2+height//2
        top_y=image_size[1]//2-height//2
        left_x=image_size[0]//2 - width
        right_x=image_size[0]//2
        #top_left pacman
        img = add_pacman(img, (left_x,top_y), r, mouth_angle, 2,pacman_color)
        #top_right pacman
        img = add_pacman(img, (right_x,top_y), r, mouth_angle, 3,pacman_color)
        #bottom_left pacman
        img = add_pacman(img, (left_x,bot_y), r, mouth_angle, 1,pacman_color)
        #bottom_right pacman
        img = add_pacman(img, (right_x,bot_y), r, mouth_angle, 0, pacman_color)
    return img
    
def circle_sqr(image_size=(160, 128), orientation=0, width=52, circle_color=light_grey, 
                    background_color=dark_grey, r=13):
    """
    Create an image with a Kaniza square, where one of the imaginary edges crosses 
    the center of the image, using full circles instead of Pac-Man shapes.

    Parameters:
        image_size (tuple): The size of the image (width, height).
        orientation (int): Orientation of the Kaniza square:
                           0: up, 1: right, 2: down, 3: left.
        width (int): The width (or height) of the Kaniza square element.
        circle_color (tuple): The fill color (RGB) for the circle inducers.
        background_color (tuple): The background color (RGB) for the entire image.
        r (int): The radius of the circle inducers.
        
    Returns:
        PIL.Image: The generated image.
    """

    # Create a new image with the specified background color
    img = Image.new("RGB", image_size, background_color)

    # For simplicity, we calculate the positions for four circles at the corners 
    # of the "square" element. The square is centered on the image.
    
    if orientation == 0:  # Up orientation
        bot_y = image_size[1] // 2
        top_y = image_size[1] // 2 - width
        left_x = image_size[0] // 2 - width // 2
        right_x = image_size[0] // 2 + width // 2
        
    elif orientation == 1:  # Right orientation
        bot_y = image_size[1] // 2 + width // 2
        top_y = image_size[1] // 2 - width // 2
        left_x = image_size[0] // 2
        right_x = image_size[0] // 2 + width
        
    elif orientation == 2:  # Down orientation
        bot_y = image_size[1] // 2 + width
        top_y = image_size[1] // 2
        left_x = image_size[0] // 2 - width // 2
        right_x = image_size[0] // 2 + width // 2
        
    elif orientation == 3:  # Left orientation
        bot_y = image_size[1] // 2 + width // 2
        top_y = image_size[1] // 2 - width // 2
        left_x = image_size[0] // 2 - width
        right_x = image_size[0] // 2

    # Draw the four circles at the corresponding positions.
    # Top-left circle
    img = add_full_circle(img, (left_x, top_y), r, circle_color)
    # Top-right circle
    img = add_full_circle(img, (right_x, top_y), r, circle_color)
    # Bottom-left circle
    img = add_full_circle(img, (left_x, bot_y), r, circle_color)
    # Bottom-right circle
    img = add_full_circle(img, (right_x, bot_y), r, circle_color)
    
    return img

def circle_rec(image_size=(160, 128), orientation=0, width=52,height=52, circle_color=light_grey, 
                    background_color=dark_grey, r=13):
    """
    Create an image with four circles forming a rectangle , where one of the imaginary edges crosses 
    the center of the image, using full circles instead of Pac-Man shapes.

    Parameters:
        image_size (tuple): The size of the image (width, height).
        orientation (int): Orientation of the Kaniza square:
                           0: up, 1: right, 2: down, 3: left.
        width (int): The width (or height) of the Kaniza square element.
        circle_color (tuple): The fill color (RGB) for the circle inducers.
        background_color (tuple): The background color (RGB) for the entire image.
        r (int): The radius of the circle inducers.
        
    Returns:
        PIL.Image: The generated image.
    """

    # Create a new image with the specified background color
    img = Image.new("RGB", image_size, background_color)

    # For simplicity, we calculate the positions for four circles at the corners 
    # of the "square" element. The square is centered on the image.
    
    if orientation == 0:  # Up orientation
        bot_y = image_size[1] // 2
        top_y = image_size[1] // 2 - height
        left_x = image_size[0] // 2 - width // 2
        right_x = image_size[0] // 2 + width // 2
        
    elif orientation == 1:  # Right orientation
        bot_y = image_size[1] // 2 + height // 2
        top_y = image_size[1] // 2 - height // 2
        left_x = image_size[0] // 2
        right_x = image_size[0] // 2 + width
        
    elif orientation == 2:  # Down orientation
        bot_y = image_size[1] // 2 + height
        top_y = image_size[1] // 2
        left_x = image_size[0] // 2 - width // 2
        right_x = image_size[0] // 2 + width // 2
        
    elif orientation == 3:  # Left orientation
        bot_y = image_size[1] // 2 + height // 2
        top_y = image_size[1] // 2 - height // 2
        left_x = image_size[0] // 2 - width
        right_x = image_size[0] // 2

    # Draw the four circles at the corresponding positions.
    # Top-left circle
    img = add_full_circle(img, (left_x, top_y), r, circle_color)
    # Top-right circle
    img = add_full_circle(img, (right_x, top_y), r, circle_color)
    # Bottom-left circle
    img = add_full_circle(img, (left_x, bot_y), r, circle_color)
    # Bottom-right circle
    img = add_full_circle(img, (right_x, bot_y), r, circle_color)
    
    return img



# Example usage:
if __name__ == "__main__":
    # 1) Single filled‐square figure (Up on dark BG)
    img_single = square_generator(
        image_size=image_size,
        orientation=0,
        width=52,
        background_color=dark_grey,
        square_fill_color=light_grey
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_single)
    ax.axis('off')
    plt.show()

    # 2) Four‐in‐a‐row: Up Dark, Down Light, Up Light, Down Dark (with dashed circle)
    from matplotlib.patches import Circle

    params = [
        (0, dark_grey, light_grey, 'Up Dark'),
        (2, light_grey, dark_grey, 'Down Light'),
        (0, light_grey, dark_grey, 'Up Light'),
        (2, dark_grey, light_grey, 'Down Dark'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, (ori, bg, fill, label) in zip(axes, params):
        img = square_generator(
            image_size=image_size,
            orientation=ori,
            width=52,
            background_color=bg,
            square_fill_color=fill
        )
        ax.imshow(img)
        # dashed circle at center of each
        circ = Circle((image_size[0]/2, image_size[1]/2), 10, fill=False, linestyle='--')
        ax.add_patch(circ)
        ax.axis('off')
        ax.set_title(label)

    plt.tight_layout()
    plt.show()

    # 3) Two rows × five columns: various inducers, up on dark then down on light
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # Row 1: dark BG, up orientation, with titles
    funcs_up = [
        (circle_sqr, dict(image_size=image_size, orientation=0, width=52,
                          circle_color=light_grey, background_color=dark_grey, r=13), 'Circle Sq'),
        (border_kaniza_sqr, dict(image_size=image_size, orientation=0, width=52,
                                 pacman_color=light_grey, background_color=dark_grey, r=13), 'Kanizsa Sq'),
        (non_kaniza_sqr, dict(image_size=image_size, orientation=0, width=52,
                             pacman_color=light_grey, background_color=dark_grey, r=13), 'Non-Kanizsa Sq'),
        (kaniza_cross_sqr, dict(image_size=image_size, orientation=0, width=52,
                               cross_color=light_grey, background_color=dark_grey, r=13, thickness_divisor=2), 'Cross Sq'),
        (square_generator, dict(image_size=image_size, orientation=0, width=52,
                                background_color=dark_grey, square_fill_color=light_grey), 'Fill Sq')
    ]
    for ax, (func, kwargs, title) in zip(axes[0], funcs_up):
        img = func(**kwargs)
        ax.imshow(img)
        # dashed circle on every subplot
        circ = Circle((image_size[0]/2, image_size[1]/2), 10, fill=False, linestyle='--')
        ax.add_patch(circ)
        ax.axis('off')
        ax.set_title(f"{title}\nUp Dark")

    # Row 2: light BG, down orientation, with titles
    funcs_down = [
        (circle_sqr, dict(image_size=image_size, orientation=2, width=52,
                          circle_color=dark_grey, background_color=light_grey, r=13), 'Circle Sq'),
        (border_kaniza_sqr, dict(image_size=image_size, orientation=2, width=52,
                                 pacman_color=dark_grey, background_color=light_grey, r=13), 'Kanizsa Sq'),
        (non_kaniza_sqr, dict(image_size=image_size, orientation=2, width=52,
                             pacman_color=dark_grey, background_color=light_grey, r=13), 'Non-Kanizsa Sq'),
        (kaniza_cross_sqr, dict(image_size=image_size, orientation=2, width=52,
                               cross_color=dark_grey, background_color=light_grey, r=13, thickness_divisor=2), 'Cross Sq'),
        (square_generator, dict(image_size=image_size, orientation=2, width=52,
                                background_color=light_grey, square_fill_color=dark_grey), 'Fill Sq')
    ]
    for ax, (func, kwargs, title) in zip(axes[1], funcs_down):
        img = func(**kwargs)
        ax.imshow(img)
        # dashed circle on every subplot
        circ = Circle((image_size[0]/2, image_size[1]/2), 10, fill=False, linestyle='--')
        ax.add_patch(circ)
        ax.axis('off')
        ax.set_title(f"{title}\nDown Light")

    plt.tight_layout(h_pad=0.3)
    plt.show()

        # Create the four images
    orientations = ['Top', 'Right', 'Down', 'Left']
    images = [circle_sqr(orientation=i, circle_color=light_grey, background_color=dark_grey) for i in range(4)]

    # Plot them side by side
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))

    for ax, img, label in zip(axs, images, orientations):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')
        # Draw red circle at the center
        center_x = img.width // 2
        center_y = img.height // 2
        red_circle = Circle((center_x, center_y), radius=10, edgecolor='red', facecolor='none', linewidth=0.5)
        ax.add_patch(red_circle)

    plt.tight_layout()
    plt.show()
    
    """
     # Test parameters
    base_params = {
        'image_size': (160, 128),
        'width': 52,
        'height': 40,
        'pacman_color': light_grey,
        'cross_color': light_grey,
        'background_color': dark_grey,
        'r': 13,
        'thickness_divisor': 2
    }
    for orientation in range(4):
        # Generate all four images
        img_pac_sq = border_kaniza_sqr(
            image_size=base_params['image_size'],
            orientation=orientation,
            width=base_params['width'],
            pacman_color=base_params['pacman_color'],
            background_color=base_params['background_color'],
            r=base_params['r']
        )
        img_cross_sq = kaniza_cross_sqr(
            image_size=base_params['image_size'],
            orientation=orientation,
            width=base_params['width'],
            cross_color=base_params['cross_color'],
            background_color=base_params['background_color'],
            r=base_params['r'],
            thickness_divisor=base_params['thickness_divisor']
        )
        img_pac_rc = border_kaniza_rec(
            image_size=base_params['image_size'],
            orientation=orientation,
            width=base_params['width'],
            height=base_params['height'],
            pacman_color=base_params['pacman_color'],
            background_color=base_params['background_color'],
            r=base_params['r']
        )
        img_cross_rc = kaniza_cross_rec(
            image_size=base_params['image_size'],
            orientation=orientation,
            width=base_params['width'],
            height=base_params['height'],
            cross_color=base_params['cross_color'],
            background_color=base_params['background_color'],
            r=base_params['r'],
            thickness_divisor=base_params['thickness_divisor']
        )

        # Decide layout: side-by-side for 0 & 2, stacked for 1 & 3
        if orientation in (0, 2):
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        else:
            fig, axes = plt.subplots(4, 1, figsize=(3, 12))

        titles = [
            f'Pac-Man Sq {orientation}',
            f'Cross Sq {orientation}',
            f'Pac-Man Rec {orientation}',
            f'Cross Rec {orientation}'
        ]
        for ax, img, title in zip(axes, [img_pac_sq, img_cross_sq, img_pac_rc, img_cross_rc], titles):
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title)

        plt.tight_layout()
        plt.show()
    """
    # test pacman function code
    """
    # Load an existing image or create a new one (here we create a white image)
    img = Image.new("RGB", image_size, light_grey)
    
    # Pac-Man parameters
    center = (100, 60)      # center of the image
    radius = 7              # size of Pac-Man
    mouth_angle = 90        # mouth opening angle in degrees
    orientation = -45         # mouth faces right (0°)
    
    # Add the Pac-Man figure to the image
    img_with_pacman = add_pacman(img, (50,20), radius, mouth_angle, 0)
    img_with_pacman = add_pacman(img, (100,20), radius, mouth_angle, 1)
    img_with_pacman = add_pacman(img, (100,70), radius, mouth_angle, 2)
    img_with_pacman = add_pacman(img, (50,70), radius, mouth_angle, 3)
    img_with_pacman.show()
    """
    """
    # Test  border_kaniza_sqr()
    radius = 13
    orent=[0,1,2,3]
    image_size=(160,128)
    inducer_color_=light_grey
    background_color_=dark_grey
    width,height=image_size
    middle_y= height//2
    middle_x=width//2
    """
    """
    # test border_kaniza_sqr ():
    for i in range(len(orent)):
        image= border_kaniza_sqr (image_size, orent[i], 52, inducer_color_, background_color_,radius)
        draw =ImageDraw.Draw(image)
        # Draw a red horizontal line across the middle of the image
        draw.line((0, middle_y, width, middle_y), fill='red', width=2)
        # Draw a red vertical line across the middle of the image
        draw.line((middle_x, 0, middle_x, height), fill='red', width=2)

        image.show()

    #test non_kaniza_sqr():
    for i in range(len(orent)):
        image= non_kaniza_sqr (image_size, orent[i], 52, inducer_color_, background_color_,radius)
        draw =ImageDraw.Draw(image)
        # Draw a red horizontal line across the middle of the image
        draw.line((0, middle_y, width, middle_y), fill='red', width=2)
        # Draw a red vertical line across the middle of the image
        draw.line((middle_x, 0, middle_x, height), fill='red', width=2)

        image.show()
 
    #test circle_sqr():
    for i in range(len(orent)):
        image= circle_sqr (image_size, orent[i], 52, inducer_color_, background_color_,radius)
        draw =ImageDraw.Draw(image)
        # Draw a red horizontal line across the middle of the image
        draw.line((0, middle_y, width, middle_y), fill='red', width=2)
        # Draw a red vertical line across the middle of the image
        draw.line((middle_x, 0, middle_x, height), fill='red', width=2)

        image.show()

    # test border_kaniza_sqr_with_square()
    for i in range(len(orent)):
        image= border_kaniza_sqr_with_square (image_size, orent[i], 52, inducer_color_, background_color_,radius,square_line_color=inducer_color_,square_line_width=1)
        draw =ImageDraw.Draw(image)
        # Draw a red horizontal line across the middle of the image
            #draw.line((0, middle_y, width, middle_y), fill='red', width=2)
        # Draw a red vertical line across the middle of the image
            #draw.line((middle_x, 0, middle_x, height), fill='red', width=2)

        image.show()"""

    """# test line_border_sqr()
    for i in range(len(orent)):
        image= line_border_sqr(image_size, orent[i], 52, background_color_,square_line_color=inducer_color_,square_line_width=1)
        draw =ImageDraw.Draw(image)
        # Draw a red horizontal line across the middle of the image
            #draw.line((0, middle_y, width, middle_y), fill='red', width=2)
        # Draw a red vertical line across the middle of the image
            #draw.line((middle_x, 0, middle_x, height), fill='red', width=2)

        image.show()"""