import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from drawing_pacman import border_kaniza_rec

# Define color values
light_grey_value = 255 * 2 // 3
dark_grey_value = 255 // 3
light_grey = (light_grey_value,) * 3
dark_grey = (dark_grey_value,) * 3

# Define 9 shape configurations
widths = [48, 52, 56, 60, 64, 68, 72, 76, 80]
heights = [48, 47, 46, 45, 44, 43, 42, 41, 40]
radii = [12, 13, 14, 15, 16, 17, 18, 19, 20]
shape_triplets = list(zip(widths, heights, radii))

# Generate 9 Kanizsa rectangle images
images = []
for width, height, r in shape_triplets:
    img = border_kaniza_rec(
        image_size=(160, 128),
        orientation=0,
        width=width,
        height=height,
        pacman_color=light_grey,
        background_color=dark_grey,
        r=r
    )
    images.append(img)

# Plot in 2 rows: 5 images on top (axs[0-4]), 4 on bottom (axs[5-8]), leave axs[9] blank
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.flatten()

# Turn off all axes
for ax in axs:
    ax.axis('off')

# First row: 5 images
for i in range(5):
    ax = axs[i]
    ax.imshow(images[i])
    ax.set_title(f"Kanizsa {i+1}")
    circ = Circle((160 // 2, 128 // 2), radius=10, edgecolor='black',
                  facecolor='none', linestyle='--', linewidth=0.5)
    ax.add_patch(circ)

# Second row: next 4 images
for i in range(4):
    ax = axs[i + 5]
    ax.imshow(images[i + 5])
    ax.set_title(f"Kanizsa {i + 6}")
    circ = Circle((160 // 2, 128 // 2), radius=10, edgecolor='black',
                  facecolor='none', linestyle='--', linewidth=0.5)
    ax.add_patch(circ)

# axs[9] is left blank
axs[9].axis('off')

plt.subplots_adjust(hspace=0.6)
plt.tight_layout()
plt.show()