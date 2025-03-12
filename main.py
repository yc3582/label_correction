from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, session
import os
import scipy.io as sio  # For loading the .mat file
from werkzeug.utils import secure_filename
from PIL import Image  # For getting the image dimensions
import numpy as np
from matplotlib.path import Path

app = Flask(__name__)

# Set the secret key for session management
app.secret_key = 'this_is_my_secret_key_place_holder'

# Define the folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

#displays file uplaod page
@app.route('/')
def index():
    # Render the file upload page (index.html)
    return render_template('index.html')

def create_label_map(mat_path):
    # Load the MAT file, adjust class probabilities, and return a label map
    mat_data = sio.loadmat(mat_path)
    if 'class_probs' not in mat_data:
        raise KeyError("The MAT file does not contain 'class_probs'.")
    class_probs = mat_data['class_probs']
    bg_sc = 0.8 #background scale
    tm_sc = 2   #tumor scale
    sb_sc = 4   #sebacious scale
    ep_sc = 2   #epi scale
    ec_sc = 2   #eccrine scale
    nt_sc = 1   #no tissue scale
    hf_sc = 2   #hair follicle scale

    class_probs[:,:,0] = class_probs[:,:,0] * bg_sc
    class_probs[:,:,1] = class_probs[:,:,1] * tm_sc
    class_probs[:,:,2] = class_probs[:,:,2] * sb_sc
    class_probs[:,:,3] = class_probs[:,:,3] * ep_sc
    class_probs[:,:,4] = class_probs[:,:,4] * ec_sc
    class_probs[:,:,5] = class_probs[:,:,5] * nt_sc
    class_probs[:,:,6] = class_probs[:,:,6] * hf_sc

    label_map = np.argmax(class_probs, axis=-1) + 1  # add 1 to convert from 0-based to 1-based indexing
    return label_map

# Saves the PNG and MAT files, then loads the drawing page
@app.route('/upload', methods=['POST'])
def upload():
    # Check if both files are provided in the request
    if 'png_file' not in request.files or 'mat_file' not in request.files:
        return redirect(url_for('index'))

    png_file = request.files['png_file']
    mat_file = request.files['mat_file']

    if png_file.filename == '' or mat_file.filename == '':
        return redirect(url_for('index'))

    
    # Secure filenames and save them to the upload folder
    png_filename = secure_filename(png_file.filename)
    mat_filename = secure_filename(mat_file.filename)
    png_path = os.path.join(app.config['UPLOAD_FOLDER'], png_filename)
    mat_path = os.path.join(app.config['UPLOAD_FOLDER'], mat_filename)
    png_file.save(png_path)
    mat_file.save(mat_path)

    session['png_filename'] = png_filename

    label_map = create_label_map(mat_path)
    label_map_path = os.path.join(app.config['UPLOAD_FOLDER'], "label_map.npy")
    np.save(label_map_path, label_map)

    # Render the drawing page, passing the PNG filename and MAT file name to the template.
    return render_template('draw.html', image_path=png_filename, mat_file=mat_filename)

# serves the uploaded image so it can be displayed in the browser
@app.route('/get_image/<filename>')
def get_image(filename):
    # Serves the uploaded image to the browser.
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def get_pixels_in_contour(points, image_shape):
    '''
    Given a list of point (each with keys 'x' and 'y') and the image shape (height, width),
    return a list of [row,col] indices that fall inside the polygon defined by these points.
    '''
    polygon = [(pt['x'], pt['y']) for pt in points]
    height, width = image_shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    grid_points = np.vstack((x.flatten(), y.flatten())).T
    poly_path = Path(polygon)
    mask = poly_path.contains_points(grid_points)
    mask = mask.reshape((height, width))
    pixel_indices = np.argwhere(mask)  # [row, col] pairs
    return pixel_indices.tolist()

def override_label_map(labelmap, all_contour_pixels):
    """
    Update the label_map with the tissue type based on contour pixel data.
    Expecting each entry in all_contour_pixels to have a 'tissue' and 'pixels'
    (where each pixel is [row, col]).
    """
    for contour in all_contour_pixels:
        tissue = contour['tissue']
        pixels = contour['pixels']
        for row, col in pixels:
            labelmap[row, col] = tissue
    return labelmap

def rgb_to_img(rgb_image):
    """
    Convert an RGB array (with values 0-1) to a PIL image.
    """
    img = Image.fromarray((rgb_image * 255).astype(np.uint8))
    return img
def create_rgb(label_map):
    """
    Map labe_map values (1-based) to RGB colors.
    """
    CLRs = np.array([
        [0.5, 0.5, 0.5],  # Background    - Gray
        [1.0, 0.0, 0.0],  # tumor         - Red
        [0.0, 1.0, 0.0],  # epi           - Green
        [0.0, 0.0, 1.0],  # sebaceous     - Blue
        [1.0, 0.0, 1.0],  # eccrine       - Violet
        [0.0, 1.0, 1.0],  # No Tissue     - Cyan
        [1.0, 1.0, 0.0],  # Hair Follicle - Yellow
        [1.0, 0.5, 0.0]   # 2nd tumor     - Orange
    ])
    indices = label_map - 1
    rgb_image = CLRs[indices]
    return rgb_image
@app.route('/save_contours', methods=['POST'])
def save_contours():
    """
    Receive the JSON contour data from the client, update the label map based on contours,
    and generate RGB images before and after correction.
    """

    data = request.get_json()
    print("Received contour data:", data)
    #png_file = request.files['png_file']
    #png_filename = secure_filename(png_file.filename)
    #png_path = os.path.join(app.config['UPLOAD_FOLDER'], png_filename)
    #print(png_file)
    #print(png_filename)
    #print(png_path)
    png_filename = session.get('png_filename', None)
    if not png_filename:
        return jsonify({'status': 'error', 'message': 'Image filename not found in session.'})
    #image_filename = "img2.png"  # adjust as needed
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], png_filename)

    try:
        with Image.open(image_path) as im:
            width, height = im.size  # PIL returns (width, height)
            image_shape = (height, width)
    except Exception as e:
        print("Error opening image:", e)
        # For testing, you could set a default shape, e.g., 512x512.
        image_shape = (512, 512)
    
	# Process each contour: convert the clicked points to all pixel coordinates inside the contour.
    all_contour_pixels = []
    for contour in data:
        points = contour['points']
        tissue = contour['tissue']
        pixels = get_pixels_in_contour(points, image_shape)
        all_contour_pixels.append({
            'tissue': tissue,
            'pixels': pixels
        })
        # Print out the pixel data for debugging.
        print(f"Contour for tissue {tissue} includes {len(pixels)} pixels.")


    # Save the contour data to a text file.
    #with open("saved_contours.txt", "w") as f:
    #    f.write(str(all_contour_pixels))
    #print("Contour data saved successfully to saved_contours.txt")

    # Load the label map.
    label_map_path = os.path.join(app.config['UPLOAD_FOLDER'], "label_map.npy")
    label_map = np.load(label_map_path)

    # Create and save the initial RGB iamge
    rgb_before = create_rgb(label_map)
    img_before = rgb_to_img(rgb_before)
    img_before.save(os.path.join(app.config['UPLOAD_FOLDER'], "initial_label_map.png"))

    # Apply the contour data to the label map.
    corrected_labelmap = override_label_map(label_map, all_contour_pixels)
    rgb_after = create_rgb(corrected_labelmap)
    img_after = rgb_to_img(rgb_after)
    img_after.save(os.path.join(app.config['UPLOAD_FOLDER'], "corrected_label_map.png"))




    #corrected_labelmap = override_label_map(labelmap, all_contour_pixels)
    # Here you would process the contours:
    # - Call your existing code to create the label map from the MAT file.
    # - Correct the label map based on the contour data.
    # - Save the final label image and the underlying matrix.
    # For now, we leave this section as a placeholder.
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
