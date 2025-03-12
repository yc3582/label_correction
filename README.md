# Label Correction Web App

Label Correction Web App is a Flask-based application designed for interactive correction of automated label maps generated from pathology images. The application allows users to upload a PNG image along with a MATLAB (.mat) file containing class probability data. It processes these files to generate a label map and provides an interactive drawing interface to refine the label boundaries.

## Features

- **File Upload:** Upload a PNG image and a corresponding MATLAB file containing class probabilities.
- **Automated Label Map Generation:** Processes the MATLAB file using predefined scaling factors for different classes (e.g., background, tumor, sebaceous, etc.) and creates a 1-based indexed label map.
- **Interactive Drawing Interface:** Use an HTML5 Canvas to draw and correct image contours.
- **Tissue Type Selection:** Choose from various tissue types to annotate or adjust the drawing.

## Prerequisites

- Python 3.6 or higher
- A virtual environment (recommended)
- Dependencies as listed in `requirements.txt`

## Installation

1. **Clone the Repository:**

   ```bash
   git clone git@github.com:yc3582/label_correction.git
   cd label_correction

2. **Create a Virtual Environment**

  ```bash
  python3 -m venv venv
  source venv/bin/activate # For Windows use: venv\Scripts\activate

3. **Install Dependencies**
  '''bash
  pip install -r requirements.txt

4. **Run the Label Correction Process**
  '''bash
  python main.py
