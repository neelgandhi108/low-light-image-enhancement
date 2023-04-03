# Low-light Image Enhancement 
## Introduction
This project demonstrates the implementation of the MIRNet-v2 model for low-light image enhancement using TensorFlow. The MIRNet-v2 architecture is a fully-convolutional model that learns enriched feature representations for image restoration and enhancement. It can be applied to various image restoration tasks, such as denoising, super-resolution, and deblurring.

## Dataset
The project uses the LoL (LOw Light) dataset, which contains 485 training images and 15 test images. Each image pair consists of a low-light input image and a well-exposed reference image. The input pipeline is built using TensorFlow's tf.data API, which handles large amounts of data, reads from different formats, and performs complex transformations.

## MIRNet-v2 Architecture
The MIRNet-v2 architecture includes a novel feature extraction model, a regularly repeated mechanism for information exchange, a new approach to fuse multi-scale features, and a recursive residual design. The model is trained using Charbonnier Loss, and performance is evaluated using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity (SSIM) metrics. The project also demonstrates distributed training using the tf.distribute.Strategy API with tf.distribute.MirroredStrategy for synchronous training on multiple GPUs.

## Results
The notebook showcases the results of test images from the LoL dataset enhanced by MIRNet-v2 and compares them with images enhanced via the PIL.ImageOps.autocontrast() function. The implementation is based on the papers "Learning Enriched Features for Fast Image Restoration and Enhancement" and "Selective Kernel Networks," and closely follows the official PyTorch implementation of MIRNet-v2 and the TensorFlow implementation of MIRNet-v1.

## Usage
To run the project, follow these steps:

Clone the repository: git clone [https://github.com/yourusername/low-light-image-enhancement](https://github.com/neelgandhi108/low-light-image-enhancement/)
Install the required packages: pip install -r requirements.txt
Open the notebook in Jupyter or any other compatible notebook environment and run the cells.
