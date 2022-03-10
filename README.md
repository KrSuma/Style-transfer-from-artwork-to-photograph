Project for the Machine Learning Workkshop course.

Implements style transfer, based on the method proposed by Gatys et al. (https://arxiv.org/abs/1508.06576).
Current implementation is largely based on the code provided by Gatys(https://github.com/leongatys/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb).

Usage:

Before running the code, install CUDA on your device to reduce the amount of time style transfer takes. 
Then install the correct version of PyTorch from the website.

You must download the vgg19 model and install it in the model/vgg19 subdirectory here:
https://bethgelab.org/media/uploads/pytorch_models/vgg_conv.pth

also:
(from conda interpreter. just doing 'conda install PIL' didnt work for me.)
easy_install PIL

To run the gui:
    In the main folder, run > python gui.py

To run style transfer:

    (style.py)

    # Set these to the path to the content and style images
    CONTENT_IMG_PATH = 'images/old-house.jpg'
    STYLE_IMG_PATH = 'styles/van-gogh-cottages.jpg'

    # Set to path of output
    RESULT_IMG_PATH = 'results/loss_diff_test_cottages.jpg'
    PARTIAL_RESULT_IMG_PATH = 'partial_results/house_as_gogh_cottages.jpg'

    # Init stylizer
    stylizer = Stylizer(content_weight=400, style_weight=1600)

    # Run a set number of iterations
    stylizer.run(CONTENT_IMG_PATH, STYLE_IMG_PATH, RESULT_IMG_PATH, iterations=100)
