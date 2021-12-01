# Hateful Meme Detector

Tool that extracts text from images, considers text and image as input to multimodal BiTransformer which classifies if image is hateful or not. This repo includs a UI built with dash to showcase the tool as well as a command line interface.

### OCR

Please refer to this [repo](https://github.com/RishabhGhora/OCR) for more details about the OCR algorithm.

### Setup

1. Create python virtual env with python version >= 3.7 `python3.7 -m venv venv`
2. Enter venv `source venv/bin/avtivate`
3. First install Image library to avoid errors with other package installation `pip3 install Image`
4. Install remaining libraries `pip3 install -r requirements.txt`
5. Create models directory for Transformers model `mkdir models`
6. Download the final model weights from [here](https://drive.google.com/drive/folders/1Q7845UwHYhQztt_FUXlgRbuiTrVTV9No?usp=sharing)<br /> and place in models directory
   NOTE: make sure CHECKPOINT variable in `transfomers_model.py` is the same as model weight file name

### Usage

1. To run the UI `python app.py`
2. To use CLI `python hmd sample_images`

This tool accepts images as input and returns the extracted text from the image and the classification of the image + text whether it is hateful or not and its probability. Comnand line tool creates csv from input images in sample_images folder.
