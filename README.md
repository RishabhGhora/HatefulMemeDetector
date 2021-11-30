# Hateful Meme Detector

Tool that extracts text from images, considers text and image as input to multi-input classifier which classifies if image is hateful or not. This repo includs a UI built with dash to showcase the tool as well as a command line interface.

### OCR

Please refer to this [repo](https://github.com/RishabhGhora/OCR) for more details about the OCR algorithm.

### Setup

1. Create python virtual env with python version >= 3.7 `python3.7 -m venv venv`
2. Enter venv `source venv/bin/avtivate`
3. First install Image library to avoid errors with other package installation `pip3 install Image`
4. Install remaining libraries `pip3 install -r requirements.txt`
5. Create dbert directory for text only model `mkdir dbert`
6. Download the weights from [here](https://drive.google.com/drive/folders/1CL-QO4-o5W1ZdDK721CvkCfAXWGXc0ws?usp=sharing) and place in dbert directory
7. Create models directory `mkdir models`
8. Download the final model weights from [here](https://drive.google.com/drive/folders/1Q7845UwHYhQztt_FUXlgRbuiTrVTV9No?usp=sharing)<br /> and place in models directory
   NOTE: make sure CHECKPOINT variable in `transfomers_model.py` is the same as model weight file name
9. To run the UI `python app.py`
10. To use CLI `python hmd sample_images`
