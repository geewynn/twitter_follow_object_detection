# twitter_follow_object_detection

This project is a test challeng to test my skills on building an object detection model from scratch with my own dataset.

This project requires me to detect the twitter follow button using transfer learning.

I used the yolov3 pretrreained model.

### Data Processing
- First step I annotated the images using labelImg tool. This tool creates an XML file for each annotated image. 
- Created a train and valdation dataset for training the pretrained model. Each of the train/validation folder contain 2 other subfolders images containing the image files and annotation folder containing the corresponding annotated files.
- I have 5 images overall. The train contains 3 image while the validation has 1 and the Last 1 for testing the final model.

### Object Detection
- I used the imageAI object detection Library which uses yolov3 for object detection.
- I installed imageAI from the following link https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/hololens.zip
- Installed Tensorflow 1.13.1. ImageAI currently support this version of tensorflow.
- I downloaded the pretrained Yolov3 model from here 
https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/pretrained-yolov3.h5

### Training the Model
```
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="/content/followButton/")
trainer.setTrainConfig(object_names_array=["follow"], batch_size=2, num_experiments=50, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()

```
- The first imports the Model trainer from ImageAI
- Line 2 and 3 creates instance and set our model to Yolov3.
- 4th sets data path.
- 5th sets the names array(follow object). batch size = 2, number of training examples/num of epochs, set the model to the pretrained model.
-  I used a batch size of 2 bcause I had just 3 traning example. This also represents mini batch gradient descent.
- The last line starts our model training.

** The model trains for approximateley 20 minutes.

### Model Evaluation
```

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="/content/followButton")
trainer.evaluateModel(model_path="/content/followButton/models", json_path="/content/followButton/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)

```

- In the above code we first set an instance of our model and then set the model to Yolov3
- Set the data directory
- Set the various trained models directory, and sets the evaluation thresholds

```
from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("/content/followButton/models/detection_model-ex-018--loss-0016.013.h5") 
detector.setJsonPath("/content/followButton/json/detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="/content/4.png", output_image_path="/content/detect-4.png")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

```
- In the above code the 1st line imports the CustomObjectDetection from ImageAI
- 2nd creates an instance of the object detection.
- 3rd set the model to Yolov3.
- 4th Set the model paths to the best performing model.
- 5th set the path to the model Json detector.
- 6th loads the model.
- 7th line trys out the custom image detector on the test data, and prints out the name, probability and box points.

