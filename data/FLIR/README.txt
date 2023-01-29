Teledyne FLIR Free ADAS Thermal Dataset v2

By downloading this data you agree to the following terms of service: https://www.flir.com/oem/adas/adas-dataset-agree/

The Teledyne FLIR free starter thermal dataset provides fully annotated thermal and visible spectrum frames for development of
object detection neural networks. This data was constructed to encourage research on visible + thermal spectrum sensor
fusion algorithms ("RGBT") in order to advance the safety of autonomous vehicles.  A total of 26,442 fully-annotated
frames are included with 15 different object classes.

Data Organization

Data is organized by three splits: training images ("train"), validation images ("val"), and validation 
video ("val_video"). Frames from each of the three categories were sampled from completely independent video sequences
to encourage model generalization. "video" refers to sequential frames at the maximum sampling rate, whereas "images" are
more carefully selected for optimal object detector training. More details are provided on this in the Data Capture section.
The data organization and splits are a suggestion to facilitate reproducible research, however the end-user is free to
use a different scheme to fit their specific needs. A total of 9,711 thermal and 9,233 RGB training/validation images
are included with a 90%/10% train/val split. The validation videos make up 7,498 total frames. 
The 3,749 thermal/RGB video pairs each consist of a series of continuous frames captured at 
30 frames per second (FPS). Each video frame in one spectrum is mapped to its time-synced frame pair in the other spectrum,
specified in the file rgb_to_thermal_vid_map.json. The footage in this dataset was collected at a variety of locations, and
includes various lighting/weather conditions. See "extra_info" in the images section of the annotations (coco.json).

Data Capture

The dataset was acquired via a thermal and visible camera pair mounted on a vehicle. Thermal images were acquired with 
a Teledyne FLIR Tau 2 13 mm f/1.0 with a 45-degree horizontal field of view (HFOV) and 37-degree vertical field of view (VFOV).
The thermal camera was operating in T-linear mode mode.
Visible images were captured with a Teledyne FLIR BlackFly S BFS-U3-51S5C (IMX250) camera with a 52.8 degree HFOV Edmund Optics lens (part number 68215).
Time-synced capture was executed by Teledyne FLIR's Guardian software. Validation videos were collected at a frame
rate of 30 frames per second and include target IDs which allow a developer to compute tracking metrics such
as MOTA/MOTP. The train/val image segment frames were sampled across a wide range of diverse footage to ensure high
levels of diversity for training and validation. In some cases the frame samples were chosen manually and in other
cases a frame skip was used. Each video typically uses a different frame skip. The curation team excluded redundant 
footage, such as nearly identical frames when the vehicle is waiting at a red light.

Annotations

All annotations for a given split (folder) are provided in a file called index.json. This file format was created for 
Teledyne FLIR's proprietary dataset management tool called Conservator. It resembles Microsoft's COCO annotation file 
format, but allows for richer annotations with frame-level and annotation-level attributes. In addition to the Conservator
JSON files, we included COCO-formatted files named coco.json which contain a filtered version of the annotations. The 
filtered version has remapped or removed some superfluous or redundant classes like "face" and "license plate". The 
included Conservator JSON files provides the raw / unfiltered data from the annotators while the COCO JSON file is a refined 
version ready for experimentation.

The COCO-formatted files (coco.json) are recommended for training models. 
The reported metrics are based on models that were trained on the COCO-formatted files.

Labels

A modified MSCOCO label map was used with conventions that were largely inspired by the Berkeley Deep Drive dataset.
The following classes are included:
• Category Id 1:  person
  • Category Id 2:  bike (renamed from "bicycle")
  • Category Id 3:  car (this includes pick-up trucks and vans)
  • Category Id 4:  motor (renamed from "motorcycle" for brevity)
  • Category Id 6:  bus
  • Category Id 7:  train
  • Category Id 8:  truck (semi/freight truck, excluding pickup truck)
  • Category Id 10: light (renamed from "traffic light" for brevity)
  • Category Id 11: hydrant (renamed "fire hydrant" for brevity)
  • Category Id 12: sign (renamed from "street sign" for brevity)
  • Category Id 17: dog
  • Category Id 37: skateboard
  • Category Id 73: stroller (four-wheeled carriage for a child, also called pram)
  • Category Id 77: scooter
  • Category Id 79: other vehicle (less common vehicles like construction equipment and trailers)

Annotation Counts

Thermal Image Annotations
+---------------+-------------+------------+
| Label         | Train       | Val        |
+---------------+-------------+------------+
| person        | 50,478      | 4,470      |
| bike          | 7,237       | 170        |
| car           | 73,623      | 7,133      |
| motor         | 1,116       | 55         |
| bus           | 2,245       | 179        |
| train         | 5           | 0          |
| truck         | 829         | 46         |
| light         | 16,198      | 2,005      |
| hydrant       | 1,095       | 94         |
| sign          | 20,770      | 2,472      |
| dog           | 4           | 0          |
| deer          | 8           | 0          |
| skateboard    | 29          | 3          |
| stroller      | 15          | 6          |
| scooter       | 15          | 0          |
| other vehicle | 1,373       | 63         |
| Total     | 175,040 | 16,696 |
+---------------+-------------+------------+

Visible Image Annotations
+---------------+-------------+------------+
| Label         | Train       | Val        |
+---------------+-------------+------------+
| person        | 35,007      | 3,223      |
| bike          | 7,560       | 193        |
| car           | 71,281      | 7,285      |
| motor         | 1,837       | 77         |
| bus           | 1,879       | 183        |
| train         | 9           | 0          |
| truck         | 1,251       | 47         |
| light         | 18,640      | 2,143      |
| hydrant       | 990         | 126        |
| sign          | 29,531      | 3,581      |
| skateboard    | 412         | 4          |
| stroller      | 38          | 7          |
| scooter       | 41          | 0          |
| other vehicle | 698         | 40         |
| Total     | 169,174 | 16,909 |
+---------------+-------------+------------+

Annotation Guidelines

Human annotators were instructed to make bounding boxes as tight as possible. Tight bounding boxes that omitted small parts 
of the object, such as extremities, were favored over broader bounding boxes. Personal accessories were not included 
in the bounding boxes on people. When occlusion occurred, only non-occluded parts of the object were annotated.
Heads and shoulders were favored for inclusion in the bounding box over other parts of the body. When occlusion allowed
only parts of limbs or other minor parts of an object to be visible, they were not annotated. Wheels were the most 
important part of the "bike" category.  Bicycle parts typically occluded by riders, such as handlebars, were 
not included in the bounding box. People riding the bicycle were annotated separately from the bicycle (with the 
corresponding "is_rider" flag).

For many objects the occlusion and truncation information is also noted. In addition, a flag is included to
denote if an object is a reflection (e.g. a pedestrian reflecting off of a car window or puddle). 
For the "person" category, an additional flag is included if they are a rider (on a bicycle, motorcycle, or other vehicle).

Download Contents

Once data is extracted, the following directories are exposed:
  • images_rgb_train
  • images_rgb_val
  • images_thermal_train
  • images_thermal_val
  • video_rgb_test
  • video_thermal_test
Additionally, there is a metadata file that maps rgb to thermal frames for the video sets called rgb_to_thermal_vid_map.json
In each of the top-level directories are the following:
  • data: directory containing jpeg image files. 8-bit post-processed images in the case of thermal directories
    and RGB images in the case of rgb directories
• analyticsData: (only in thermal directories) directory containing 16-bit raw thermal images
• index.json: annotations in Teledyne FLIR Conservator format.
• coco.json: annotations in MSCOCO format. This is a filtered form of index.json
• coco_annotation_counts.txt and coco_annotation_counts.tsv: table of summary of annotations for each directory in 
    two different file formats

For visible spectrum images some minimal blurring has been applied to license plates and faces to make them illegible.

Please contact the Teledyne FLIR Automotive team at ADAS-Support@flir.com for assistance.

Baseline Model

Baseline accuracy for object detection was established using the YOLOX-m neural network designed for 640 X 640 images. 
Both the RGB and thermal detectors were pre-trained on MSCOCO data (https://arxiv.org/abs/2107.08430 and 
https://github.com/Megvii-BaseDetection/YOLOX). The base neural networks were trained on the training set data provided 
in this dataset and tested on the video test data also provided in this dataset. http://cocodataset.org/#detection-eval 
was used for accuracy assessment criteria.
The following AP @ IoU=0.5 scores were obtained:
   • RGB:
     • person: 51.42
     • car: 55.79
   • Thermal:
     • person: 75.33
     • car: 77.23



Release Notes

ADAS Dataset v2.0.0 (January 19, 2022)

• Expanded labels to 15 categories vs 5 original categories.
• Expanded annotated frames to 26,442 total (9,233 visible-spectrum images, 9,711 thermal-spectrum and 7,498 matched thermal/visible video frames), an +83% increase compared to the v1 release.
• All included images are annotated with 15 categories.
• Image data added from England and France.
• Validation video data added from various locations in the US: Northern California, Idaho, and Michigan.

This dataset was composed of multiple smaller datasets hosted on Conservator. Dataset IDs and their commit hashes:
```
Dataset ID         Commit Hash

nwehqYrxd4Nd3tGv6: 841232451ba2ddfb5436ce310e32253e8c044a80
YuWMYT2WLitDfyaEo: 9892b0586232563ae6b8d09370003f8a6274d34c
kM77KS9iGCXY3hupQ: 0bb257982e7a092f6ba928b6a33d7ed9575401ff
2TWc2H3cxPc662hbZ: 5b2b4a90d23157880198686a97c2653042780f2a
XD3eTyiXvAE79zJBX: 43b88bd337ff165c05966ff9a00eeb50c18a2fc7
bfmsZ6gKqdR7gq4nq: 8bf30d2ad2edd050a46c5064fdf99c49de0713d0
ntmccBPG3SbDbgGNf: e219f8df0465484927e986b6ac173b738fe2bdce
DSbDDZAeSBuwFy5gd: 25a5c3de8d27fed76fe1a1c036e604c8230d9f31
iaQuoXEE9nBnyXH7Y: 931ea91cd46d1eee9f386843adfc40d16b8b3af8
2u7gWwBp6CnkqAtZQ: 0c1a22f75d5a0532c4d7e42b11368ef40001bfef
```

ADAS Dataset v1.0.0 (Jun 19, 2018)

The initial version of the dataset released included
  • 5 label categories.
• 14,452 total annotated frames (10,228 images and 4,224 video frames).
• Data from the release was captured in the Santa Barbara, California, USA area.
• The original download can be found here: https://flir.box.com/s/suwst0b3k9rko35homhr3rnyytf3102d.



