# `ObjectDetectionPipeline`

The core of the system. A VLM "detector" agent proposes bounding boxes, a VLM
"judge" agent critiques them, and the loop repeats with feedback until a score
threshold is hit or rounds run out.

::: free_detection.detection_pipeline