# Trackers

Realtime object tracking used by the detection interfaces. `bytetrack.py`
provides a ByteTrack implementation (Kalman filter + IoU linear assignment);
`siamonnx.py` provides a Siamese-ONNX tracker; `trackers.py` combines them in
a `MultiAlgorithmTracker`.

::: free_detection.bytetrack
::: free_detection.siamonnx
::: free_detection.trackers