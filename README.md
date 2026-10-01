# 2D Cable Curvature Measurement with ArUco Calibration

A Python and OpenCV proof of concept for measuring the two-dimensional curvature and radius of curvature of a flexible cable, hose, tube, or other dark elongated object over a light planar background.

The application supports:

- video files and live webcams;
- planar metric calibration with four ArUco markers;
- perspective rectification to a top-down view;
- configurable margins outside the ArUco marker rectangle;
- automatic Otsu or manual thresholding;
- binary-mask cleanup and connected-component filtering;
- skeleton-based centerline extraction;
- geodesic ordering of skeleton pixels;
- smoothed parametric spline fitting;
- signed curvature and radius-of-curvature estimation;
- camera-pose visualization relative to the ArUco layout;
- an interactive dashboard with playback controls, seeking, sliders, plots, and snapshots.

> This project is a proof of concept. Validate the complete setup against independent physical references before treating the output as metrology-grade data.

## Main script

```text
2d_curvature.py
```

## Project structure

The following structure is recommended:

```text
2d_curvature/
├── 2d_curvature.py
├── README.md
├── sample_videos/
│   └── cable_curvature_demo.mp4
├── snapshots/
├── calibration_aruco.json
└── requirements.txt
```

The suggested folder name for test videos is:

```text
sample_videos/
```

The suggested test-video filename is:

```text
cable_curvature_demo.mp4
```

The test video will be stored in that folder and can be executed with:

```bash
python 2d_curvature.py --video sample_videos/cable_curvature_demo.mp4
```

On Windows, either forward slashes or escaped backslashes may be used:

```cmd
python 2d_curvature.py --video "sample_videos\cable_curvature_demo.mp4"
```

## Video demonstration

A demonstration video can be added below after uploading it through the GitHub README editor.



https://github.com/user-attachments/assets/07d50391-cbe8-43f8-bf89-c600c98c09b6



**Demo video:** _add the uploaded video URL here_

<!-- VIDEO_PLACEHOLDER_START -->

https://github.com/user-attachments/assets/REPLACE-WITH-VIDEO-ID

<!-- VIDEO_PLACEHOLDER_END -->

The demonstration should ideally show:

- detection of the four ArUco markers;
- planar calibration;
- the original and rectified views;
- cable segmentation and the binary mask;
- centerline extraction;
- curvature and radius plots;
- playback controls and parameter sliders.

## Processing pipeline

```text
Video or webcam frame
        |
        v
Optional four-marker ArUco calibration
        |
        v
Perspective rectification and metric scale
        |
        v
Grayscale conversion and Gaussian smoothing
        |
        v
Automatic Otsu or manual thresholding
        |
        v
Morphological cleanup and largest connected component
        |
        v
One-pixel skeleton
        |
        v
Longest geodesic path through the skeleton
        |
        v
Smoothed parametric spline x(t), y(t)
        |
        v
Signed curvature and radius of curvature
        |
        v
Dashboard, plots, pose view, CSV, and PNG snapshots
```

## ArUco and ChArUco assets

ArUco marker images and ChArUco calibration assets can be generated with the companion repository:

- [casfervi/aruco_charuco](https://github.com/casfervi/aruco_charuco)

For the default marker arrangement, generate marker IDs:

```text
0, 1, 2, 3
```

The default OpenCV dictionary used by this application is:

```text
DICT_5X5_250
```

The same dictionary must be used for marker generation and marker detection. Marker ID `2` from `DICT_5X5_250`, for example, is not the same binary pattern as marker ID `2` from `DICT_4X4_50`.

Example marker-generation command in the companion repository:

```bash
python aruco_generation.py \
  --num-markers 4 \
  --start-id 0 \
  --dictionary DICT_5X5_250 \
  --output-dir markers
```

The companion repository also contains ChArUco utilities that can be used to prepare intrinsic camera-calibration assets.

## Default physical marker layout

Place the four markers on the same physical plane as the cable.

The default arrangement is:

```text
ID 0  ----------------------  ID 1
  |                              |
  |       measurement area       |
  |                              |
ID 3  ----------------------  ID 2
```

The `--aruco-ids` argument always follows this positional order:

```text
top-left  top-right  bottom-right  bottom-left
```

For example, if the physical arrangement is:

```text
ID 2  ----------------------  ID 3
  |                              |
  |       measurement area       |
  |                              |
ID 0  ----------------------  ID 1
```

use:

```bash
--aruco-ids 2 3 1 0
```

## Physical measurement arguments

### `--aruco-size`

Physical side length, in millimeters, of the external black square of each ArUco marker.

```bash
--aruco-size 20
```

### `--aruco-spacing`

Physical center-to-center distance between neighboring markers, in millimeters.

Two values define horizontal and vertical spacing:

```bash
--aruco-spacing 225 130
```

This means:

```text
225 mm horizontally between marker centers
130 mm vertically between marker centers
```

One value defines equal horizontal and vertical spacing:

```bash
--aruco-spacing 225
```

Do not use edge-to-edge measurements without converting them to center-to-center spacing.

### `--aruco-margin`

Adds a physical margin, in millimeters, around all four sides of the marker rectangle in the rectified image.

```bash
--aruco-margin 40
```

This allows the rectified image to include regions beyond the ArUco markers, provided those regions remain on the same physical plane.

For example, with:

```text
marker size:       20 mm
horizontal space: 225 mm
vertical space:   130 mm
margin:            40 mm on each side
```

the rectified physical area becomes:

```text
width  = 225 + 20 + 2 x 40 = 325 mm
height = 130 + 20 + 2 x 40 = 230 mm
```

A large margin causes the homography to extrapolate farther outside the marker rectangle. Measurements are normally most reliable inside or near the calibrated marker region.

## Installation

The virtual environment name used by this project is:

```text
2d_curvature
```

### Windows

Create the environment:

```cmd
python -m venv 2d_curvature
```

Activate it in Command Prompt:

```cmd
2d_curvature\Scripts\activate
```

Activate it in PowerShell:

```powershell
.\2d_curvature\Scripts\Activate.ps1
```

Install the dependencies:

```cmd
python -m pip install --upgrade pip
pip install opencv-contrib-python numpy scipy scikit-image matplotlib
```

### Linux or macOS

Create and activate the environment:

```bash
python3 -m venv 2d_curvature
source 2d_curvature/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install opencv-contrib-python numpy scipy scikit-image matplotlib
```

### Verify OpenCV ArUco support

```bash
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'aruco')); print(hasattr(cv2.aruco, 'ArucoDetector'))"
```

The last two values should be:

```text
True
True
```

Avoid installing incompatible versions of `opencv-python` and `opencv-contrib-python` in the same environment because both packages provide the `cv2` module.

## Quick start with the test video

Assuming the test video is stored at:

```text
sample_videos/cable_curvature_demo.mp4
```

run:

```bash
python 2d_curvature.py \
  --video sample_videos/cable_curvature_demo.mp4 \
  --aruco-size 20 \
  --aruco-spacing 225 130 \
  --aruco-dict DICT_5X5_250 \
  --aruco-ids 0 1 2 3 \
  --aruco-margin 40 \
  --calibration-frames 5
```

Windows Command Prompt version:

```cmd
python 2d_curvature.py ^
  --video "sample_videos\cable_curvature_demo.mp4" ^
  --aruco-size 20 ^
  --aruco-spacing 225 130 ^
  --aruco-dict DICT_5X5_250 ^
  --aruco-ids 0 1 2 3 ^
  --aruco-margin 40 ^
  --calibration-frames 5
```

Adjust the marker size, center-to-center spacing, ID order, and margin to match the real test setup.

## Additional usage examples

### Custom marker arrangement

```bash
python 2d_curvature.py \
  --video sample_videos/cable_curvature_demo.mp4 \
  --aruco-size 20 \
  --aruco-spacing 225 130 \
  --aruco-dict DICT_5X5_250 \
  --aruco-ids 2 3 1 0
```

### Webcam

```bash
python 2d_curvature.py \
  --camera 0 \
  --aruco-size 20 \
  --aruco-spacing 225 130 \
  --aruco-margin 40
```

### Fixed scale without ArUco perspective correction

```bash
python 2d_curvature.py \
  --video sample_videos/cable_curvature_demo.mp4 \
  --mm-per-pixel 0.25
```

### Load a saved calibration

```bash
python 2d_curvature.py \
  --video sample_videos/cable_curvature_demo.mp4 \
  --load-calibration \
  --calibration-file calibration_aruco.json
```

Reuse a saved calibration only when the camera position, orientation, resolution, focus, zoom, measurement plane, and marker positions remain unchanged.

## Calibration behavior

The application detects the four expected markers in consecutive frames and then:

1. takes the median marker-corner positions across the calibration frames;
2. estimates a native plane resolution unless `--calibration-px-per-mm` is supplied;
3. calculates a four-center homography that is robust to individual marker rotation;
4. uses all 16 marker corners when their geometry is consistent;
5. applies the requested physical margin;
6. creates a top-down rectified image;
7. stores the metric scale as millimeters per pixel;
8. saves the calibration to JSON.

### Calibration frame count

```bash
--calibration-frames 5
```

More consecutive frames can reduce subpixel detection noise, but all four markers must remain visible during the full collection interval.

### Rectified resolution

By default, the program estimates a native `pixels/mm` value from the apparent marker dimensions.

Force a specific resolution with:

```bash
--calibration-px-per-mm 2
```

This corresponds to:

```text
2 pixels/mm = 0.5 mm/pixel
```

A larger value creates a larger rectified image but does not create physical detail that is absent from the original video.

## Dashboard

The default dashboard contains:

```text
+---------------------------+---------------+-------------------+
| RESULT                    | INFORMATION   | CAMERA POSE       |
+---------------------------+---------------+-------------------+
| ORIGINAL / RECTIFIED      | MASK          | CONTROLS          |
+---------------------------+---------------+-------------------+
| CURVATURE                 | RADIUS OF CURVATURE              |
+---------------------------+-----------------------------------+
```

The polished dashboard theme includes:

- grouped measurement cards;
- calibration status;
- dark-themed graphs;
- playback controls;
- a seek bar;
- threshold and spline-smoothing sliders;
- camera-pose visualization.

## Playback controls

- `-1s`: move backward approximately one second;
- `<`: previous frame;
- `PLAY/PAUSE`: toggle playback;
- `>`: next frame;
- `+1s`: move forward approximately one second;
- progress bar: click or drag to seek through a video.

When a video reaches the end, the application remains on the last frame. Pressing play again restarts the video from the beginning.

## Segmentation controls

### Threshold

Threshold separates the dark object from the light background.

- `0` enables automatic Otsu thresholding;
- a low manual value may fragment the cable;
- a high manual value may include shadows or background regions.

Use the mask panel to verify that the cable is continuous and white while the background remains black.

### Spline smoothing

Spline smoothing controls how closely the final centerline follows pixel-level skeleton irregularities.

- a low value preserves local bends but may produce noisy curvature;
- a high value reduces noise but may flatten real bends and increase the reported radius;
- start around `0.5` to `1.0 px` and increase gradually when necessary.

## Keyboard and mouse controls

```text
Left click       Select nearest centerline point and display local radius
Right click      Clear the selected point
C                Rebuild ArUco calibration
Space            Play or pause
S                Save snapshot, CSV, and graph image
Q or Esc         Exit
```

## Segmentation assumptions

The program assumes a dark object over a light background.

The segmentation pipeline uses:

1. BGR-to-grayscale conversion;
2. Gaussian blur;
3. inverted binary thresholding;
4. morphological opening;
5. morphological closing;
6. connected-component analysis;
7. selection of the largest valid component.

The largest dark connected component is assumed to be the cable. Hands, shadows, supports, and other large dark objects may therefore cause incorrect segmentation.

For best results:

- use a matte light background;
- use a dark cable;
- keep hands outside the measurement area;
- use uniform lighting;
- avoid strong shadows and reflections;
- keep the cable in the marker plane.

## Centerline, curvature, and radius

The binary mask is reduced to a one-pixel skeleton. The application builds an eight-neighbor graph and uses two Dijkstra searches to obtain the longest geodesic skeleton path while suppressing short branches.

A parametric spline is fitted to the ordered points:

```text
x(t), y(t)
```

Signed curvature is calculated as:

```text
k = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)
```

Radius of curvature is:

```text
R = 1 / |k|
```

Curve endpoints are trimmed from summary statistics because skeleton and spline derivatives are less stable near the ends.

## Camera pose panel

When the four ArUco markers are visible, the application estimates and displays the camera pose relative to the marker plane.

The panel includes:

- ArUco-plane axes;
- camera axes;
- a schematic camera frustum;
- distance from the plane origin;
- camera height;
- camera X and Y displacement;
- approximate inclination;
- pose reprojection error.

If the real focal length in pixels is known, provide:

```bash
--focal-px 1500
```

Without `--focal-px`, the application assumes approximately `0.8 x image width`. This is only a rough pose-estimation aid and is not a substitute for intrinsic camera calibration.

## Output and snapshots

Press `S` to save outputs in the directory provided by `--output`. The default is:

```text
snapshots/
```

Outputs can include:

- result image with centerline;
- CSV with arc length, X, Y, curvature, and radius;
- PNG plots for curvature and radius.

Example CSV columns:

```text
s_mm
x_mm
y_mm
curvature_1/mm
radius_mm
```

Without metric calibration, units are pixels and inverse pixels.

## Important limitations

### Planar assumption

The homography assumes that all four markers and the cable lie approximately on the same physical plane. A cable segment moving toward or away from the camera cannot be measured correctly with one planar homography.

### Margin and extrapolation

`--aruco-margin` allows the rectified image to include areas beyond the markers. Those areas must still belong to the same plane. Error may grow farther away from the marker rectangle because the homography is being extrapolated.

### Resolution is not accuracy

A rectified scale such as `2 px/mm` describes output sampling. It does not guarantee `0.5 mm` measurement accuracy.

Actual error also depends on:

- marker-size accuracy;
- center-to-center spacing accuracy;
- print quality;
- camera focus and resolution;
- lens distortion;
- segmentation quality;
- spline smoothing;
- cable thickness;
- measurement-plane stability.

### RMS error

When calibration uses exactly four marker centers, reprojection error at those same points can approach zero by construction. Validate the calibration with independent known dimensions that were not used to calculate the homography.

### Recommended validation

Place an independent reference in the measurement plane. If a known 100 mm distance is present and the rectified scale is 2 px/mm, the reference should span approximately 200 pixels.

Calculate:

```text
absolute error = |measured value - reference value|
relative error = absolute error / reference value x 100%
```

Repeat validation in several regions of the rectified image, including any area added through `--aruco-margin`.

## Command-line reference

Display all available arguments:

```bash
python 2d_curvature.py --help
```

Important options:

```text
--video PATH
--camera INDEX
--smoothing VALUE
--mm-per-pixel VALUE
--output DIRECTORY
--aruco-size MM
--aruco-spacing MM [MM]
--aruco-ids TL TR BR BL
--aruco-dict NAME
--aruco-margin MM
--calibration-px-per-mm VALUE
--focal-px VALUE
--calibration-frames COUNT
--calibration-file PATH
--load-calibration
--display dashboard|separate
--dashboard-width PIXELS
--dashboard-height PIXELS
```

## References

- [casfervi/aruco_charuco](https://github.com/casfervi/aruco_charuco)
- [OpenCV: Detection of ArUco Markers](https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html)
- [OpenCV: Detection of ArUco Boards](https://docs.opencv.org/4.13.0/db/da9/tutorial_aruco_board_detection.html)
- [OpenCV: Detection of ChArUco Boards](https://docs.opencv.org/4.13.0/df/d4a/tutorial_charuco_detection.html)
- [OpenCV: Calibration with ArUco and ChArUco](https://docs.opencv.org/4.13.0/da/d13/tutorial_aruco_calibration.html)
- [OpenCV: Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

