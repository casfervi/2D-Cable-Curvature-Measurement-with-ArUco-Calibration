# 2D Cable Curvature Measurement with ArUco Calibration

A Python and OpenCV application for measuring the two-dimensional curvature and radius of curvature of a flexible cable, hose, tube, or other dark elongated object moving over a light planar background.

The application supports:

- video files and live webcams;
- automatic or manual image thresholding;
- binary-mask cleanup and connected-component filtering;
- skeleton-based centerline extraction;
- geodesic ordering of skeleton pixels;
- smoothed parametric spline fitting;
- signed curvature and radius-of-curvature estimation;
- planar metric calibration with four ArUco markers;
- perspective rectification to a top-down view;
- camera-pose visualization relative to the ArUco layout;
- an interactive dashboard with playback, seeking, sliders, plots, and snapshots.

> This is a proof-of-concept measurement tool. Validate the complete setup against independent physical references before treating the output as metrology-grade data.

## Main script

```text
curvatura_2d_aruco_dashboard_v2.py
```

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

## ArUco marker generation and calibration assets

The ArUco marker images and ChArUco utilities used with this project can be generated with the companion repository:

- [casfervi/aruco_charuco](https://github.com/casfervi/aruco_charuco)

For the default four-marker arrangement, generate or use marker IDs **0, 1, 2, and 3** from the same OpenCV dictionary configured in this application. The current default dictionary is:

```text
DICT_5X5_250
```

The companion repository includes utilities for:

- generating ArUco marker images;
- generating multiple marker IDs, including IDs 0 through 3;
- generating a ChArUco board;
- testing ArUco pose estimation;
- preparing intrinsic camera calibration assets.

The dictionary used to generate the markers must be identical to the dictionary supplied to this application. For example, marker ID 2 from `DICT_5X5_250` is not the same binary pattern as marker ID 2 from `DICT_4X4_50`.

## Default physical marker layout

Place four markers on the same plane as the cable. The default ID ordering is:

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

## Physical measurement definitions

### `--aruco-size`

The physical side length, in millimeters, of the external black square of each ArUco marker.

Example:

```bash
--aruco-size 20
```

### `--aruco-spacing`

The physical **center-to-center** distance between neighboring markers, in millimeters.

Two values represent horizontal and vertical spacing:

```bash
--aruco-spacing 225 130
```

This means:

```text
225 mm horizontally between marker centers
130 mm vertically between marker centers
```

One value creates a square center layout:

```bash
--aruco-spacing 225
```

Do not use edge-to-edge spacing without converting it to center-to-center spacing.

## Installation

Create and activate a virtual environment, then install the dependencies.

### Windows

```cmd
python -m venv gimbal
gimbal\Scripts\activate
pip install opencv-contrib-python numpy scipy scikit-image matplotlib
```

### Linux or macOS

```bash
python3 -m venv gimbal
source gimbal/bin/activate
pip install opencv-contrib-python numpy scipy scikit-image matplotlib
```

### Verify OpenCV ArUco support

```bash
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'aruco')); print(hasattr(cv2.aruco, 'ArucoDetector'))"
```

The final two values should normally be:

```text
True
True
```

Avoid installing incompatible versions of `opencv-python` and `opencv-contrib-python` in the same environment because both packages provide the `cv2` module.

## Basic usage

### Process a video with ArUco calibration

```bash
python curvatura_2d_aruco_dashboard_v2.py \
  --video cable.mp4 \
  --aruco-size 20 \
  --aruco-spacing 225 130 \
  --aruco-dict DICT_5X5_250 \
  --aruco-ids 0 1 2 3
```

### Process a video with a custom physical ID arrangement

```bash
python curvatura_2d_aruco_dashboard_v2.py \
  --video cable.mp4 \
  --aruco-size 20 \
  --aruco-spacing 225 130 \
  --aruco-dict DICT_5X5_250 \
  --aruco-ids 2 3 1 0
```

### Use a webcam

```bash
python curvatura_2d_aruco_dashboard_v2.py \
  --camera 0 \
  --aruco-size 20 \
  --aruco-spacing 225 130
```

### Process without ArUco calibration

Results are reported in pixels unless a fixed scale is supplied:

```bash
python curvatura_2d_aruco_dashboard_v2.py --video cable.mp4
```

### Use a fixed scale without perspective correction

```bash
python curvatura_2d_aruco_dashboard_v2.py \
  --video cable.mp4 \
  --mm-per-pixel 0.25
```

A fixed `mm/pixel` scale is appropriate only when perspective is sufficiently controlled and the object remains in the calibrated plane.

## Calibration behavior

The application detects all four expected markers in consecutive frames. It then:

1. takes the median detected corners across the requested calibration frames;
2. estimates the native plane resolution from the apparent marker dimensions unless `--calibration-px-per-mm` is supplied;
3. calculates a four-center homography that is robust to individual marker rotation;
4. uses all 16 marker corners when they are geometrically consistent;
5. creates a top-down rectified image;
6. stores the metric scale as millimeters per pixel;
7. saves the calibration to JSON.

### Number of calibration frames

```bash
--calibration-frames 5
```

Using more consecutive frames reduces subpixel detection noise but requires all four markers to remain visible during the full collection interval.

### Rectified resolution

By default, the program estimates a native `pixels/mm` value from the marker size in the source image.

To force a specific rectified resolution:

```bash
--calibration-px-per-mm 2
```

This creates a nominal output scale of:

```text
2 pixels/mm = 0.5 mm/pixel
```

A larger `pixels/mm` value creates a larger rectified image but does not create new physical detail beyond the source image resolution.

### Save and load calibration

The default calibration file is:

```text
calibration_aruco.json
```

Load an existing calibration:

```bash
python curvatura_2d_aruco_dashboard_v2.py \
  --video cable.mp4 \
  --load-calibration \
  --calibration-file calibration_aruco.json
```

Reuse a calibration only when the following remain unchanged:

- camera position and orientation;
- video resolution;
- lens, zoom, and focus;
- measurement plane;
- physical marker positions.

Press `C` during execution to discard the active planar calibration and collect a new one.

## Camera pose panel

When the four ArUco markers are visible, the application estimates and displays the camera pose relative to the marker plane.

The pose panel includes:

- ArUco-plane coordinate axes;
- camera coordinate axes;
- a schematic camera frustum;
- camera distance from the plane origin;
- camera height relative to the plane;
- camera X and Y displacement;
- approximate camera inclination;
- pose reprojection error.

If the real focal length in pixels is known, provide:

```bash
--focal-px 1500
```

Without `--focal-px`, the application assumes a focal length equal to approximately `0.8 x image width`. This is a rough pose-estimation aid, not a replacement for intrinsic camera calibration.

For more reliable camera pose, calibrate the camera intrinsically with a ChArUco board from the companion repository and extend the application to load the resulting camera matrix and distortion coefficients.

## Dashboard

The default dashboard includes:

```text
+---------------------------+---------------+-------------------+
| RESULT                    | INFORMATION   | CAMERA POSE       |
+---------------------------+---------------+-------------------+
| ORIGINAL / RECTIFIED      | MASK          | CONTROLS          |
+---------------------------+---------------+-------------------+
| CURVATURE                 | RADIUS OF CURVATURE              |
+---------------------------+-----------------------------------+
```

### Playback controls

The dashboard provides:

- `-1s`: move backward approximately one second;
- `<`: previous frame;
- `PLAY/PAUSE`: toggle playback;
- `>`: next frame;
- `+1s`: move forward approximately one second;
- progress bar: click or drag to seek through a video.

Playback timing attempts to respect the source video FPS while accounting for processing time.

### Segmentation controls

The dashboard includes sliders for:

- threshold;
- spline smoothing.

Threshold value `0` enables automatic Otsu thresholding.

Increasing spline smoothing may reduce curvature noise but can flatten real bends. Start with a low value and increase only when necessary.

## Keyboard and mouse controls

```text
Left click       Selects the nearest centerline point and displays R
Right click      Clears the selected point
C                Rebuilds the ArUco calibration
Space            Play or pause
S                Saves a snapshot, CSV, and graph image
Q or Esc         Exits
```

## Segmentation assumptions

The segmentation function assumes:

```text
dark object over a light background
```

The processing stages are:

1. BGR-to-grayscale conversion;
2. Gaussian blur;
3. inverted binary threshold;
4. morphological opening;
5. morphological closing;
6. connected-component analysis;
7. selection of the largest valid component.

The largest dark connected component is assumed to be the cable. Hands, shadows, dark supports, or other large objects can therefore cause incorrect segmentation.

For best results:

- use a matte light background;
- use a dark cable;
- keep hands out of the measurement region;
- use uniform lighting;
- avoid strong shadows and reflections;
- keep the cable in the marker plane.

## Centerline and curvature

The binary mask is reduced to a one-pixel skeleton. The application builds an eight-neighbor graph over the skeleton and uses two Dijkstra searches to estimate the longest geodesic path, which suppresses short skeleton branches.

A parametric spline is fitted to the ordered points:

```text
x(t), y(t)
```

The signed curvature is calculated as:

```text
k = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)
```

The radius of curvature is:

```text
R = 1 / |k|
```

The curve endpoints are trimmed from summary statistics because skeleton and spline derivatives are less stable near the ends.

## Output and snapshots

Press `S` to save a snapshot in the output directory.

The output can include:

- result image with centerline;
- CSV containing arc length, X, Y, curvature, and radius;
- PNG plots for curvature and radius.

Example CSV columns:

```text
s_mm
x_mm
y_mm
curvature_1/mm
radius_mm
```

When no metric calibration is active, the units are pixels and inverse pixels.

## Important measurement limitations

### Planar assumption

The ArUco homography assumes that:

- all four markers are on one plane;
- the cable remains approximately in that same plane.

A cable segment moving toward or away from the camera cannot be measured correctly with a single planar homography.

### Resolution is not accuracy

A rectified scale such as `2 px/mm` means that the output image is sampled at that nominal resolution. It does not guarantee `0.5 mm` measurement accuracy.

Actual error also depends on:

- marker dimension accuracy;
- center-to-center spacing accuracy;
- print quality;
- image focus and resolution;
- lens distortion;
- segmentation quality;
- spline smoothing;
- cable thickness;
- calibration-plane stability.

### RMS error

When calibration uses exactly four marker centers, the reprojection error at those same four points can be nearly zero by construction. Validate the calibration with independent dimensions that were not used to calculate the homography.

### Recommended validation

Place an independent physical reference in the measurement plane. For example, if a known 100 mm distance is present and the rectified scale is 2 px/mm, that reference should span approximately 200 pixels.

Calculate:

```text
absolute error = |measured - reference|
relative error = absolute error / reference x 100%
```

Repeat the validation in multiple parts of the rectified image.

## Command-line reference

Display all arguments:

```bash
python curvatura_2d_aruco_dashboard_v2.py --help
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
--calibration-px-per-mm VALUE
--focal-px VALUE
--calibration-frames COUNT
--calibration-file PATH
--load-calibration
--display dashboard|separate
--dashboard-width PIXELS
--dashboard-height PIXELS
```

## Companion ArUco and ChArUco repository

Use [casfervi/aruco_charuco](https://github.com/casfervi/aruco_charuco) to generate the ArUco markers and ChArUco calibration assets used by this project.

For the default layout, generate marker IDs:

```text
0, 1, 2, 3
```

using:

```text
DICT_5X5_250
```

unless another dictionary is explicitly selected in both projects.

Example marker-generation workflow in the companion repository:

```bash
python aruco_generation.py \
  --num-markers 4 \
  --start-id 0 \
  --dictionary DICT_5X5_250 \
  --output-dir markers
```

The companion repository can also generate a ChArUco board for intrinsic camera calibration.

## References

- [casfervi/aruco_charuco](https://github.com/casfervi/aruco_charuco)
- [OpenCV: Detection of ArUco Markers](https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html)
- [OpenCV: Detection of ArUco Boards](https://docs.opencv.org/4.13.0/db/da9/tutorial_aruco_board_detection.html)
- [OpenCV: Detection of ChArUco Boards](https://docs.opencv.org/4.13.0/df/d4a/tutorial_charuco_detection.html)
- [OpenCV: Calibration with ArUco and ChArUco](https://docs.opencv.org/4.13.0/da/d13/tutorial_aruco_calibration.html)
- [OpenCV: Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

## Suggested next improvements

- load a ChArUco-derived intrinsic camera matrix and distortion coefficients;
- undistort each frame before planar ArUco calibration;
- add independent calibration validation points;
- log frame-by-frame curvature statistics;
- add temporal filtering for the centerline and curvature;
- support color or neural-network segmentation for uncontrolled backgrounds;
- add stereo or depth sensing for true 3D cable curvature.
