# Canonical Motion Data

This directory contains the 100 canonical MotionFix motion files used by the
release motion benchmark. They are already in PoseShield's public motion
format and can be used directly by the demo, optimization, and evaluation
tools.

```text
shape: [frames, 135]

[0:132]   22 joints × 6D rotations, HY-Motion column-interleaved layout,
          ordered as [root, body0, ..., body20]
[132:135] absolute global translation [x, y_up, z_forward]
```

Coordinate convention:

```text
Y-up
X = right
Y = height/up
Z = forward
frame 0 human facing +Z
```

Small demo samples in `demo_asset/` use the same format.
