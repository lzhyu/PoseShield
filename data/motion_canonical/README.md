# Canonical Motion Data

PoseShield motion code expects ready-to-use canonical motion files:

```text
shape: [frames, 135]

[0:132]   22 joints × 6D rotations, HY-Motion column-interleaved layout
[132:135] absolute global translation [abs_x, abs_y, abs_z]
```

Coordinate convention:

```text
Y-up
X = right
Y = height/up
Z = forward
frame 0 human facing +Z
```

Small demo samples are included in `demo_asset/`. The full canonical motion subset is distributed separately through the project release assets.
