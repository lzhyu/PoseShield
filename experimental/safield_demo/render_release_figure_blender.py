"""Blender renderer for the SAField experimental release figure.

Run from the repository root with:

    blender --background --python experimental/safield_demo/render_release_figure_blender.py -- \
        --obj-dir experimental/safield_demo/artifacts/release_example/release_objs

The script imports four pre-exported OBJ meshes:
Shape A/B input and Shape A/B resolved, places them in one Blender scene, and
renders a PNG suitable for README/code-release use.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import math
import sys

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[2]
OBJ_DIR = ROOT / "experimental/safield_demo/artifacts/release_example/release_objs"
OUT_PATH = ROOT / "assets/safield_experimental_shape_demo_blender_full.png"


def parse_args() -> argparse.Namespace:
    """Parse arguments passed after Blender's `--` separator."""
    script_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj-dir", type=Path, default=OBJ_DIR)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--resolution-x", type=int, default=2600)
    parser.add_argument("--resolution-y", type=int, default=1300)
    return parser.parse_args(script_args)


def clear_scene() -> None:
    """Remove the default Blender scene objects."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.62,
) -> bpy.types.Material:
    """Create a simple principled material."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = 0.0
    return material


def import_obj(path: Path, name: str, material: bpy.types.Material, location: tuple[float, float, float]) -> bpy.types.Object:
    """Import one OBJ mesh, assign material, normalize origin, and place it."""
    before = set(bpy.data.objects)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.wm.obj_import(filepath=str(path))
    new_meshes = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
    if not new_meshes:
        raise RuntimeError(f"No mesh object was imported from {path}")
    if len(new_meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for mesh in new_meshes:
            mesh.select_set(True)
        bpy.context.view_layer.objects.active = new_meshes[0]
        bpy.ops.object.join()
        obj = bpy.context.object
    else:
        obj = new_meshes[0]
    obj.name = name
    obj.data.name = f"{name}_mesh"
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = 0
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.shade_smooth()

    center = sum((Vector(corner) for corner in obj.bound_box), Vector()) / 8.0
    obj.location = Vector(location) - center
    obj.rotation_euler[2] = math.radians(-8.0)
    return obj


def add_text(text: str, location: tuple[float, float, float], size: float = 0.2) -> bpy.types.Object:
    """Add a flat label above a panel."""
    bpy.ops.object.text_add(location=location, rotation=(math.radians(90), 0, 0))
    obj = bpy.context.object
    obj.name = f"label_{text.replace(' ', '_').lower()}"
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = size
    material = make_material(f"mat_{obj.name}", (0.08, 0.09, 0.10, 1.0), roughness=0.8)
    obj.data.materials.append(material)
    return obj


def setup_camera(objects: list[bpy.types.Object]) -> None:
    """Add orthographic camera looking at all panels."""
    bpy.ops.object.light_add(type="AREA", location=(0.0, -4.8, 5.8))
    light = bpy.context.object
    light.name = "large_softbox"
    light.data.energy = 580.0
    light.data.size = 5.4

    bpy.ops.object.light_add(type="POINT", location=(-4.2, -3.2, 2.5))
    rim = bpy.context.object
    rim.name = "warm_rim_light"
    rim.data.energy = 80.0

    bpy.ops.object.camera_add(location=(0.0, -7.4, -0.12), rotation=(math.radians(90.0), 0, 0))
    camera = bpy.context.object
    camera.name = "release_camera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 7.55
    bpy.context.scene.camera = camera


def setup_render(output_path: Path, resolution_x: int, resolution_y: int) -> None:
    """Configure Blender render settings."""
    scene = bpy.context.scene
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 96
    scene.cycles.use_denoising = True
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("release_world") if scene.world is None else scene.world
    scene.world.color = (0.96, 0.97, 0.985)
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.96, 0.97, 0.985, 1.0)
        background.inputs["Strength"].default_value = 0.82
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_path)


def main() -> None:
    """Build and render the SAField release figure."""
    args = parse_args()
    clear_scene()
    setup_render(args.output, args.resolution_x, args.resolution_y)
    blue = make_material("shape_a_blue", (0.03, 0.28, 0.80, 1.0))
    orange = make_material("shape_b_orange", (0.92, 0.48, 0.16, 1.0))
    objects = [
        import_obj(args.obj_dir / "shape_a_input.obj", "shape_a_input", blue, (-2.7, 0.0, 0.0)),
        import_obj(args.obj_dir / "shape_b_input.obj", "shape_b_input", orange, (-1.0, 0.0, 0.0)),
        import_obj(args.obj_dir / "shape_a_resolved.obj", "shape_a_resolved", blue, (1.0, 0.0, 0.0)),
        import_obj(args.obj_dir / "shape_b_resolved.obj", "shape_b_resolved", orange, (2.7, 0.0, 0.0)),
    ]
    labels = [
        add_text("Shape A input", (-2.7, -0.55, 1.04)),
        add_text("Shape B input", (-1.0, -0.55, 1.04)),
        add_text("Shape A resolved", (1.0, -0.55, 1.04)),
        add_text("Shape B resolved", (2.7, -0.55, 1.04)),
    ]
    setup_camera(objects + labels)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.with_suffix(".blend")))
    bpy.ops.render.render(write_still=True)
    print(f"Rendered {args.output}")


if __name__ == "__main__":
    main()
