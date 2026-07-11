"""Blender script to render SMPL motion as an animated video.

Renders each timestep as a separate frame with ref and opt bodies side-by-side.
Aesthetic reference: LIGHT (ziyinwang1.github.io/LIGHT) and InterMask papers.

Output: a directory of PNG frames that can be stitched into MP4 via ffmpeg.
"""

import argparse
import sys
import os
import math
import numpy as np

import bpy
import mathutils


def parse_args():
    """Parse script arguments passed after '--'."""
    if "--" in sys.argv:
        args_start = sys.argv.index("--") + 1
        script_args = sys.argv[args_start:]
    else:
        script_args = []

    parser = argparse.ArgumentParser(description="Blender SMPL Motion Video Renderer")
    parser.add_argument("--mesh-path", type=str, required=True, help="Path to temp_meshes.npz")
    parser.add_argument("--output-dir", type=str, default="frames", help="Directory to write frame PNGs")
    parser.add_argument("--engine", type=str, default="BLENDER_EEVEE")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--color-a", type=float, nargs=4, default=[0.85, 0.18, 0.18, 1.0])
    parser.add_argument("--color-b", type=float, nargs=4, default=[0.25, 0.75, 0.32, 1.0])
    parser.add_argument("--metallic", type=float, default=0.0)
    parser.add_argument("--roughness", type=float, default=0.85)
    parser.add_argument("--camera-yaw", type=float, default=0.0)
    parser.add_argument("--camera-pitch", type=float, default=12.0)
    parser.add_argument("--light-energy", type=float, default=1.0)
    parser.add_argument("--contact-offset", type=float, default=0.022)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--res-x", type=int, default=1920)
    parser.add_argument("--res-y", type=int, default=1080)
    return parser.parse_args(script_args)


def clean_scene():
    """Clear default objects and lights."""
    if bpy.context.active_object and bpy.context.active_object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in [bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras]:
        for item in block:
            if item.users == 0:
                block.remove(item)


def create_clean_material(name, base_color, metallic=0.0, roughness=0.85):
    """Clean Principled BSDF — matching LIGHT/InterMask paper style (matte diffuse)."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)

    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Base Color'].default_value = base_color
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Roughness'].default_value = roughness
    if 'Specular' in bsdf.inputs:
        bsdf.inputs['Specular'].default_value = 0.0
    elif 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = 0.0
    bsdf.inputs['Alpha'].default_value = 1.0

    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return mat


def create_contact_material(name):
    """High-visibility yellow material for exact contact surface overlays."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Base Color'].default_value = (1.0, 0.82, 0.02, 1.0)
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['Roughness'].default_value = 0.58
    if 'Specular' in bsdf.inputs:
        bsdf.inputs['Specular'].default_value = 0.1
    elif 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = 0.1
    if 'Emission Color' in bsdf.inputs:
        bsdf.inputs['Emission Color'].default_value = (1.0, 0.68, 0.0, 1.0)
    if 'Emission Strength' in bsdf.inputs:
        bsdf.inputs['Emission Strength'].default_value = 0.25

    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return mat


def resolve_render_engine(scene, requested_engine):
    """Return a Blender render-engine identifier compatible with this version."""
    available_engines = {
        item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items
    }
    if requested_engine in available_engines:
        return requested_engine
    if requested_engine == "BLENDER_EEVEE" and "BLENDER_EEVEE_NEXT" in available_engines:
        return "BLENDER_EEVEE_NEXT"
    raise ValueError(
        f"Render engine {requested_engine!r} is not available; "
        f"available engines: {sorted(available_engines)}"
    )


def set_display_color_management(scene):
    """Use standard display colors so red/green meshes stay readable in GIFs."""
    try:
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception as exc:
        print(f"Could not set color management: {exc}")


def create_mesh_object(name, vertices, faces, material, location_offset):
    """Construct a Blender mesh object from vertex and face data."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.validate()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location_offset
    obj.data.materials.append(material)

    # Fix normals
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Smooth shading + auto-smooth
    for poly in obj.data.polygons:
        poly.use_smooth = True
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
    if hasattr(mesh, "auto_smooth_angle"):
        mesh.auto_smooth_angle = 1.0472  # 60 degrees

    return obj


def create_contact_mesh_object(name, vertices, faces, face_mask, material, location_offset, offset=0.014):
    """Create a raised yellow surface patch for contact faces in the current frame."""
    selected_faces = faces[face_mask]
    if len(selected_faces) == 0:
        return None

    used_vertices = np.unique(selected_faces.reshape(-1))
    remap = {int(old): idx for idx, old in enumerate(used_vertices)}
    compact_faces = np.vectorize(remap.__getitem__)(selected_faces).astype(np.int32)
    compact_vertices = vertices[used_vertices].copy()

    normals = np.zeros_like(vertices, dtype=np.float64)
    tri = vertices[selected_faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.maximum(lengths, 1e-8)
    for corner in range(3):
        np.add.at(normals, selected_faces[:, corner], face_normals)
    compact_normals = normals[used_vertices]
    compact_lengths = np.linalg.norm(compact_normals, axis=1, keepdims=True)
    compact_normals = compact_normals / np.maximum(compact_lengths, 1e-8)
    compact_vertices = compact_vertices + compact_normals * offset

    obj = create_mesh_object(name, compact_vertices, compact_faces, material, location_offset)
    return obj


def setup_world_lighting():
    """Clean studio-style world background."""
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output = nodes.new(type='ShaderNodeOutputWorld')
    background = nodes.new(type='ShaderNodeBackground')
    background.inputs['Color'].default_value = (0.72, 0.72, 0.75, 1.0)
    background.inputs['Strength'].default_value = 0.6
    links.new(background.outputs['Background'], output.inputs['Surface'])


def setup_lights(center, body_height, multiplier=1.0):
    """Three-point lighting rig with area lights and EEVEE contact shadows."""
    dist = body_height * 2.5

    # Key light
    key = bpy.data.lights.new("Key", type='AREA')
    key.energy = 800.0 * multiplier
    key.size = body_height * 0.5
    key.color = (1.0, 0.98, 0.95)
    if hasattr(key, "use_contact_shadow"):
        key.use_contact_shadow = True
        key.contact_shadow_distance = 0.15
    key_obj = bpy.data.objects.new("Key", key)
    key_obj.location = center + mathutils.Vector((dist * 0.8, -dist * 0.5, dist * 1.5))
    bpy.context.collection.objects.link(key_obj)
    d = center - key_obj.location
    key_obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()

    # Fill light
    fill = bpy.data.lights.new("Fill", type='AREA')
    fill.energy = 50.0 * multiplier
    fill.size = body_height * 2.5
    fill.color = (0.92, 0.95, 1.0)
    fill_obj = bpy.data.objects.new("Fill", fill)
    fill_obj.location = center + mathutils.Vector((-dist, -dist * 0.4, dist * 0.6))
    bpy.context.collection.objects.link(fill_obj)
    d = center - fill_obj.location
    fill_obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()

    # Rim light
    rim = bpy.data.lights.new("Rim", type='AREA')
    rim.energy = 200.0 * multiplier
    rim.size = body_height * 1.2
    rim_obj = bpy.data.objects.new("Rim", rim)
    rim_obj.location = center + mathutils.Vector((0, dist, dist * 1.2))
    bpy.context.collection.objects.link(rim_obj)
    d = center - rim_obj.location
    rim_obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def main():
    args = parse_args()

    print("Cleaning scene...")
    clean_scene()

    # Load mesh data
    data = np.load(args.mesh_path)
    verts_ref_raw = data["verts_a"]  # [L, 6890, 3]
    verts_opt_raw = data["verts_b"]  # [L, 6890, 3]
    faces = data["faces"]
    # Optional yellow overlays are precomputed exact-FCL face masks. They must
    # come from the same original motion, SMPL-H topology, and TOPO threshold as
    # the mesh data in this render package.
    contact_masks = data["contact_masks"] if "contact_masks" in data.files else np.zeros((len(verts_ref_raw), len(faces)), dtype=bool)

    # Y-up → Z-up
    verts_ref = np.stack([verts_ref_raw[..., 0], -verts_ref_raw[..., 2], verts_ref_raw[..., 1]], axis=-1)
    verts_opt = np.stack([verts_opt_raw[..., 0], -verts_opt_raw[..., 2], verts_opt_raw[..., 1]], axis=-1)

    L = len(verts_ref)
    print(f"Loaded mesh data: {L} frames")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Materials
    mat_ref = create_clean_material("MatRef", args.color_a, args.metallic, args.roughness)
    mat_opt = create_clean_material("MatOpt", args.color_b, args.metallic, args.roughness)
    mat_contact = create_contact_material("MatContact")

    # Compute global bounding box across ALL frames for consistent camera
    all_verts = np.concatenate([verts_ref, verts_opt], axis=0)  # [2L, 6890, 3]
    global_min = all_verts.min(axis=(0, 1))
    global_max = all_verts.max(axis=(0, 1))
    body_height = global_max[2] - global_min[2]

    # Side-by-side offset (ref left, opt right)
    x_sep = 1.2  # separation between the two bodies

    # Center point between the two bodies (averaged across all frames)
    global_center_x = (global_min[0] + global_max[0]) / 2.0
    global_center_y = (global_min[1] + global_max[1]) / 2.0
    global_center_z = (global_min[2] + global_max[2]) / 2.0
    center = mathutils.Vector((global_center_x, global_center_y, global_center_z))

    # Floor plane
    floor_size = body_height * 5.0
    bpy.ops.mesh.primitive_plane_add(size=floor_size, location=(center.x, center.y, global_min[2]))
    floor = bpy.context.active_object
    floor.name = "Floor"

    floor_mat = bpy.data.materials.new("FloorMat")
    floor_mat.use_nodes = True
    fn = floor_mat.node_tree.nodes
    fl = floor_mat.node_tree.links
    fn.clear()
    fo = fn.new(type='ShaderNodeOutputMaterial')
    fb = fn.new(type='ShaderNodeBsdfPrincipled')
    fb.inputs['Base Color'].default_value = (0.50, 0.50, 0.52, 1.0)
    fb.inputs['Roughness'].default_value = 0.85
    fl.new(fb.outputs['BSDF'], fo.inputs['Surface'])
    floor.data.materials.append(floor_mat)

    # World + Lights
    setup_world_lighting()
    setup_lights(center, body_height, args.light_energy)

    # Camera
    cam_data = bpy.data.cameras.new("Cam")
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    cam_dist = body_height * 3.3
    yaw = math.radians(args.camera_yaw)
    pitch = math.radians(args.camera_pitch)
    cx = cam_dist * math.cos(pitch) * math.sin(yaw)
    cy = -cam_dist * math.cos(pitch) * math.cos(yaw)
    cz = cam_dist * math.sin(pitch)
    cam_obj.location = center + mathutils.Vector((cx, cy, cz))

    target = bpy.data.objects.new("CamTarget", None)
    target.location = center
    bpy.context.collection.objects.link(target)
    tt = cam_obj.constraints.new(type='TRACK_TO')
    tt.target = target
    tt.track_axis = 'TRACK_NEGATIVE_Z'
    tt.up_axis = 'UP_Y'

    # Render settings
    scene = bpy.context.scene
    render_engine = resolve_render_engine(scene, args.engine)
    scene.render.engine = render_engine
    scene.render.resolution_x = args.res_x
    scene.render.resolution_y = args.res_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = False

    if render_engine == 'CYCLES':
        scene.cycles.samples = args.samples
        scene.cycles.device = 'GPU'
    else:
        if hasattr(scene, "eevee"):
            if hasattr(scene.eevee, "taa_render_samples"):
                scene.eevee.taa_render_samples = args.samples
            if hasattr(scene.eevee, "use_soft_shadows"):
                scene.eevee.use_soft_shadows = True

    # Render each frame
    print(f"Rendering {L} frames to {args.output_dir}/...")
    ref_obj = None
    opt_obj = None
    contact_obj = None

    for frame_idx in range(L):
        # Delete previous frame's meshes
        if ref_obj is not None:
            bpy.data.objects.remove(ref_obj, do_unlink=True)
        if opt_obj is not None:
            bpy.data.objects.remove(opt_obj, do_unlink=True)
        if contact_obj is not None:
            bpy.data.objects.remove(contact_obj, do_unlink=True)

        # Create meshes for this frame
        ref_offset = mathutils.Vector((-x_sep / 2.0, 0.0, 0.0))
        opt_offset = mathutils.Vector((x_sep / 2.0, 0.0, 0.0))

        ref_obj = create_mesh_object(
            f"Ref_{frame_idx}", verts_ref[frame_idx], faces, mat_ref, ref_offset
        )
        opt_obj = create_mesh_object(
            f"Opt_{frame_idx}", verts_opt[frame_idx], faces, mat_opt, opt_offset
        )
        contact_obj = create_contact_mesh_object(
            f"Contact_{frame_idx}",
            verts_ref[frame_idx],
            faces,
            contact_masks[frame_idx],
            mat_contact,
            ref_offset,
            offset=args.contact_offset,
        )

        # Render frame
        frame_path = os.path.join(os.path.abspath(args.output_dir), f"frame_{frame_idx:04d}.png")
        scene.render.filepath = frame_path
        bpy.ops.render.render(write_still=True)

        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx}/{L} done")

    print(f"All {L} frames rendered to {args.output_dir}/")


if __name__ == "__main__":
    main()
