"""FCL-based self-collision detection for SMPL meshes."""
import torch
import os
import numpy as np
import fcl
import pickle
import time
import yaml
import gc
import smplx

# Load default topology threshold from global config
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_global_config_path = os.path.join(_project_root, "config_files", "global_path.yaml")
DEFAULT_TOPOLOGY_THRESHOLD = 50
if os.path.exists(_global_config_path):
    try:
        with open(_global_config_path, "r") as _f:
            _config = yaml.safe_load(_f)
            if _config and "TOPOLOGY_THRESHOLD" in _config:
                DEFAULT_TOPOLOGY_THRESHOLD = int(_config["TOPOLOGY_THRESHOLD"])
    except Exception:
        pass

def topological_distance(distances: dict, face1: int, face2: int) -> int:
    """
    Retrieve the precomputed topological distance between two faces from the given dictionary.
    
    Parameters:
        distances (dict): A dictionary with keys as tuples (face_i, face_j)
                          and values representing the topological distance between those faces.
        face1 (int): Index of the first face.
        face2 (int): Index of the second face.
    
    Returns:
        int: The topological distance between face1 and face2.
             Returns -1 if the distance is not found.
    """
    # Attempt to fetch the distance using the key (face1, face2)
    if face1 > face2:
        face1, face2 = face2, face1
    distance = distances.get((face1, face2))
    return distance

def _build_fcl_collision_objects(vertices, faces):
    """
    Build FCL collision objects for each triangle in the mesh.
    """
    collision_objects = []
    collision_geoms = []
    for face in faces:
        triangle_vertices = vertices[face]
        model = fcl.BVHModel()
        model.beginModel()
        model.addSubModel(triangle_vertices, np.array([[0, 1, 2]], dtype=np.int32))
        model.endModel()
        collision_geoms.append(model)
        obj = fcl.CollisionObject(model)
        collision_objects.append(obj)
    return collision_objects, collision_geoms

def is_mesh_self_intersecting(vertices, faces, distances, topology_threshold=None, max_tolerable_pairs=1): # 1
    if topology_threshold is None:
        topology_threshold = DEFAULT_TOPOLOGY_THRESHOLD
    """
    Check if a mesh has self-intersections using python-fcl, ignoring collisions
    between triangles that are topologically close (neighbors).

    Parameters:
        vertices (np.ndarray): Nx3 array of vertex coordinates.
        faces (np.ndarray): Mx3 array of triangle indices.
        distances (dict): Precomputed topological distances dictionary (using pickle).
        topology_threshold (int): Minimum topological distance for a collision to be considered
                                  an error (collisions with a distance <= threshold are ignored).

    Returns:
        bool: True if the mesh has problematic self-intersections; False otherwise.
    """
    # Create a collision object for each triangle.
    assert topology_threshold >= 10, f"Topology threshold must be greater than 10, but got {topology_threshold}"
    start=time.time()
    collision_objects, collision_geoms = _build_fcl_collision_objects(vertices, faces)
    
    # Use FCL's collision manager for efficient broad-phase collision checking.
    geom_to_index = {id(geom): idx for idx, geom in enumerate(collision_geoms)}
    manager = fcl.DynamicAABBTreeCollisionManager()
    manager.registerObjects(collision_objects)
    manager.setup()

    # List to store candidate collision pairs (indices of the triangles).
    candidate_pairs = []
    crequest = fcl.CollisionRequest(num_max_contacts=1e6, enable_contact=False)
    cdata = fcl.CollisionData(crequest, fcl.CollisionResult())

    # manager.collide(cdata, collision_callback)
    manager.collide(cdata, fcl.defaultCollisionCallback)
    for contact in cdata.result.contacts:
        # Extract collision geometries that are in contact
        coll_geom_0 = contact.o1
        coll_geom_1 = contact.o2
        i = geom_to_index.get(id(coll_geom_0), None)
        j = geom_to_index.get(id(coll_geom_1), None)
        candidate_pairs.append((i, j))

    problematic_collision_count = 0
    for i, j in candidate_pairs:
        topo_dist = topological_distance(distances, i, j)
        # Only count the collision if the faces are not topologically neighboring.
        if topo_dist >= topology_threshold or topo_dist == -1:
            problematic_collision_count += 1

    # For example, you can decide that more than 10 problematic collisions indicate self-intersection.
    # Release large objects and trigger garbage collection

    candidate_pairs.clear()
    collision_objects.clear()
    collision_geoms.clear()
    manager = None
    cdata = None
    crequest = None
    collision_objects = None
    collision_geoms = None
    candidate_pairs = None
    gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    return problematic_collision_count > max_tolerable_pairs



def self_collision_status(vertices, faces, distances, topology_threshold=None, penetration_threshold=1e-6):
    if topology_threshold is None:
        topology_threshold = DEFAULT_TOPOLOGY_THRESHOLD
    """
    Check if a mesh has self-intersections using python-fcl, and report total penetration depth.

    Returns:
        has_problematic_collision: True/False
        total_penetration_depth: float, summed penetration depth of all detected problematic collisions
    """
    start = time.time()
    collision_objects, collision_geoms = _build_fcl_collision_objects(vertices, faces)

    geom_to_index = {id(geom): idx for idx, geom in enumerate(collision_geoms)}
    manager = fcl.DynamicAABBTreeCollisionManager()
    manager.registerObjects(collision_objects)
    manager.setup()

    crequest = fcl.CollisionRequest(num_max_contacts=1_000_000, enable_contact=True)
    cresult = fcl.CollisionResult()
    cdata = fcl.CollisionData(crequest, cresult)

    # Collect contacts using a custom callback to collect all collision pairs.
    manager.collide(cdata, fcl.defaultCollisionCallback)

    candidate_pairs = []
    penetration_depths = []

    # Collect all contact pairs and their penetration depth.
    for contact in cdata.result.contacts:
        coll_geom_0 = contact.o1
        coll_geom_1 = contact.o2
        i = geom_to_index.get(id(coll_geom_0), None)
        j = geom_to_index.get(id(coll_geom_1), None)
        if i is None or j is None:
            continue
        # Save penetration depth for each pair
        candidate_pairs.append((i, j, contact.penetration_depth))

    # Problematic collision statistics
    problematic_collision_count = 0
    total_penetration_depth = 0.0

    # Filter by topology
    for i, j, penetration in candidate_pairs:
        topo_dist = topological_distance(distances, i, j)
        # Only count if topological distance is sufficient (not neighbors)
        if (topo_dist >= topology_threshold or topo_dist == -1) and penetration > penetration_threshold:
            problematic_collision_count += 1
            total_penetration_depth += penetration

    end = time.time()
    # For reporting, you may set your own collision count threshold.
    has_problematic_collision = problematic_collision_count >= 1

    return has_problematic_collision, total_penetration_depth



# unit test 
if __name__ == '__main__':
    from poseshield.common.utils import sample_6d, sixd_to_mesh, quick_viz_6d
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    global_config_path = os.path.join(_project_root, "config_files", "global_path.yaml")
    with open(global_config_path, "r") as f:
        global_config = yaml.safe_load(f)

    model_path = global_config["BODY_MODEL_PATH"]
    distance_path = global_config["MESH_DISTANCE_PATH"]

    device = torch.device('cuda')
    smpl_model = smplx.create(
        model_path,
        model_type='smplh',
        gender='neutral',
        ext='npz',
        use_pca=False
    ).to(device)
    distances = pickle.load(open(distance_path, "rb"))
    times = []
    num_iters = 10
    for i in range(num_iters):
        random_sample = sample_6d(num_joints=21)
        vertices, faces = sixd_to_mesh(smpl_model, random_sample, device=device)

        t0 = time.time()
        collision_exists = is_mesh_self_intersecting(vertices, faces, distances)
        dt = time.time() - t0
        times.append(dt)

        print(f"[{i}] Self-intersecting (ignoring topologically close collisions)? {collision_exists} | detection time: {dt:.4f}s")

    if times:
        print(f"Average collision detection time over {len(times)} runs: {np.mean(times):.4f}s (std: {np.std(times):.4f}s)")
