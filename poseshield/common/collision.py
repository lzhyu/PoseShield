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
from pathlib import Path

# Load default topology threshold from global config
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_global_config_path = os.path.join(_project_root, "config_files", "global_path.yaml")
DEFAULT_TOPOLOGY_THRESHOLD = 50
DEFAULT_MOTION_TOPOLOGY_THRESHOLD = 40
if os.path.exists(_global_config_path):
    try:
        with open(_global_config_path, "r") as _f:
            _config = yaml.safe_load(_f)
            if _config and "TOPOLOGY_THRESHOLD" in _config:
                DEFAULT_TOPOLOGY_THRESHOLD = int(_config["TOPOLOGY_THRESHOLD"])
            if _config and "MOTION_TOPOLOGY_THRESHOLD" in _config:
                DEFAULT_MOTION_TOPOLOGY_THRESHOLD = int(_config["MOTION_TOPOLOGY_THRESHOLD"])
    except Exception:
        pass

class BandedTopologyDistances:
    """Compact face-topology distances for a bounded threshold range.

    Distances below ``min_distance`` are stored with one shared near marker,
    distances in ``[min_distance, max_distance]`` are stored exactly, and
    missing pairs are treated as farther than ``max_distance``. This supports
    variable thresholds inside the stored range, such as 40 for motion and 50
    for pose, without keeping the full Python tuple-key dictionary in memory.
    """

    format_name = "banded_topology_distances_v1"

    def __init__(self, indptr, indices, distances, min_distance=30, max_distance=60, fast_bits=None):
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.distances = np.asarray(distances, dtype=np.uint8)
        self.min_distance = int(min_distance)
        self.max_distance = int(max_distance)
        self.near_value = self.min_distance - 1
        self.num_faces = int(len(self.indptr) - 1)
        self.fast_bits = {int(k): np.asarray(v, dtype=np.uint8) for k, v in (fast_bits or {}).items()}

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False, mmap_mode="r")
        format_value = str(data["format"].item()) if "format" in data.files else ""
        if format_value != cls.format_name:
            raise ValueError(f"Unsupported topology cache format in {path}: {format_value}")
        fast_bits = {}
        if "fast_thresholds" in data.files:
            for threshold in data["fast_thresholds"]:
                key = f"near_bits_{int(threshold)}"
                if key in data.files:
                    fast_bits[int(threshold)] = data[key]
        return cls(
            data["indptr"],
            data["indices"],
            data["distances"],
            int(data["min_distance"].item()),
            int(data["max_distance"].item()),
            fast_bits=fast_bits,
        )

    @classmethod
    def from_distance_dict(cls, distance_dict, min_distance=30, max_distance=60, num_faces=None):
        min_distance = int(min_distance)
        max_distance = int(max_distance)
        if min_distance < 1:
            raise ValueError("min_distance must be positive")
        if max_distance < min_distance:
            raise ValueError("max_distance must be >= min_distance")

        near_value = min_distance - 1

        def _stored_pair(item):
            (face1, face2), distance = item
            face1 = int(face1)
            face2 = int(face2)
            if face1 == face2:
                return None
            if face1 > face2:
                face1, face2 = face2, face1
            distance = int(distance)
            if distance < min_distance:
                stored_distance = near_value
            elif distance <= max_distance:
                stored_distance = distance
            else:
                return None
            return face1, face2, stored_distance

        max_face = -1 if num_faces is None else int(num_faces) - 1
        for item in distance_dict.items():
            stored = _stored_pair(item)
            if stored is not None:
                face1, face2, _ = stored
                max_face = max(max_face, face1, face2)
        num_faces = max_face + 1
        counts = np.zeros(num_faces, dtype=np.int64)
        num_entries = 0
        for item in distance_dict.items():
            stored = _stored_pair(item)
            if stored is not None:
                counts[stored[0]] += 1
                num_entries += 1

        indptr = np.empty(num_faces + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        indices = np.empty(num_entries, dtype=np.int32)
        distances = np.empty(num_entries, dtype=np.uint8)
        cursor = indptr[:-1].copy()
        for item in distance_dict.items():
            stored = _stored_pair(item)
            if stored is None:
                continue
            face1, face2, stored_distance = stored
            offset = cursor[face1]
            indices[offset] = face2
            distances[offset] = stored_distance
            cursor[face1] += 1

        for face in range(num_faces):
            start = indptr[face]
            end = indptr[face + 1]
            if end - start > 1:
                order = np.argsort(indices[start:end], kind="mergesort")
                indices[start:end] = indices[start:end][order]
                distances[start:end] = distances[start:end][order]

        return cls(indptr, indices, distances, min_distance, max_distance)

    def add_fast_thresholds(self, thresholds):
        for threshold in thresholds:
            threshold = self._check_threshold(threshold)
            bit_count = self.num_faces * self.num_faces
            bitset = np.zeros((bit_count + 7) // 8, dtype=np.uint8)
            for face in range(self.num_faces):
                start = int(self.indptr[face])
                end = int(self.indptr[face + 1])
                if start == end:
                    continue
                near_mask = self.distances[start:end] < threshold
                if not np.any(near_mask):
                    continue
                cols = self.indices[start:end][near_mask].astype(np.int64, copy=False)
                keys = face * self.num_faces + cols
                byte_indices = keys >> 3
                bit_masks = (1 << (keys & 7)).astype(np.uint8, copy=False)
                np.bitwise_or.at(bitset, byte_indices, bit_masks)
            self.fast_bits[threshold] = bitset

    def save(self, path):
        payload = {
            "format": np.asarray(self.format_name),
            "min_distance": np.asarray(self.min_distance, dtype=np.int16),
            "max_distance": np.asarray(self.max_distance, dtype=np.int16),
            "indptr": self.indptr,
            "indices": self.indices,
            "distances": self.distances,
        }
        if self.fast_bits:
            thresholds = np.asarray(sorted(self.fast_bits), dtype=np.int16)
            payload["fast_thresholds"] = thresholds
            for threshold in thresholds:
                payload[f"near_bits_{int(threshold)}"] = self.fast_bits[int(threshold)]
        np.savez_compressed(path, **payload)

    def _check_threshold(self, threshold):
        threshold = int(threshold)
        if threshold < self.min_distance or threshold > self.max_distance:
            raise ValueError(
                f"Topology threshold {threshold} is outside compact cache range "
                f"[{self.min_distance}, {self.max_distance}]. Regenerate the cache "
                "with a wider range or use the legacy deps/distances.pkl."
            )
        return threshold

    def topological_distance(self, face1, face2):
        face1 = int(face1)
        face2 = int(face2)
        if face1 == face2:
            return 0
        if face1 > face2:
            face1, face2 = face2, face1
        if face1 < 0 or face1 + 1 >= len(self.indptr):
            return -1
        start = int(self.indptr[face1])
        end = int(self.indptr[face1 + 1])
        row = self.indices[start:end]
        pos = int(np.searchsorted(row, face2))
        if pos < len(row) and int(row[pos]) == face2:
            return int(self.distances[start + pos])
        return -1

    def is_far(self, face1, face2, threshold):
        threshold = self._check_threshold(threshold)
        if face1 == face2:
            return False
        if face1 > face2:
            face1, face2 = face2, face1
        bitset = self.fast_bits.get(threshold)
        if bitset is not None and 0 <= face1 < self.num_faces and 0 <= face2 < self.num_faces:
            key = int(face1) * self.num_faces + int(face2)
            return (int(bitset[key >> 3]) & (1 << (key & 7))) == 0
        distance = self.topological_distance(face1, face2)
        return distance == -1 or distance >= threshold


def load_topology_distances(path):
    path = Path(path)
    if path.suffix == ".npz":
        return BandedTopologyDistances.load(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def topological_distance(distances, face1: int, face2: int) -> int:
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
    if face1 == face2:
        return 0
    if face1 > face2:
        face1, face2 = face2, face1
    if hasattr(distances, "topological_distance"):
        return distances.topological_distance(face1, face2)
    distance = distances.get((face1, face2))
    return -1 if distance is None else int(distance)


def is_topologically_far(distances, face1: int, face2: int, threshold: int) -> bool:
    if hasattr(distances, "is_far"):
        return bool(distances.is_far(face1, face2, threshold))
    distance = topological_distance(distances, face1, face2)
    return distance == -1 or distance >= threshold

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
        # Only count the collision if the faces are not topologically neighboring.
        if is_topologically_far(distances, i, j, topology_threshold):
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
        # Only count if topological distance is sufficient (not neighbors)
        if is_topologically_far(distances, i, j, topology_threshold) and penetration > penetration_threshold:
            problematic_collision_count += 1
            total_penetration_depth += penetration

    end = time.time()
    # For reporting, you may set your own collision count threshold.
    has_problematic_collision = problematic_collision_count >= 1

    return has_problematic_collision, total_penetration_depth


def self_collision_contact_faces(
    vertices,
    faces,
    distances,
    topology_threshold=None,
    penetration_threshold=1e-6,
    max_contacts=1_000_000,
):
    """Return problematic self-collision face ids and summed penetration depth."""
    if topology_threshold is None:
        topology_threshold = DEFAULT_TOPOLOGY_THRESHOLD

    collision_objects, collision_geoms = _build_fcl_collision_objects(vertices, faces)
    geom_to_index = {id(geom): idx for idx, geom in enumerate(collision_geoms)}
    manager = fcl.DynamicAABBTreeCollisionManager()
    manager.registerObjects(collision_objects)
    manager.setup()

    contact_faces = set()
    total_penetration_depth = 0.0

    crequest = fcl.CollisionRequest(num_max_contacts=max_contacts, enable_contact=True)
    cresult = fcl.CollisionResult()
    cdata = fcl.CollisionData(crequest, cresult)
    manager.collide(cdata, fcl.defaultCollisionCallback)

    for contact in cdata.result.contacts:
        i = geom_to_index.get(id(contact.o1), None)
        j = geom_to_index.get(id(contact.o2), None)
        if i is None or j is None:
            continue
        if is_topologically_far(distances, i, j, topology_threshold) and contact.penetration_depth > penetration_threshold:
            contact_faces.add(int(i))
            contact_faces.add(int(j))
            total_penetration_depth += float(contact.penetration_depth)

    collision_objects.clear()
    collision_geoms.clear()
    manager = None
    cdata = None
    crequest = None
    gc.collect()
    return sorted(contact_faces), total_penetration_depth



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
    distances = load_topology_distances(distance_path)
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
