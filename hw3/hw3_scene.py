import json
import numpy as np
import buffer
import math
import os
import slangpy as spy
import trimesh

SPHERE = 0  # Must match Slang's ShapeType in hw3_common.slang
TRIANGLE_MESH = 1

def parse_triangle_mesh(filename):
    if filename.lower().endswith('.ply'):
        mesh = trimesh.load_mesh(filename, process=False)

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        vertex_normals = np.asarray(
            mesh.vertex_normals, dtype=np.float32)

    else:
        with open(filename) as f:
            mesh = json.load(f)

        vertices = np.array(mesh['vertices'], dtype=np.float32)
        faces = np.array(mesh['faces'], dtype=np.uint32)

        if 'vertex_normals' in mesh:
            vertex_normals = np.array(
                mesh['vertex_normals'], dtype=np.float32)
        else:
            # Generate smooth vertex normals.
            tri_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=False)
            vertex_normals = np.asarray(
                tri_mesh.vertex_normals, dtype=np.float32)

    return vertices, faces, vertex_normals

def compose_transformations(transformations):
    """
        Given a list of transformations, compose them into a matrix.
    """
    # HW 3.4: copy paste your HW2.3 code here

    F = np.eye(4, dtype=np.float32)

    for transform in transformations:

        # HW 2.3 TODO: your code here

        (name,value), = transform.items()
        A = np.eye(4, dtype=np.float32)

        if name == 'scale':
            sx, sy, sz = value
            A = np.array([
                [sx, 0, 0, 0],
                [0, sy, 0, 0],
                [0, 0, sz, 0],
                [0, 0, 0, 1]
            ])

        elif name == 'translate':
            tx, ty, tz = value
            A = np.array([
                [1, 0, 0, tx],
                [0, 1, 0, ty],
                [0, 0, 1, tz],
                [0, 0, 0, 1]
            ])

        elif name == 'rotate_x':
            sin_t = np.sin(np.radians(value[0]))
            cos_t = np.cos(np.radians(value[0]))

            A = np.array([
                [1, 0, 0, 0],
                [0, cos_t, -1 * sin_t, 0],
                [0, sin_t, cos_t, 0],
                [0, 0, 0, 1]
            ])

        elif name == 'rotate_y':
            sin_t = np.sin(np.radians(value[0]))
            cos_t = np.cos(np.radians(value[0]))

            A = np.array([
                [cos_t, 0, sin_t, 0],
                [0, 1, 0, 0],
                [-1 * sin_t, 0, cos_t, 0],
                [0, 0, 0, 1]
            ])

        elif name == 'rotate_z':
            sin_t = np.sin(np.radians(value[0]))
            cos_t = np.cos(np.radians(value[0]))

            A = np.array([
                [cos_t, -1 * sin_t, 0, 0],
                [sin_t, cos_t, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        elif name == 'lookat':
            p = np.array(value['position'])
            t = np.array(value['target'])
            u = np.array(value['up'])

            d = (t-p) / np.linalg.norm(t-p)
            dxu = np.cross(d,u)
            r = dxu / np.linalg.norm(dxu)
            u_p = np.cross(r,d)

            A = np.array([
                [r[0], u_p[0], -1 * d[0], p[0]],
                [r[1], u_p[1], -1 * d[1], p[1]],
                [r[2], u_p[2], -1 * d[2], p[2]],
                [0, 0, 0, 1]
            ])
        
        F = A @ F

    return F

def interpolate_transformation(transform_keyframes, t):
    # HW 3.4: copy your HW2.4 code here.

    # create a new transform from the two keyframes
    interp_transforms = []
    i = 0
    max_time = transform_keyframes[-1]["time"]
    t_in = t%max_time
    while i < len(transform_keyframes):
        if t_in >= transform_keyframes[i]["time"] and t_in <= transform_keyframes[i+1]["time"]:
            break
        i += 1

    kf_0 = transform_keyframes[i]["transform"]
    kf_1 = transform_keyframes[i+1]["transform"]
    t_0 = transform_keyframes[i]["time"]
    t_1 = transform_keyframes[i+1]["time"]
    w = (t_in - t_0) / (t_1 - t_0)

    x_0 = []
    x_1 = []
    for i in range(len(kf_0)):
        (n0,v0), = kf_0[i].items()
        x_0.append([n0,v0])
        (n1,v1), = kf_1[i].items()
        x_1.append([n1,v1])

    x_t = []

    for i in range(len(x_0)):

        if x_0[i][0] == 'lookat':
            x_t.append(
                {
                    x_0[i][0]: {
                        "position": np.array(x_0[i][1]["position"])*(1-w) + np.array(x_1[i][1]["position"])*w,
                        "target": np.array(x_0[i][1]["target"])*(1-w) + np.array(x_1[i][1]["target"])*w,
                        "up": np.array(x_0[i][1]["up"])*(1-w) + np.array(x_1[i][1]["up"])*w
                    }
                }
            )
        else:
            x_t.append( { x_0[i][0]: [(x_0[i][1][j]*(1-w) + x_1[i][1][j]*w) for j in range(len(x_0[i][1]))] })
    interp_transforms = x_t

    return interp_transforms

def get_transformation(obj, t):
    transforms = obj.get('transform', [])
    if 'transform_keyframes' in obj:
        transforms = interpolate_transformation(
            obj['transform_keyframes'], t)
    return compose_transformations(transforms)

def build_bvh(vertices, faces, triangle_offset, bvh_nodes):
    # A recursive BVH builder that split the longest axis
    # using the median of the centroids. The leaf nodes
    # have maximum of 4 triangles.

    assert(len(faces) > 0)
    # Needs a copy because we will sort them during the recursion
    faces = faces.copy()
    centers = vertices[faces].mean(axis=1)

    def build(begin, end):
        node_id = len(bvh_nodes)
        # Important: append a node first
        # before the recursion, so that the order
        # is "preorder" 
        bvh_nodes.append(None)

        # Bounding box of all triangles in this node
        points = vertices[faces[begin:end]].reshape(-1, 3)
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)

        # Stop recursion if the node has fewer than 4 triangles
        is_leaf = end - begin <= 4
        if not is_leaf:
            # Find the longest axis
            c = centers[begin:end]
            axis = np.argmax(c.max(axis=0) - c.min(axis=0))

            # Sort triangles by their centroids
            order = np.argsort(c[:, axis])
            faces[begin:end] = faces[begin:end][order]
            centers[begin:end] = c[order]

            # Split
            mid = (begin + end) // 2
            build(begin, mid)
            build(mid, end)

        bvh_nodes[node_id] = {
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
            'next_index': len(bvh_nodes),
            'triangle_begin': triangle_offset + begin,
            'triangle_end': triangle_offset + end,
            'is_leaf': is_leaf
        }

        return node_id

    root = build(0, len(faces))
    return root, faces

def upload_scene(filename, module, slang_device, use_bvh = False):
    with open(filename) as f:
        scene = json.load(f)

    background = scene.get('background', (0.5, 0.5, 0.5))

    camera = {'cam_to_world': np.eye(4, dtype=np.float32),
              'fov_x': 90.0}

    if 'camera' in scene:
        camera['cam_to_world'] = \
            get_transformation(scene['camera'], 0.0)
        camera['fov_x'] = scene['camera'].get('fov_x', 90.0)

    # Triangle meshes loading + BVH building
    mesh_descs = []
    vertices_list = []
    normals_list = []
    triangles_list = []
    bvh_nodes = []

    vertex_offset = 0
    triangle_offset = 0

    scene_dir = os.path.dirname(filename)

    for mesh_data in scene.get('meshes', []):
        mesh_filename = os.path.join(
            scene_dir, mesh_data['filename'])
        vertices, faces, vertex_normals = \
            parse_triangle_mesh(mesh_filename)

        # Construct the BVH and reorder triangles
        # The returned faces is a copy of our
        # original faces.
        if use_bvh:
            bvh_root_index, faces = build_bvh(
                vertices, faces, triangle_offset, bvh_nodes)
        else:
            bvh_root_index = 0

        mesh_descs.append({
            'vertex_offset': vertex_offset,
            'triangle_begin': triangle_offset,
            'triangle_end': triangle_offset + len(faces),
            'bvh_root_index': bvh_root_index
        })

        vertices_list.append(vertices)
        normals_list.append(vertex_normals)
        triangles_list.append(faces)

        vertex_offset += len(vertices)
        triangle_offset += len(faces)

    # Merge into global arrays
    if len(mesh_descs) > 0:
        vertices = np.concatenate(vertices_list).astype(np.float32)
        normals = np.concatenate(normals_list).astype(np.float32)
        triangles = np.concatenate(triangles_list).astype(np.uint32)
    else:
        # Use dummy entries if the scene has no meshes
        vertices = np.zeros((1, 3), dtype=np.float32)
        normals = np.zeros((1, 3), dtype=np.float32)
        triangles = np.zeros((1, 3), dtype=np.uint32)

    # Parse object instances.
    spheres = []
    shapes = []

    for obj in scene['objects']:
        if obj['type'] == 'sphere':
            shape_type = SPHERE
            shape_index = len(spheres)
            spheres.append({
                'center': obj['center'],
                'radius': obj['radius']
            })
        elif obj['type'] == 'mesh':
            shape_type = TRIANGLE_MESH
            shape_index = obj['index']
        else:
            raise ValueError(f"Unknown shape type: {obj['type']}")

        obj_to_world = get_transformation(obj, 0.0)
        world_to_obj = np.linalg.inv(obj_to_world)

        shapes.append({
            'type': shape_type,
            'index': shape_index,
            'Kd': obj['Kd'],
            'Kr': obj['Kr'],
            'world_to_obj': world_to_obj,
            'obj_to_world': obj_to_world
        })

    lights = scene['lights']

    max_depth = scene.get('max_depth', 20)

    bindings = {
        'background': spy.float3(background),
        'camera': camera,
        'shapes': buffer.create_structured_buffer(
            slang_device, module.Shape, shapes),
        'num_shapes': len(shapes),
        'spheres': buffer.create_structured_buffer(
            slang_device, module.Sphere, spheres),
        'meshes': buffer.create_structured_buffer(
            slang_device, module.TriangleMesh, mesh_descs),
        'vertices': buffer.structured_typed_buffer_from_numpy(
            slang_device, vertices, 'float3'),
        'vertex_normals': buffer.structured_typed_buffer_from_numpy(
            slang_device, normals, 'float3'),
        'triangles': buffer.structured_typed_buffer_from_numpy(
            slang_device, triangles, 'uint3'),
        'bvh_nodes': buffer.create_structured_buffer(
            slang_device, module.BVHNode, bvh_nodes),
        'lights': buffer.create_structured_buffer(
            slang_device, module.PointLight, lights),
        'num_lights': len(lights),
        'max_depth': max_depth
    }

    return bindings, scene, shapes
