import hw3_common;

static const float T_MAX = 1e30;

void check_sphere_closer(
    int i,
    Sphere s,
    float3 d,
    float3 o,
    inout int bestShape,
    inout float bestT,
    float tMax
) {
    float2 roots;
    float aEq = dot(d, d);
    float bEq = 2 * dot(d, o - s.center);
    float cEq = dot(o, o) - (2 * dot(o, s.center)) + dot(s.center, s.center) - pow(s.radius, 2);
    if(!solve_quadratic(aEq,bEq,cEq,roots)) return;

    // if intersects sphere twice, choose closer intersection s.t. t>EPSILON
    float bestRoot = (roots.x > EPSILON && (roots.x < roots.y && roots.y > EPSILON) || roots.y <= EPSILON) ? roots.x : roots.y;
    if(bestRoot > EPSILON && bestRoot <= tMax) {
        if(bestShape == -1 || bestRoot < bestT) {
            bestShape = i;
            bestT = bestRoot;
        }
    }
    return;
}

void check_triangle_closer(
    int i,
    TriangleMesh mesh,
    StructuredBuffer<float3> vertices,
    StructuredBuffer<uint3> triangles,
    StructuredBuffer<BVHNode> bvh_nodes,
    float3 d,
    float3 o,
    inout int bestShape,
    inout int bestTri,
    inout float bestT,
    inout float3 bestB,
    float tMax
) {
    float3 b;
    float t;

    BVHNode curr;
    int nodeInd = mesh.bvh_root_index;
    int end = bvh_nodes[mesh.bvh_root_index].next_index;

    while (nodeInd < end) {
        curr = bvh_nodes[nodeInd];

        bool intersect = intersect_aabb(curr.bbox_min, curr.bbox_max, o, d, EPSILON, tMax);

        if (curr.is_leaf && intersect) {
            for (int j = curr.triangle_begin; j < curr.triangle_end; j++) {
                if (!intersect_tri( vertices[triangles[j].x+mesh.vertex_offset], vertices[triangles[j].y+mesh.vertex_offset], vertices[triangles[j].z+mesh.vertex_offset], o, d, EPSILON, tMax, b, t )) continue;

                if (bestShape == -1 || t < bestT) {
                    bestShape = i;
                    bestTri = j;
                    bestT = t;
                    bestB = b;
                }
            }
        }

        nodeInd = intersect ? nodeInd + 1 : curr.next_index;
    }
}

float3 ray_trace(
    float3 o,
    float3 d,
    inout int bestShape,
    inout int bestTri,
    inout float bestT,
    inout float3 bestB,
    inout float3 normal,
    uniform float3 background,
    StructuredBuffer<Shape> shapes,
    uniform int num_shapes,
    StructuredBuffer<Sphere> spheres,
    StructuredBuffer<TriangleMesh> meshes,
    StructuredBuffer<float3> vertices,
    StructuredBuffer<float3> vertex_normals,
    StructuredBuffer<uint3> triangles,
    StructuredBuffer<BVHNode> bvh_nodes,
    StructuredBuffer<PointLight> lights,
    uniform int num_lights
) {

    for (int i = 0; i < num_shapes; i++) {
        if (shapes[i].type == ShapeType.Sphere) {
            check_sphere_closer(i, spheres[shapes[i].index], xform_vec(shapes[i].world_to_obj,d), xform_pt(shapes[i].world_to_obj,o), bestShape, bestT, T_MAX);
        }

        if (shapes[i].type == ShapeType.TriangleMesh) {
            check_triangle_closer(i, meshes[shapes[i].index], vertices, triangles, bvh_nodes, xform_vec(shapes[i].world_to_obj, d), xform_pt(shapes[i].world_to_obj, o), bestShape, bestTri, bestT, bestB, T_MAX);
        }
    }

    if (bestShape == -1) { return background; }

    float3 color = float3(0);
    float3 intersection = o + bestT * d;

    if (shapes[bestShape].type == ShapeType.Sphere) {
        Sphere S = spheres[shapes[int(bestShape)].index];
        normal = xform_pt(shapes[bestShape].world_to_obj, intersection) - S.center;
        normal = xform_vec(transpose(shapes[bestShape].world_to_obj), normal);
        normal = normalize(normal);
    }

    if (shapes[bestShape].type == ShapeType.TriangleMesh) {
        TriangleMesh mesh = meshes[shapes[bestShape].index];
        uint3 triangle = triangles[bestTri];
        float3 n1 = vertex_normals[triangle.x + mesh.vertex_offset];
        float3 n2 = vertex_normals[triangle.y + mesh.vertex_offset];
        float3 n3 = vertex_normals[triangle.z + mesh.vertex_offset];

        normal = bestB.x * n1 + bestB.y * n2 + bestB.z * n3;
        normal = xform_vec(transpose(shapes[bestShape].world_to_obj), normal);
        normal = normalize(normal);
    }

    for (int i = 0; i < num_lights; i++) {
        int LbestShape = -1;
        int LbestTri = -1;
        float LbestT = 0;
        float3 LbestB = float3(0);
        float3 Ld = lights[i].position - intersection;

        for (int j = 0; j < num_shapes; j++) {
            if (shapes[j].type == ShapeType.Sphere) {
                check_sphere_closer(j, spheres[shapes[j].index], xform_vec(shapes[j].world_to_obj,Ld), xform_pt(shapes[j].world_to_obj,intersection), LbestShape, LbestT, 1 - EPSILON);
            }

            if (shapes[j].type == ShapeType.TriangleMesh) {
                check_triangle_closer(j, meshes[shapes[j].index], vertices, triangles, bvh_nodes, xform_vec(shapes[j].world_to_obj, Ld), xform_pt(shapes[j].world_to_obj, intersection), LbestShape, LbestTri, LbestT, LbestB, 1 - EPSILON);
            }
        }

        if (LbestShape != -1) continue; // shape between light & shape
        float3 I = lights[i].intensity / pow(distance(intersection, lights[i].position), 2);
        float3 l = normalize(Ld);
        float3 reflectance = shapes[bestShape].Kd * max(dot(normal, l), 0);
        color += I * reflectance;
    }
    return color;

}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform Camera camera,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<TriangleMesh> meshes,
                  StructuredBuffer<float3> vertices,
                  StructuredBuffer<float3> vertex_normals,
                  StructuredBuffer<uint3> triangles,
                  StructuredBuffer<BVHNode> bvh_nodes,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  uniform int max_depth,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    float a = float(width) / height;
    float2 p = float2(tid.x + 0.5, height - tid.y - 0.5);
    float u = p.x / width;

    float2 pScreen = float2( 
        ((2 * u - 1) * tan(radians(camera.fov_x)/2) ), 
        (((p.y / height) * 2 - 1) * tan(radians(camera.fov_x)/2) / a )
    );

    float3 d = float3(pScreen,-1);

    d = xform_vec(camera.cam_to_world, d);

    int bestShape = -1;
    int bestTri = -1;
    float bestT = 0;
    float3 bestB = float3(0);

    float3 normal = 0;

    float3 color = 0.0;
    float3 throughput = 1.0;

    float3 o = xform_pt(camera.cam_to_world, float3(0));

    for (int b = 0; b < max_depth; b++) {

        float3 contribution = ray_trace(o, d, bestShape, bestTri, bestT, bestB, normal, background, shapes, num_shapes, spheres, meshes, vertices, vertex_normals, triangles, bvh_nodes, lights, num_lights);

        o += bestT * d;
        color += throughput * contribution;

        if (bestShape == -1) break;
        if (length(shapes[bestShape].Kr) == 0) break;

        d = reflect(d, normal);
        throughput *= shapes[bestShape].Kr;
        bestT = 0;
        bestShape = -1;
    }

    image[tid] = float4(color,1);
}
