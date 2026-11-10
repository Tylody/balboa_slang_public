import hw3_common;

bool intersect(Sphere s, float3 origin, float3 dir, float t_min, float t_max, out float t_hit) {
    t_hit = t_max;
    float3 oc = origin - s.center;
    float A = dot (dir, dir);
    float B = 2 * dot(dir, oc);
    float C = dot(oc, oc) - s.radius*s.radius;

    float2 roots;
    if(!solve_quadratic(A, B, C, roots)) {
        return false;
    }
    float t0= min(roots.x, roots.y);
    float t1= max(roots.x, roots.y);

    if (t0 > t_min && t0 < t_max) {
        t_hit = t0; //(front hit)
        return true;
    }

    if(t1 > t_min && t1 < t_max) {
        t_hit = t1; // camera is inside the sphere, back hit
        return true;
    }
    return false;
}

int closest_hit(StructuredBuffer<Shape> shapes, int num_shapes, StructuredBuffer<Sphere> spheres, StructuredBuffer<TriangleMesh> meshes, StructuredBuffer<float3> vertices, StructuredBuffer<uint3> triangles, float3 origin, float3 dir, float t_min, float t_max, out float t_hit, out uint tri_id, out float3 bary) {
    t_hit = t_max;
    tri_id = 0;
    bary = float3(0, 0, 0);
    int hit_id = -1;
    for(int i=0; i < num_shapes; i++) {
        float3 o_obj = xform_pt(shapes[i].world_to_obj, origin);
        float3 d_obj = xform_vec(shapes[i].world_to_obj, dir);

        if(shapes[i].type == ShapeType.Sphere) {
            float t;
            if (intersect(spheres[shapes[i].index], o_obj, d_obj, t_min, t_hit, t)) {
                t_hit = t;
                hit_id = i;
            }
        }
        else {
            TriangleMesh mesh = meshes[shapes[i].index];
            for( uint j = mesh.triangle_begin; j<mesh.triangle_end; j++) {
                uint3 tri = triangles[j] + mesh.vertex_offset;
                float3 p0 = vertices[tri.x];
                float3 p1 = vertices[tri.y];
                float3 p2 = vertices[tri.z];
                float3 b;
                float t;
                if(intersect_tri(p0, p1, p2, o_obj, d_obj, t_min, t_hit, b, t)) {
                    t_hit = t;
                    hit_id = i;
                    tri_id = j;
                    bary = b;
                }
            }
        }
        
        
    }
    return hit_id;
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
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  uniform int max_depth,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }
    float aspect= float(width) / float(height);
    float2 pixelc= float2(tid.x + 0.5, height - tid.y - 0.5);
    float3 origin = float3(0, 0, 0);
    float3 dir = float3(2 * aspect * (pixelc.x / width) - aspect, (pixelc.y / height) * 2 - 1, -1);
    dir = normalize(dir);
    float3 color = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);

    float half_w = tan(radians(camera.fov_x) / 2);
    float half_h = half_w / aspect;
    float3 dir_cam = float3(half_w * (2 * pixelc.x / width - 1 ), half_h * (2 * pixelc.y / height - 1), -1);
    
    float3 ray_origin = xform_pt(camera.cam_to_world, float3(0, 0, 0));
    float3 ray_dir = normalize(xform_vec(camera.cam_to_world, dir_cam));

    uint tri_unused;
    float3 b_unused;

    
    for( int i = 0; i < max_depth; i++) {
        float t;
        uint hit_tri;
        float3 hit_bary;
        int hit_id = closest_hit(shapes, num_shapes, spheres, meshes, vertices, triangles, ray_origin, ray_dir, EPSILON, 1e30, t, hit_tri, hit_bary);
        if(hit_id == -1 ) {
            color+= throughput * background;
            break;
        }

        float3 psphere = ray_origin + t * ray_dir;
        float3 n_obj;
        if(shapes[hit_id].type == ShapeType.Sphere) {
            float3 p_obj = xform_pt(shapes[hit_id].world_to_obj, psphere);
            n_obj = p_obj - spheres[shapes[hit_id].index].center;
        }
        else {
            TriangleMesh mesh = meshes[shapes[hit_id].index];
            uint3 tri = triangles[hit_tri] + mesh.vertex_offset;
            n_obj = hit_bary.x * vertex_normals[tri.x] + hit_bary.y * vertex_normals[tri.y] + hit_bary.z * vertex_normals[tri.z];
        }
        

        float3 normal = normalize(xform_vec(transpose(shapes[hit_id].world_to_obj), n_obj));
        float3 Kd = shapes[hit_id].Kd;
        float3 totalcolor=float3(0, 0, 0);

        for(int j=0; j<num_lights; j++) {
            float3 to_light = lights[j].position - psphere;
            float distance = length(to_light);
            float3 unitvector = to_light / distance;


            float t_shadow;
            int hit = closest_hit(shapes, num_shapes, spheres, meshes, vertices, triangles, psphere, unitvector, EPSILON, distance, t_shadow, tri_unused, b_unused);
            if(hit !=-1) {
                continue;
            }
            
            totalcolor+=Kd * max(dot(normal, unitvector), 0) * lights[j].intensity / (distance * distance);
            
        }
        color+= throughput * totalcolor;
            
        throughput*=shapes[hit_id].Kr;
        ray_origin = psphere;
        ray_dir = reflect(ray_dir, normal);

    }
    image[tid]= float4(color, 1.0);
}
