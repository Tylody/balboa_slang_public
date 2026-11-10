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

int closest_hit(StructuredBuffer<Shape> shapes, int num_shapes, StructuredBuffer<Sphere> spheres, float3 origin, float3 dir, float t_min, float t_max, out float t_hit) {
    t_hit = t_max;
    int hit_id = -1;
    for(int i=0; i < num_shapes; i++) {
        float t;
        if (intersect(spheres[shapes[i].index], origin, dir, t_min, t_hit, t)) {
            t_hit = t;
            hit_id = i;
        }
    }
    return hit_id;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
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
    float3 ray_origin = origin;
    float3 ray_dir = dir;
    
    for( int i = 0; i < max_depth; i++) {
        float t;
        int hit_id = closest_hit(shapes, num_shapes, spheres, ray_origin, ray_dir, EPSILON, 1e30, t);
        if(hit_id == -1 ) {
            color+= throughput * background;
            break;
        }

        float3 psphere = ray_origin + t * ray_dir;
        float3 normal = normalize(psphere - spheres[shapes[hit_id].index].center);
        float3 Kd = shapes[hit_id].Kd;
        float3 totalcolor=float3(0, 0, 0);

        for(int j=0; j<num_lights; j++) {
            float3 to_light = lights[j].position - psphere;
            float distance = length(to_light);
            float3 unitvector = to_light / distance;


            float t_shadow;
            int hit = closest_hit(shapes, num_shapes, spheres, psphere, unitvector, EPSILON, distance, t_shadow);
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
