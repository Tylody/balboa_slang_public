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

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Sphere> spheres,
                  uniform int num_spheres,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    //image[tid] = float4(background, 1.0);    
    float aspect= float(width) / float(height);
    float2 pixelc= float2(tid.x + 0.5, height - tid.y - 0.5);
    float3 origin = float3(0, 0, 0);
    float3 dir = float3(2 * aspect * (pixelc.x / width) - aspect, (pixelc.y / height) * 2 - 1, -1);

    float closest_t=1e30;
    int hit_id=-1;
    

    for (int i=0; i < num_spheres; i++) {
        float t;
        if (intersect(spheres[i], origin, dir, 0.0, closest_t, t)) {
            closest_t = t;
            hit_id = i;
        }

    }

    if(hit_id==-1) {
        image[tid] = float4(background, 1.0);
        return;
    }
    float3 psphere = origin + closest_t * dir;
    float3 n = normalize(psphere - spheres[hit_id].center);
    image[tid] = float4((n + 1) / 2, 1.0);

}
