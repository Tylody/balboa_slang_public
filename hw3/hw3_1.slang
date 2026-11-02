import hw3_common;

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

    float2 p = float2(tid.x + 0.5, height - tid.y - 0.5);
    float a = float(width) / height;
    float2 pScreen = float2( (2 * a * (p.x / width) - a), ((p.y / height) * 2 - 1));
    float3 d = float3(pScreen,-1);

    float3 bestSphere = float3(-1,-1,-1);

    for (int i = 0; i < num_spheres; i++) {

        Sphere s = spheres[i];

        // at each pixel, shoot a ray from origin to pixel
        // find intersection with sphere - ||o + td - xc|| ^2 = r^2

        float2 roots;
        float aEq = dot(d,d);
        float bEq = -2 * dot(d,s.center);
        float cEq = dot(s.center,s.center) - pow(s.radius,2);
        if(!solve_quadratic(aEq,bEq,cEq,roots)) continue;

        // if intersects sphere twice, choose closer intersection s.t. t>0

        if(roots.x > 0) {
            float3 i1 = roots.x * d;
            float dist = distance(i1,float3(0,0,0));
            if(bestSphere.x == -1 || dist < bestSphere.y) {
                bestSphere.y = dist;
                bestSphere.x = i;
                bestSphere.z = roots.x;
            }
        }

        if(roots.y > 0) {
            float3 i1 = roots.y * d;
            float dist = distance(i1,float3(0,0,0));
            if(bestSphere.x == -1 || dist < bestSphere.y) {
                bestSphere.y = dist;
                bestSphere.x = i;
                bestSphere.z = roots.y;
            }
        }
    }

    if(bestSphere.x == -1) {
        image[tid] = float4(background,1.0);
        return;
    }

    Sphere S = spheres[int(bestSphere.x)];
    float3 intersection = bestSphere.z * d;
    float3 normal = normalize(intersection - S.center);

    float3 color = (normal+1) / 2;
    image[tid] = float4(color,1);

    // color = normal, let normal be n, visualize (n+1)/2
}
