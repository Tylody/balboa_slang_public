import hw3_common;

void check_sphere_closer(
    int i,
    Sphere s,
    float3 d,
    out int bestShape,
    out float bestDist,
    out float bestT
) {
    float2 roots;
    float aEq = dot(d,d);
    float bEq = -2 * dot(d,s.center);
    float cEq = dot(s.center,s.center) - pow(s.radius,2);
    if(!solve_quadratic(aEq,bEq,cEq,roots)) return;

    // if intersects sphere twice, choose closer intersection s.t. t>0

    if(roots.x > 0) {
        float3 i1 = roots.x * d;
        float dist = distance(i1,float3(0,0,0));
        if(bestShape == -1 || dist < bestDist) {
            bestDist = dist;
            bestShape = i;
            bestT = roots.x;
        }
    }
    if(roots.y > 0) {
        float3 i1 = roots.y * d;
        float dist = distance(i1,float3(0,0,0));
        if(bestShape == -1 || dist < bestDist) {
            bestDist = dist;
            bestShape = i;
            bestT = roots.y;
        }
    }
    return;
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

    int bestShape = -1;
    float bestDist = 0;
    float bestT = 0;

    for (int i = 0; i < num_shapes; i++) {
        if(shapes[i].type == ShapeType.Sphere) {
            check_sphere_closer(i,spheres[shapes[i].index],d,bestShape,bestDist,bestT);
        }
    }

    if(bestShape == -1) {
        image[tid] = float4(background,1.0);
        return;
    }

    Sphere S = spheres[int(bestShape)];
    float3 intersection = bestSphere.z * d;
    float3 normal = normalize(intersection - S.center);

    float3 color = (normal+1) / 2;
    image[tid] = float4(color,1);

    // color = normal, let normal be n, visualize (n+1)/2
}
