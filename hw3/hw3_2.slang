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

    float a = float(width) / height;

    float2 p = float2(tid.x + 0.5, height - tid.y - 0.5);
    float2 pScreen = float2( (2 * a * (p.x / width) - a), ((p.y / height) * 2 - 1));
    float3 d = float3(pScreen,-1);

    int bestShape = -1;
    float bestT = 0;

    float3 origin = float3(0, 0, 0);
    for (int i = 0; i < num_shapes; i++) {
        if(shapes[i].type == ShapeType.Sphere) {
            check_sphere_closer(i,spheres[shapes[i].index],d,origin,bestShape,bestT,T_MAX);
        }
    }

    if(bestShape == -1) {
        image[tid] = float4(background,1.0);
        return;
    }

    // color = (I / r^2) * Kd * max(n*l,0)
    float3 color = float3(0, 0, 0);
    float3 intersection = bestT * d;
    Sphere S = spheres[shapes[int(bestShape)].index];
    float3 normal = normalize(intersection - S.center);

    for (int i = 0; i < num_lights; i++) {

        int LbestShape = -1;
        float LbestT = 0;
        float3 Ld = lights[i].position - intersection;

        for (int j = 0; j < num_shapes; j++) {
            if (shapes[j].type == ShapeType.Sphere) {
                check_sphere_closer(j,spheres[shapes[j].index],Ld,intersection,LbestShape,LbestT,1-EPSILON);
            }
        }
        if (LbestShape != -1) continue;

        float3 I = lights[i].intensity / pow(distance(intersection, lights[i].position), 2);
        float3 l = normalize(Ld);
        float3 reflectance = shapes[bestShape].Kd * max(dot(normal,l),0);
        color += I * reflectance;
    }
    image[tid] = float4(color,1);
}
