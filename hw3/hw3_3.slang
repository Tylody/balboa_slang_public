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

float3 ray_trace(
    float3 o,
    float3 d,
    inout int bestShape,
    inout float bestT,
    inout float3 normal,
    uniform float3 background,
    StructuredBuffer<Shape> shapes,
    uniform int num_shapes,
    StructuredBuffer<Sphere> spheres,
    StructuredBuffer<PointLight> lights,
    uniform int num_lights
) {

    for (int i = 0; i < num_shapes; i++) {
        if (shapes[i].type == ShapeType.Sphere) {
            check_sphere_closer(i, spheres[shapes[i].index], d, o, bestShape, bestT, T_MAX);
        }
    }

    if (bestShape == -1) { return background; }

    float3 color = float3(0);
    float3 intersection = o + bestT * d;
    Sphere S = spheres[shapes[int(bestShape)].index];
    normal = normalize(intersection - S.center);

    for (int i = 0; i < num_lights; i++) {
        int LbestShape = -1;
        float LbestT = 0;
        float3 Ld = lights[i].position - intersection;

        for (int j = 0; j < num_shapes; j++) {
            if (shapes[j].type == ShapeType.Sphere) {
                check_sphere_closer(j, spheres[shapes[j].index], Ld, intersection, LbestShape, LbestT, 1 - EPSILON);
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

    float a = float(width) / height;

    float2 p = float2(tid.x + 0.5, height - tid.y - 0.5);
    float2 pScreen = float2( (2 * a * (p.x / width) - a), ((p.y / height) * 2 - 1));
    float3 d = float3(pScreen,-1);

    int bestShape = -1;
    float bestT = 0;
    float3 normal = 0;

    float3 color = 0.0;
    float3 throughput = 1.0;

    float3 o = float3(0, 0, 0);

    for (int b = 0; b < max_depth; b++) {

        float3 contribution = ray_trace(o, d, bestShape, bestT, normal, background, shapes, num_shapes, spheres, lights, num_lights);

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
