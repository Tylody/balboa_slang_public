[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0,
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform int2 aabb_min,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    // TODO: HW 2.1

    // points already in screen space
    // so test the signs from dot product

    float2 p = float2(ipixel) + 0.5;
    
    float s0 = dot((p - sp0), float2(sp1.y - sp0.y, sp0.x - sp1.x));
    float s1 = dot((p - sp1), float2(sp2.y - sp1.y, sp1.x - sp2.x));
    float s2 = dot((p - sp2), float2(sp0.y - sp2.y, sp2.x - sp0.x));

    if (
        (s0 >= 0 && s1 >= 0 && s2 >= 0)
        || (s0 <= 0 && s1 <= 0 && s2 <= 0)
    ) {
        image[ipixel] = float4(float3(color),1);
    } 
}
