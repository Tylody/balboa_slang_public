[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0, // vertex at screen space
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform float inv_depth0, // inverse depth in camera space
                  uniform float inv_depth1,
                  uniform float inv_depth2,
                  uniform int2 aabb_min,
                  uniform float3 c0, // vertex color
                  uniform float3 c1,
                  uniform float3 c2,
                  uniform float s,
                  uniform float a,
                  RWTexture2D<float4> color_image,
                  RWTexture2D<float> depth_image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    color_image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    // TODO: HW 2.2

    float2 p = float2(ipixel) + 0.5;

    float2 p0p1 = sp1 - sp0;
    float2 p1p2 = sp2 - sp1;

    float2 pp0 = sp0 - p;
    float2 pp1 = sp1 - p;
    float2 pp2 = sp2 - p;

    float area_p0p1p2 = length(cross(float3(p0p1, 0), float3(p1p2, 0))) / 2;

    float b0p = (length(cross(float3(pp1, 0), float3(pp2, 0))) / 2) / area_p0p1p2;
    float b1p = (length(cross(float3(pp0, 0), float3(pp2, 0))) / 2) / area_p0p1p2;
    float b2p = (length(cross(float3(pp0, 0), float3(pp1, 0))) / 2) / area_p0p1p2;

    float3 bi_depth = float3(b0p * inv_depth0, b1p * inv_depth1, b2p * inv_depth2);
    float bi_denom = bi_depth.x + bi_depth.y + bi_depth.z;

    float3 bi = bi_depth / bi_denom;
    float pz = bi.x / inv_depth0 + bi.y / inv_depth1 + bi.z / inv_depth2;

    float3 c = bi.x * c0 + bi.y * c1 + bi.z * c2;

    if(pz < depth_image[ipixel]) {
        float s0 = dot((p - sp0), float2(sp1.y - sp0.y, sp0.x - sp1.x));
        float s1 = dot((p - sp1), float2(sp2.y - sp1.y, sp1.x - sp2.x));
        float s2 = dot((p - sp2), float2(sp0.y - sp2.y, sp2.x - sp0.x));

        if (
            (s0 >= 0 && s1 >= 0 && s2 >= 0)
            || (s0 <= 0 && s1 <= 0 && s2 <= 0)
        ) {
            color_image[ipixel] = float4(c, 1.0);
            depth_image[ipixel] = pz;
        }
    }

}
