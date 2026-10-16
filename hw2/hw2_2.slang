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

    // heron's formula:
    // s = (len(a)+len(b)+len(c)) / 2, area = sqrt(s(s-len(a))(s-len(b))(s-len(c)))

    float2 p = float2(ipixel) + 0.5;

    float sm[3] = { distance(sp0,sp1), distance(sp1,sp2), distance(sp2,sp0) };
    float Sm = (sm[0] + sm[1] + sm[2]) / 2;

    float sm_0[3] = { distance(p,sp1), distance(sp1,sp2), distance(sp2,p) };
    float Sm_0 = (sm_0[0] + sm_0[1] + sm_0[2]) / 2;

    float sm_1[3] = { distance(sp0,p), distance(p,sp2), distance(sp2,sp0) };
    float Sm_1 = (sm_1[0] + sm_1[1] + sm_1[2]) / 2;

    float sm_2[3] = { distance(sp0,sp1), distance(sp1,p), distance(p,sp0) };
    float Sm_2 = (sm_2[0] + sm_2[1] + sm_2[2]) / 2;

    float area_p0p1p2 = sqrt( Sm * (Sm - sm[0]) * (Sm - sm[1]) * (Sm - sm[2]) );

    float b0p = sqrt( Sm_0 * (Sm_0 - sm_0[0]) * (Sm_0 - sm_0[1]) * (Sm_0 - sm_0[2]) ) / area_p0p1p2;
    float b1p = sqrt( Sm_1 * (Sm_1 - sm_1[0]) * (Sm_1 - sm_1[1]) * (Sm_1 - sm_1[2]) ) / area_p0p1p2;
    float b2p = sqrt( Sm_2 * (Sm_2 - sm_2[0]) * (Sm_2 - sm_2[1]) * (Sm_2 - sm_2[2]) ) / area_p0p1p2;

    float3 bi_proj = float3(b0p / inv_depth0, b1p / inv_depth1, b2p / inv_depth2);
    float bi_denom = bi_proj.x + bi_proj.y + bi_proj.z;

    float3 bi = bi_proj / bi_denom;
    float pz = bi.x * inv_depth0 + bi.y * inv_depth1 + bi.z * inv_depth2;

    float3 c = bi.x * c0 + bi.y * c1 + bi.z * c2;

    if(pz < depth_image[ipixel]) {
        color_image[ipixel] = float4(c, 1.0);
        depth_image[ipixel] = pz;
    }

}
