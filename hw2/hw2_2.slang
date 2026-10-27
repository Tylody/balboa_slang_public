float signed_area(float2 a, float2 b, float2 c) {
    float2 u=b-a;
    float2 v=c-a;

    return (u.x*v.y - u.y*v.x) / 2;
}


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
    float2 p = float2(ipixel) + .5;


    // barycentric coordinate coefficients
    float A = signed_area(sp0, sp1, sp2);
    float b0p = signed_area(p, sp1, sp2)/A;
    float b1p = signed_area(sp0, p, sp2)/A;
    float b2p = signed_area(sp0, sp1, p)/A;

    if(b0p < 0 || b1p < 0 || b2p < 0) {
        return;
    }


    float w0=b0p*inv_depth0;
    float w1=b1p*inv_depth1;
    float w2=b2p*inv_depth2;
    float sum=w0+w1+w2;
    float b0=w0/sum;
    float b1=w1/sum;
    float b2=w2/sum;
    float d=1.0/sum;

    if(d<depth_image[ipixel]) {
        depth_image[ipixel]=d;
        color_image[ipixel]=float4(b0*c0+b1*c1+b2*c2, 1.0);
    }

    //color_image[ipixel] = float4(float3(0.0), 1.0);
}
