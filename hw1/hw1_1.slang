[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 center,
                  uniform float radius,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }
    center.y = height - center.y;

    image[tid] = float4(1.0);
    // TODO: your code here
    float dx = (tid.x + 0.5) - center.x;
    float dy = (tid.y + 0.5) - center.y;

    float dist = sqrt((dx*dx) + (dy*dy));

    if (dist <= radius){
        image[tid] = float4(color,1);
    }
}
