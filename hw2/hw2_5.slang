
float3 interpolate_barycentric(
    int2 ipixel,
    uniform float2 sp0,
    uniform float2 sp1,
    uniform float2 sp2,
    uniform float inv_depth0,
    uniform float inv_depth1,
    uniform float inv_depth2
) {

    float2 p = float2(ipixel) + 0.5;

    float2 p0p1 = sp1 - sp0;
    float2 p1p2 = sp2 - sp1;

    float2 pp0 = sp0 - p;
    float2 pp1 = sp1 - p;
    float2 pp2 = sp2 - p;

    float area_p0p1p2 = length(cross(float3(p0p1, 0), float3(p1p2, 0))) / 2;

    if (area_p0p1p2 == 0) return float3(0, 0, 0);

    float b0p = (length(cross(float3(pp1, 0), float3(pp2, 0))) / 2) / area_p0p1p2;
    float b1p = (length(cross(float3(pp0, 0), float3(pp2, 0))) / 2) / area_p0p1p2;
    float b2p = (length(cross(float3(pp0, 0), float3(pp1, 0))) / 2) / area_p0p1p2;

    float3 bi_depth = float3(b0p * inv_depth0, b1p * inv_depth1, b2p * inv_depth2);
    float bi_denom = bi_depth.x + bi_depth.y + bi_depth.z;

    float3 bi = bi_depth / bi_denom;
    return bi;
}



[shader("compute")]
[numthreads(64, 1, 1)]
void clear_buffers(uint tid : SV_DispatchThreadID,
                   uniform uint num_pixels,
                   RWStructuredBuffer<uint> depth_buffer,
                   RWStructuredBuffer<uint> triangle_id_buffer) {
    if (tid >= num_pixels)
        return;

    depth_buffer[tid] = 0xffffffff;
    triangle_id_buffer[tid] = 0xffffffff;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void transform_vertices(
    uint tid : SV_DispatchThreadID,
    uniform uint num_vertices,
    uniform uint vertex_offset,
    uniform float4x4 MVP,
    StructuredBuffer<float3> vertices,
    RWStructuredBuffer<float3> projected_vertices)
{
    if (tid >= num_vertices)
        return;

    uint i = vertex_offset + tid;
    float4 p = mul(MVP, float4(vertices[i], 1.0));
    projected_vertices[i] = p.xyz / p.w;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_depth(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        // A projected vertex store the screen space coordinates
        // in xy, and the inverse depth in z.
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer) {
    // HW 2.5 TODO

    if (tid >= num_triangles) return;

    uint3 triangle = faces[tid];
    float2 sp0 = projected_vertices[triangle.x].xy;
    float inv_depth0 = projected_vertices[triangle.x].z;
    float2 sp1 = projected_vertices[triangle.y].xy;
    float inv_depth1 = projected_vertices[triangle.y].z;
    float2 sp2 = projected_vertices[triangle.z].xy;
    float inv_depth2 = projected_vertices[triangle.z].z;

    float2 aabb_min = float2(
        max(floor(min(min(sp0.x, sp1.x),sp2.x)), 0), 
        max(floor(min(min(sp0.y, sp1.y),sp2.y)),0)
    );

    float2 aabb_max = float2(
        min(ceil(max(max(sp0.x, sp1.x), sp2.x)), float(width) - 1),
        min(ceil(max(max(sp0.y, sp1.y), sp2.y)), float(height) - 1)
    );

    for (int i = (int) aabb_min.x; i <= (int) aabb_max.x; i++) {
        for (int j = (int) aabb_min.y; j <= (int) aabb_max.y; j++) {

            float2 p = float2(i,j) + 0.5;
            float s0 = dot((p - sp0), float2(sp1.y - sp0.y, sp0.x - sp1.x));
            float s1 = dot((p - sp1), float2(sp2.y - sp1.y, sp1.x - sp2.x));
            float s2 = dot((p - sp2), float2(sp0.y - sp2.y, sp2.x - sp0.x));

            if (
                !((s0 >= 0 && s1 >= 0 && s2 >= 0)
                || (s0 <= 0 && s1 <= 0 && s2 <= 0))
            ) continue;
                
            float3 bi = interpolate_barycentric(int2(i, j), sp0, sp1, sp2, inv_depth0, inv_depth1, inv_depth2);
            if (bi.x == 0 && bi.y == 0 && bi.z == 0) continue;

            float pz = bi.x / inv_depth0 + bi.y / inv_depth1 + bi.z / inv_depth2;
            if (!(pz > 0)) continue;

            uint depth_bits = asuint(pz);
            InterlockedMin(depth_buffer[j*width+i], depth_bits);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_id(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer,
        RWStructuredBuffer<uint> triangle_id_buffer) {
    // HW 2.5 TODO

    if (tid >= num_triangles) return;

    uint3 triangle = faces[tid];
    float2 sp0 = projected_vertices[triangle.x].xy;
    float inv_depth0 = projected_vertices[triangle.x].z;
    float2 sp1 = projected_vertices[triangle.y].xy;
    float inv_depth1 = projected_vertices[triangle.y].z;
    float2 sp2 = projected_vertices[triangle.z].xy;
    float inv_depth2 = projected_vertices[triangle.z].z;

    float2 aabb_min = float2(
        max(floor(min(min(sp0.x, sp1.x),sp2.x)), 0), 
        max(floor(min(min(sp0.y, sp1.y),sp2.y)),0)
    );

    float2 aabb_max = float2(
        min(ceil(max(max(sp0.x, sp1.x), sp2.x)), float(width) - 1),
        min(ceil(max(max(sp0.y, sp1.y), sp2.y)), float(height) - 1)
    );

    for (int i = (int) aabb_min.x; i <= (int) aabb_max.x; i++) {
        for (int j = (int) aabb_min.y; j <= (int) aabb_max.y; j++) {

            float2 p = float2(i,j) + 0.5;
            float s0 = dot((p - sp0), float2(sp1.y - sp0.y, sp0.x - sp1.x));
            float s1 = dot((p - sp1), float2(sp2.y - sp1.y, sp1.x - sp2.x));
            float s2 = dot((p - sp2), float2(sp0.y - sp2.y, sp2.x - sp0.x));

            if (
                !((s0 >= 0 && s1 >= 0 && s2 >= 0)
                || (s0 <= 0 && s1 <= 0 && s2 <= 0))
            ) continue;
                
            float3 bi = interpolate_barycentric(int2(i, j), sp0, sp1, sp2, inv_depth0, inv_depth1, inv_depth2);
            if (bi.x == 0 && bi.y == 0 && bi.z == 0) continue;

            float pz = bi.x / inv_depth0 + bi.y / inv_depth1 + bi.z / inv_depth2;
            if (!(pz > 0)) continue;

            uint depth_bits = asuint(pz);
            if (depth_bits == depth_buffer[j * width + i]) {
                InterlockedMin(triangle_id_buffer[j * width + i], tid);
            }

        }
    }


}

[shader("compute")]
[numthreads(8, 8, 1)]
void shade(
        uint2 tid : SV_DispatchThreadID,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> vertex_colors,
        StructuredBuffer<float3> projected_vertices,
        StructuredBuffer<uint> triangle_id_buffer,
        RWTexture2D<float4> image) {
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // HW 2.5 TODO
    uint tri_id = triangle_id_buffer[tid.y * width + tid.x];
    if (tri_id == 0xffffffff) return;

    uint3 triangle = faces[tri_id];
    float2 sp0 = projected_vertices[triangle.x].xy;
    float inv_depth0 = projected_vertices[triangle.x].z;
    float2 sp1 = projected_vertices[triangle.y].xy;
    float inv_depth1 = projected_vertices[triangle.y].z;
    float2 sp2 = projected_vertices[triangle.z].xy;
    float inv_depth2 = projected_vertices[triangle.z].z;

    float3 bi = interpolate_barycentric(int2(tid.x, tid.y), sp0, sp1, sp2, inv_depth0, inv_depth1, inv_depth2);
    if (bi.x == 0 && bi.y == 0 && bi.z == 0) return;

    float3 c0 = vertex_colors[triangle.x];
    float3 c1 = vertex_colors[triangle.y];
    float3 c2 = vertex_colors[triangle.z];
    float3 c = bi.x * c0 + bi.y * c1 + bi.z * c2;

    image[tid] = float4(c, 1.0);
}
