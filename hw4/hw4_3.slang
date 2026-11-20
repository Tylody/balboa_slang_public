
struct CorrectionContext {
    StructuredBuffer<float> inv_masses;
    StructuredBuffer<float3> positions_in;
}

float3 calculate_correction(
    int i,
    int j,
    float rest_length,
    CorrectionContext c
) {
    float3 xi = c.positions_in[i];
    float3 xj = c.positions_in[j];

    float wi = c.inv_masses[i];
    float wj = c.inv_masses[j];

    float3 correction = 0;
    float3 d = xj - xi;
    float len = length(d);
    float w = wi + wj;
    if (w > 0 && len > 1e-8) {
        correction = (wi / w) * (len - rest_length) * (d / len);
    }
    return correction;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         uniform int2 grid_size,
                         uniform float2 spacing) {
    if (tid >= grid_size.x * grid_size.y) {
        return;
    }
    // TODO: HW 4.3. Your code below.

    int2 ind = int2(tid % grid_size.x, tid / grid_size.x);
    float3 correction = 0;
    int count = 0;

    CorrectionContext c = CorrectionContext(inv_masses, positions_in);

    // structure constraint
    float rl = spacing.x;
    if (ind.x > 0) { correction += calculate_correction(tid,tid-1,rl,c); count += 1; }
    if (ind.x < grid_size.x - 1) { correction += calculate_correction(tid, tid + 1, rl, c); count += 1; }
    rl = spacing.y;
    if (ind.y > 0) { correction += calculate_correction(tid,tid-grid_size.x,rl,c); count += 1; }
    if (ind.y < grid_size.y - 1) { correction += calculate_correction(tid,tid+grid_size.x,rl,c); count += 1; }

    // shearing constraint
    rl = length(spacing);
    if (ind.x > 0 && ind.y > 0) { correction += calculate_correction(tid, tid - 1 - grid_size.x,rl,c); count += 1; }
    if (ind.x < grid_size.x - 1 && ind.y > 0) { correction += calculate_correction(tid, tid + 1 - grid_size.x,rl,c); count += 1; }
    if (ind.x > 0 && ind.y < grid_size.y - 1) { correction += calculate_correction(tid, tid - 1 + grid_size.x,rl,c); count += 1; }
    if (ind.x < grid_size.x - 1 && ind.y < grid_size.y - 1) { correction += calculate_correction(tid, tid + 1 + grid_size.x,rl,c); count += 1; }

    // bending constraint
    rl = 2 * spacing.x;
    if (ind.x > 1) { correction += calculate_correction(tid, tid - 2,rl,c); count += 1; }
    if (ind.x < grid_size.x - 2) { correction += calculate_correction(tid, tid + 2, rl, c); count += 1; }
    rl = 2 * spacing.y;
    if (ind.y > 1) { correction += calculate_correction(tid,tid-(grid_size.x*2),rl,c); count += 1; }
    if (ind.y < grid_size.y - 2) { correction += calculate_correction(tid,tid+(grid_size.x*2),rl,c); count += 1; }

    if (count > 0) correction /= count;

    positions_out[tid] = positions_in[tid] + correction;
}
