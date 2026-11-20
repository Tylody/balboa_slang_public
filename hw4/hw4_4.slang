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
                         // The CSR matrix data structure
                         StructuredBuffer<uint> row_ptr,
                         StructuredBuffer<uint> neighbors,
                         // Rest lengthes of the neighbor edges
                         StructuredBuffer<float> neighbor_rest,
                         uniform uint num_particles) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.4. Your code below.

    uint n;
    uint _;
    neighbors.GetDimensions(n, _);

    float3 correction = 0;
    int count = 0;
    CorrectionContext c = CorrectionContext(inv_masses, positions_in);

    int start = row_ptr[tid];
    int end = tid < num_particles - 1 ? row_ptr[tid + 1] : n;

    for (int i = start; i < end; i++) {
        correction += calculate_correction(tid, neighbors[i], neighbor_rest[i], c);
        count += 1;
    }

    if (count > 0) correction /= count;

    positions_out[tid] = positions_in[tid] + correction;
}
