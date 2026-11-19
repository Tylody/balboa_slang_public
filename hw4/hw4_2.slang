[shader("compute")]
[numthreads(64, 1, 1)]
void predict(uint tid : SV_DispatchThreadID,
             // Updated position
             RWStructuredBuffer<float3> positions,
             // Store the position before update
             RWStructuredBuffer<float3> old_positions,
             RWStructuredBuffer<float3> velocities,
             StructuredBuffer<float> inv_masses,
             uniform uint num_particles,
             // time step size
             uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.2. Your code below.

    float3 x = positions[tid];
    float3 v = velocities[tid];

    old_positions[tid] = x;
    if (inv_masses[tid] == 0) return;    

    float3 vp = v + dt * float3(0,-9.8,0);
    float3 xp = x + dt * vp;

    positions[tid] = xp;
    velocities[tid] = vp;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         StructuredBuffer<float> rest_lengths,
                         uniform uint num_particles) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.2. Your code below.

    float3 correction1 = 0;
    bool c1_init = false;
    float3 correction2 = 0;
    bool c2_init = false;

    if (tid > 0) {
        float3 xi = positions_in[tid];
        float3 xj = positions_in[tid - 1];
        float3 d = xj - xi;

        float len = length(d);
        float w = inv_masses[tid] + inv_masses[tid - 1];

        if (w > 0 && len > 1e-8) {
            correction1 = (inv_masses[tid] / w) * (length(d) - rest_lengths[tid - 1]) * (d / length(d));
        }
        c1_init = true;
    }

    if (tid < num_particles - 1) {
        float3 xi = positions_in[tid];
        float3 xj = positions_in[tid + 1];
        float3 d = xj - xi;

        float len = length(d);
        float w = inv_masses[tid] + inv_masses[tid + 1];

        if (w > 0 && len > 1e-8) {
            correction2 = (inv_masses[tid] / w) * (length(d) - rest_lengths[tid]) * (d / length(d));
        }
        c2_init = true;
    }

    float3 correction;

    if (!(c1_init && c2_init)) {
        correction = c1_init ? correction1 : correction2;
    } else {
        correction = (correction1 + correction2) / 2;
    }
    positions_out[tid] = positions_in[tid] + correction;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void update_velocity(uint tid : SV_DispatchThreadID,
                     RWStructuredBuffer<float3> positions,
                     StructuredBuffer<float3> old_positions,
                     RWStructuredBuffer<float3> velocities,
                     StructuredBuffer<float> inv_masses,
                     uniform uint num_particles,
                     // time step size
                     uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.2. Your code below.

    if (inv_masses[tid] == 0) return;

    if (positions[tid].y < -0.97) {
        positions[tid].y = -0.97;
    }
    velocities[tid] = (positions[tid] - old_positions[tid]) / dt;
}
