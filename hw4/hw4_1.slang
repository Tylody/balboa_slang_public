[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint tid : SV_DispatchThreadID,
                  RWStructuredBuffer<float3> positions,
                  RWStructuredBuffer<float3> velocities,
                  uniform uint num_particles,
                  // time step size
                  uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.1. Your code below.

    float3 x = positions[tid];
    float3 v = velocities[tid];

    float3 vp = v + dt * float3(0,-9.8,0);
    float3 xp = x + dt * vp;

    if (xp.y < -0.97) {
        xp.y = -0.97;
        vp.y *= -1;
    }

    positions[tid] = xp;
    velocities[tid] = vp;
}
