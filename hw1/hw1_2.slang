import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  StructuredBuffer<float2> polyline,
                  uniform int num_polyline,
                  uniform bool is_closed,
                  uniform bool use_fill_color,
                  uniform float3 fill_color,
                  uniform bool use_stroke_color,
                  uniform float3 stroke_color,
                  uniform float stroke_width,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(1.0);
    // TODO: your code here
    bool isInStroke = false;

    float2 q = float2(tid.x + 0.5, height - (tid.y + 0.5));
    float line_intersects = 0;
    for (int i = 0; i < num_polyline; i++) {
        float2 a = polyline[i];
        float2 b = polyline[(i+1)%num_polyline];

        bool upGoing = b.y > a.y;
        float2 p_0 = upGoing ? a : b;
        float2 p_1 = upGoing ? b : a;
        int windNum = upGoing ? 1 : -1;

        // check if line from tid goes between p_0 and p_1
        // find x on line where y = tid.y

        if (p_0.y < q.y && q.y <= p_1.y) {
            float t  = (q.y - p_0.y) / (p_1.y - p_0.y);
            float lX = p_0.x + t * (p_1.x - p_0.x);
            if (lX >= q.x) line_intersects += windNum;
        }

        // stroke fill
        float lineLen = length(b - a);
        float l = dot((b - a),(q - a)) / lineLen;
        float2 qp = a + l * ((b - a) / lineLen);

        if ((i != num_polyline-1 || is_closed)
            && l > 0 
            && l < lineLen
            && length(q-qp) < (stroke_width / 2)
        ) isInStroke = true;

        // edge case: within w/2 from join point means round cap
        if(
            (
                (i != num_polyline-1 && i != 0) 
                || is_closed
            ) && length(q-a) < (stroke_width / 2)
        ) isInStroke = true;
    }

    if ((line_intersects != 0
        && use_fill_color)
    ) image[tid] = float4(fill_color,1);
    if (isInStroke && use_stroke_color) image[tid] = float4(stroke_color,1);
}