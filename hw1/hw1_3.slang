import hw1_common;

void draw_circle(Shape shape, Circle circle, uint2 tid, uint height,
                 RWTexture2D<float4> image) {
    circle.center.y = height - circle.center.y;
    float dist = distance(circle.center, float2(tid) + float2(0.5,0.5));

    if (dist <= circle.radius && shape.use_fill) {
        image[tid] = float4(shape.fill_rgba);
    }

    if (dist >= circle.radius - (shape.stroke_width / 2)
        && dist <= circle.radius + (shape.stroke_width / 2)
        && shape.use_stroke
    ) {
        image[tid] = float4(shape.stroke_rgba);
    }
}

void draw_polyline(Shape shape, Polyline polyline, uint2 tid, uint height,
                   StructuredBuffer<float2> points,
                   RWTexture2D<float4> image) {
    bool isInStroke = false;
    float2 q = float2(tid.x + 0.5, height - (tid.y + 0.5));

    float line_intersects = 0;

    for (int i = polyline.pts_begin; i < polyline.pts_end; i++) {
        float2 a = points[i];
        float2 b = points[max((i+1)%polyline.pts_end,polyline.pts_begin)];

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

        if ((i != polyline.pts_end-1 || polyline.closed)
            && l > 0
            && l < lineLen
            && length(q-qp) < (shape.stroke_width / 2)
        ) isInStroke = true;

        // edge case: within w/2 from join point means round cap
        if(
            (
                (i != polyline.pts_end-1 && i != polyline.pts_begin)
                || polyline.closed
            ) && length(q-a) < (shape.stroke_width / 2)
        ) isInStroke = true;
    }

    if ((line_intersects != 0
        && shape.use_fill)
    ) image[tid] = float4(shape.fill_rgba);
    if (isInStroke && shape.use_stroke) image[tid] = float4(shape.stroke_rgba);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform int /*samples_per_axis*/, // ignore for now
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Circle> circles,
                  StructuredBuffer<Polyline> polylines,
                  StructuredBuffer<QuadraticBezier> /*quadratic_beziers*/, // ignore for now
                  StructuredBuffer<float2> points,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // TODO: your code here
    image[tid] = float4(background,1);

    for (int s = num_shapes-1; s >= 0; s--) {
        Shape shape = shapes[s];

        if (shape.type == ShapeType.Circle) {
            draw_circle(shape, circles[shape.index], tid, height, image);
        }

        else if (shape.type == ShapeType.Polyline) {
            draw_polyline(shape, polylines[shape.index], tid, height, points, image);
        }
    }
}
