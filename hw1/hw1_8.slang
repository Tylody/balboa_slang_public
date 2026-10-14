import hw1_common;

float4 alphaRedraw(float4 color, float4 res) {
    float4 newColor = float4(color * (1 - res.w));
    newColor.xyz = newColor.xyz * color.w;
    res = res + newColor;
    return res;
}

float4 draw_circle(Shape shape, Circle circle, float2 tid, uint height, 
    RWTexture2D<float4> image,
    float4 res) {
        
        float2 q = float2(tid.x, tid.y);
        float3 qAug = float3(q, 1);
        float3 qObjSpace = mul(shape.world_to_obj,qAug);
        
        q = float2(qObjSpace.x,qObjSpace.y);
        float dist = distance(circle.center, q);
        
        if (dist >= circle.radius - (shape.stroke_width / 2)
            && dist <= circle.radius + (shape.stroke_width / 2)
        && shape.use_stroke
    ) {
        res = alphaRedraw(shape.stroke_rgba, res);
    }
    
    if (dist <= circle.radius && shape.use_fill) {
        res = alphaRedraw(shape.fill_rgba, res);
    }
    
    return res;
}

float4 draw_polyline(Shape shape, Polyline polyline, float2 tid, uint height,
    StructuredBuffer<float2> points,
    RWTexture2D<float4> image,
    float4 res) {
        
        float2 q = float2(tid.x, tid.y);
        float3 qAug = float3(q, 1);
        float3 qObjSpace = mul(shape.world_to_obj, qAug);
        
        q = float2(qObjSpace.x, qObjSpace.y);
        
        bool isInStroke = false;
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
    
    if (isInStroke && shape.use_stroke) {
        res = alphaRedraw(shape.stroke_rgba, res);
    }
    
    if (line_intersects != 0 && shape.use_fill) {
        res = alphaRedraw(shape.fill_rgba, res);
    }
    
    return res;
}

float4 draw_bezier(Shape shape, QuadraticBezier bezier, float2 tid, uint height,
    StructuredBuffer<float2> points,
    RWTexture2D<float4> image,
    float4 res) {
        
    float2 q = float2(tid.x, tid.y);
    float3 qAug = float3(q, 1);
    float3 qObjSpace = mul(shape.world_to_obj, qAug);
    
    q = float2(qObjSpace.xy);
    
    bool isInStroke = false;
    int line_intersects = 0;
    
    for (int i = bezier.pts_begin; i < bezier.pts_end-1; i+=2) {
        float2 p0 = points[i];
        float2 p1 = points[(i + 1)];
        float2 p2 = points[max((i + 2)%bezier.pts_end,bezier.pts_begin)];
        
        // first find t
        
        float ay = (p0.y) + (-2 * p1.y) + (p2.y);
        float by = (-2 * p0.y) + (2 * p1.y);
        float cy = (p0.y);
        
        float2 t;
        if (solve_quadratic(ay, by, cy - q.y, t) && t.x != t.y) {
            
            float ax = (p0.x) + (-2 * p1.x) + (p2.x);
            float bx = (-2 * p0.x) + (2 * p1.x);
            float cx = (p0.x);
            
            
            float x1 = (ax * (t.x * t.x)) + (bx * t.x) + cx;
            float x2 = (ax * (t.y * t.y)) + (bx * t.y) + cx;
            
            // to find upGoing, calculate tangent at q(t)
            
            float tan1 = (-2 * p0.y) + (2 * p1.y) + t.x * ( (2 * p0.y) + (-4 * p1.y) + (2 * p2.y) );
            float tan2 = (-2 * p0.y) + (2 * p1.y) + t.y * ( (2 * p0.y) + (-4 * p1.y) + (2 * p2.y) );
            
            bool upGoing1 = tan1 > 0;
            if (q.x <= x1) {
                if (upGoing1 && t.x > 0 && t.x <= 1) {
                    line_intersects += 1;
                }
                else if (!upGoing1 && t.x >= 0 && t.x < 1) {
                    line_intersects -= 1;
                }
            }
            
            bool upGoing2 = tan2 > 0;
            if (q.x <= x2) {
                if (upGoing2 && t.y > 0 && t.y <= 1) {
                    line_intersects += 1;
                }
                else if (!upGoing2 && t.y >= 0 && t.y < 1) {
                    line_intersects -= 1;
                }
            }
        }
        
        // stroke fill
        
        float Ax = p0.x - 2 * p1.x + p2.x;
        Ax = 2 * Ax * Ax;
        
        float Bx = 6 * (p0.x - 2 * p1.x + p2.x) * (p1.x - p0.x);
        
        float Cx = p1.x - p0.x;
        Cx = Cx * Cx * 4 + 2 * (p0.x - q.x) * (p0.x - 2 * p1.x + p2.x);
        
        float Dx = 2 * (p0.x - q.x) * (p1.x - p0.x);
        
        float Ay = p0.y - 2 * p1.y + p2.y;
        Ay = 2 * Ay * Ay;
        
        float By = 6 * (p0.y - 2 * p1.y + p2.y) * (p1.y - p0.y);
        
        float Cy = p1.y - p0.y;
        Cy = Cy * Cy * 4 + 2 * (p0.y - q.y) * (p0.y - 2 * p1.y + p2.y);
        
        float Dy = 2 * (p0.y - q.y) * (p1.y - p0.y);
        
        float3 tt;
        int n = solve_cubic(Ax + Ay, Bx + By, Cx + Cy, Dx + Dy, tt);
        
        float2 r1 = (1 - tt.x) * (1 - tt.x) * p0 + 2 * (1 - tt.x) * tt.x * p1 + tt.x * tt.x * p2;
        float2 r2 = (1 - tt.y) * (1 - tt.y) * p0 + 2 * (1 - tt.y) * tt.y * p1 + tt.y * tt.y * p2;
        float2 r3 = (1 - tt.z) * (1 - tt.z) * p0 + 2 * (1 - tt.z) * tt.z * p1 + tt.z * tt.z * p2;
        
        if (
            (n >= 1 && distance(r1, q) < shape.stroke_width / 2 && tt.x >= 0 && tt.x <= 1)
            || (n >= 2 && distance(r2,q) < shape.stroke_width / 2 && tt.y >= 0 && tt.y <= 1)
            || (n >= 3 && distance(r3,q) < shape.stroke_width / 2 && tt.z >= 0 && tt.z <= 1)
        ) isInStroke = true;

        // rounded joins 
        if ((i != bezier.pts_begin || bezier.closed)
            && distance(q, p0) < shape.stroke_width / 2
        ) isInStroke = true;
    }
    
    if (isInStroke && shape.use_stroke) {
        res = alphaRedraw(shape.stroke_rgba, res);
    }
    
    if (line_intersects != 0 && shape.use_fill) {
        res = alphaRedraw(shape.fill_rgba, res);
    }
    
    return res;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
    uniform float3 background,
    uniform int samples_per_axis,
    StructuredBuffer<Shape> shapes,
    uniform int num_shapes,
    StructuredBuffer<Circle> circles,
    StructuredBuffer<Polyline> polylines,
    StructuredBuffer<QuadraticBezier> quadratic_beziers,
    StructuredBuffer<float2> points,
    RWTexture2D<float4> image) {
        uint width, height;
        image.GetDimensions(width, height);
        if (tid.x >= width || tid.y >= height) {
            return;
        }
        
        image[tid] = float4(1.0);
        // TODO: your code here
        
        image[tid] = float4(background,1);
        
        float4 colorSum = float4(0, 0, 0, 0);
        float subpixelSize = 1.0 / samples_per_axis;
        
        for (int a = 0; a < samples_per_axis; a++) {
            for (int b = 0; b < samples_per_axis; b++) {
                
                float x = (subpixelSize / 2) + a * subpixelSize;
                float y = (subpixelSize / 2) + b * subpixelSize;
                float2 pt = float2(tid.x + x, height - tid.y - y);
                
                float4 subpxColor = float4(0,0,0,0);
                
                for (int s = 0; s < num_shapes; s++) {
                    Shape shape = shapes[s];
                    
                    if (shape.type == ShapeType.Circle) {
                        subpxColor = draw_circle(shape, circles[shape.index], pt, height, image, subpxColor);
                    }
                    else if (shape.type == ShapeType.Polyline) {
                        subpxColor = draw_polyline(shape, polylines[shape.index], pt, height, points, image, subpxColor);
                    }
                    else if (shape.type == ShapeType.QuadraticBezier) {
                        subpxColor = draw_bezier(shape, quadratic_beziers[shape.index], pt, height, points, image, subpxColor);
                    }
                }
                
                // end alpha blend, handle the bg
                if (subpxColor.w < 1.0) {
                    subpxColor += (1 - subpxColor.w) * float4(background,1);
                }
                
                colorSum += subpxColor;
            }
            
        }
        image[tid] = colorSum / float(samples_per_axis * samples_per_axis);
    }
    