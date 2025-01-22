#pragma once

float3 hsv2rgb(float3 hsv)
{
    float hh = saturate(hsv.r) * 360.0;
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    int i = int(hh);
    float ff = hh - i;
    float p = hsv.b * (1.0 - hsv.g);
    float q = hsv.b * (1.0 - (hsv.g * ff));
    float t = hsv.b * (1.0 - (hsv.g * (1.0 - ff)));

    float3 rgb = 0;

    switch(i) {
    case 0:
        rgb.r = hsv.b;
        rgb.g = t;
        rgb.b = p;
        break;
    case 1:
        rgb.r = q;
        rgb.g = hsv.b;
        rgb.b = p;
        break;
    case 2:
        rgb.r = p;
        rgb.g = hsv.b;
        rgb.b = t;
        break;

    case 3:
        rgb.r = p;
        rgb.g = q;
        rgb.b = hsv.b;
        break;
    case 4:
        rgb.r = t;
        rgb.g = p;
        rgb.b = hsv.b;
        break;
    case 5:
    default:
        rgb.r = hsv.b;
        rgb.g = p;
        rgb.b = q;
        break;
    }
    
    return rgb;     
}

// cosine based palette, 4 vec3 params
float3 iq_palette( in float t, in float3 a, in float3 b, in float3 c, in float3 d )
{
    return a + b*cos( 6.283185*(c*t+d) );
}