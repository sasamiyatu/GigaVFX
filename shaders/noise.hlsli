// The MIT License
// Copyright © 2017 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// https://www.youtube.com/c/InigoQuilez
// https://iquilezles.org/

#pragma once

#include "simplex_noise.hlsli"
#include "random.hlsli"

float3 hash(uint3 x)
{
     return -1.0 + 2.0 * (float3(pcg3d(x)) / float(0xffffffffu));
}

// 0: cubic
// 1: quintic
#define INTERPOLANT 0

float gradient_noise3d( in float3 p )
{
    float3 i = floor( p );
    float3 f = frac( p );

#if INTERPOLANT==1
    // quintic interpolant
    float3 u = f*f*f*(f*(f*6.0-15.0)+10.0);
#else
    // cubic interpolant
    float3 u = f*f*(3.0-2.0*f);
#endif    

    return lerp( lerp( lerp( dot( hash( uint3(i) + uint3(0,0,0) ), f - float3(0.0,0.0,0.0) ), 
                          dot( hash( uint3(i) + uint3(1,0,0) ), f - float3(1.0,0.0,0.0) ), u.x),
                     lerp( dot( hash( uint3(i) + uint3(0,1,0) ), f - float3(0.0,1.0,0.0) ), 
                          dot( hash( uint3(i) + uint3(1,1,0) ), f - float3(1.0,1.0,0.0) ), u.x), u.y),
                lerp( lerp( dot( hash( uint3(i) + uint3(0,0,1) ), f - float3(0.0,0.0,1.0) ), 
                          dot( hash( uint3(i) + uint3(1,0,1) ), f - float3(1.0,0.0,1.0) ), u.x),
                     lerp( dot( hash( uint3(i) + uint3(0,1,1) ), f - float3(0.0,1.0,1.0) ), 
                          dot( hash( uint3(i) + uint3(1,1,1) ), f - float3(1.0,1.0,1.0) ), u.x), u.y), u.z );
}


// Computes the analytic derivatives of a 3D Gradient Noise. This can be used for example to compute normals to a
// 3d rocks based on Gradient Noise without approximating the gradient by having to take central differences.
//
// More info here: https://iquilezles.org/articles/gradientnoise
// return value noise (in x) and its derivatives (in yzw)
float4 gradient_noise_deriv( in float3 x )
{
     // grid
     int3 i = int3(floor(x));
     float3 f = frac(x);
    
#if INTERPOLANT==1
    // quintic interpolant
    float3 u = f*f*f*(f*(f*6.0-15.0)+10.0);
    float3 du = 30.0*f*f*(f*(f-2.0)+1.0);
#else
     // cubic interpolant
     float3 u = f*f*(3.0-2.0*f);
     float3 du = 6.0*f*(1.0-f);
#endif    
    
     // gradients
     float3 ga = -1.0 + 2.0 * (float3(pcg3d( i+int3(0,0,0) )) / float(0xFFFFFFFFu));
     float3 gb = -1.0 + 2.0 * (float3(pcg3d( i+int3(1,0,0) )) / float(0xFFFFFFFFu));
     float3 gc = -1.0 + 2.0 * (float3(pcg3d( i+int3(0,1,0) )) / float(0xFFFFFFFFu));
     float3 gd = -1.0 + 2.0 * (float3(pcg3d( i+int3(1,1,0) )) / float(0xFFFFFFFFu));
     float3 ge = -1.0 + 2.0 * (float3(pcg3d( i+int3(0,0,1) )) / float(0xFFFFFFFFu));
     float3 gf = -1.0 + 2.0 * (float3(pcg3d( i+int3(1,0,1) )) / float(0xFFFFFFFFu));
     float3 gg = -1.0 + 2.0 * (float3(pcg3d( i+int3(0,1,1) )) / float(0xFFFFFFFFu));
     float3 gh = -1.0 + 2.0 * (float3(pcg3d( i+int3(1,1,1) )) / float(0xFFFFFFFFu));
    
    // projections
    float va = dot( ga, f-float3(0.0,0.0,0.0) );
    float vb = dot( gb, f-float3(1.0,0.0,0.0) );
    float vc = dot( gc, f-float3(0.0,1.0,0.0) );
    float vd = dot( gd, f-float3(1.0,1.0,0.0) );
    float ve = dot( ge, f-float3(0.0,0.0,1.0) );
    float vf = dot( gf, f-float3(1.0,0.0,1.0) );
    float vg = dot( gg, f-float3(0.0,1.0,1.0) );
    float vh = dot( gh, f-float3(1.0,1.0,1.0) );
	
    // interpolations
    return float4( va + u.x*(vb-va) + u.y*(vc-va) + u.z*(ve-va) + u.x*u.y*(va-vb-vc+vd) + u.y*u.z*(va-vc-ve+vg) + u.z*u.x*(va-vb-ve+vf) + (-va+vb+vc-vd+ve-vf-vg+vh)*u.x*u.y*u.z,    // value
                 ga + u.x*(gb-ga) + u.y*(gc-ga) + u.z*(ge-ga) + u.x*u.y*(ga-gb-gc+gd) + u.y*u.z*(ga-gc-ge+gg) + u.z*u.x*(ga-gb-ge+gf) + (-ga+gb+gc-gd+ge-gf-gg+gh)*u.x*u.y*u.z +   // derivatives
                 du * (float3(vb,vc,ve) - va + u.yzx*float3(va-vb-vc+vd,va-vc-ve+vg,va-vb-ve+vf) + u.zxy*float3(va-vb-ve+vf,va-vb-vc+vd,va-vc-ve+vg) + u.yzx*u.zxy*(-va+vb+vc-vd+ve-vf-vg+vh) ));
}

// [Bridson et al. 2007, Curl-Noise for Procedural Fluid Flow]
// https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf
float3 curl_noise(float3 x, float t)
{
#if 0
     // Derivatives of gradient noise
     float scale = max(x.x, 0.0);
     float3 psi1 = gradient_noise_deriv(x).yzw * scale;
     float3 psi2 = gradient_noise_deriv(x + float3(123.2213, -1053.4, 60421.62)).yzw * scale;
     float3 psi3 = gradient_noise_deriv(x + float3(-9591.4, 1053.12, -7123.95)).yzw * scale;

     float3 curl = float3(psi3.y - psi2.z, psi1.z - psi3.x, psi2.x - psi1.y);
#else
     const int n_octaves = 3;
     float w_sum = 0;
     float weight = 1.0;
     float frequency = 1.0;

     float dp3dy = 0; 
     float dp2dz = 0; 
     float dp1dz = 0; 
     float dp3dx = 0; 
     float dp2dx = 0; 
     float dp1dy = 0; 

     for (int i = 0; i < n_octaves; ++i)
     {
          float4 x1 = float4(x, t);
          x1.xyz *= frequency;
          float4 x2 = float4(x + float3(123.2213, -1053.4, 60421.62), t);
          x2.xyz *= frequency;
          float4 x3 = float4(x + float3(-9591.4, 1053.12, -7123.95), t);
          x3.xyz *= frequency;

          const float epsilon = 1e-3f;
          dp3dy += weight * (snoise(x3 + float4(0, 1, 0, 0) * epsilon) - snoise(x3)) / epsilon;
          dp2dz += weight * (snoise(x2 + float4(0, 0, 1, 0) * epsilon) - snoise(x2)) / epsilon;
          dp1dz += weight * (snoise(x1 + float4(0, 0, 1, 0) * epsilon) - snoise(x1)) / epsilon;
          dp3dx += weight * (snoise(x3 + float4(1, 0, 0, 0) * epsilon) - snoise(x3)) / epsilon;
          dp2dx += weight * (snoise(x2 + float4(1, 0, 0, 0) * epsilon) - snoise(x2)) / epsilon;
          dp1dy += weight * (snoise(x1 + float4(0, 1, 0, 0) * epsilon) - snoise(x1)) / epsilon;

          w_sum += weight;
          frequency *= 2.0;
          weight *= 0.5;
     }

     dp3dy /= w_sum; 
     dp2dz /= w_sum; 
     dp1dz /= w_sum; 
     dp3dx /= w_sum; 
     dp2dx /= w_sum; 
     dp1dy /= w_sum; 

     float3 curl = float3(dp3dy - dp2dz, dp1dz - dp3dx, dp2dx - dp1dy);
#endif

     return curl;
}