#pragma once

#include "shared.h"

#define BLOCKER_SEARCH_NUM_SAMPLES 16
#define PCF_NUM_SAMPLES 16
#define NEAR_PLANE 10
#define LIGHT_WORLD_SIZE .26
#define LIGHT_FRUSTUM_WIDTH 30.0
// Assuming that LIGHT_FRUSTUM_WIDTH == LIGHT_FRUSTUM_HEIGHT
#define LIGHT_SIZE_UV (LIGHT_WORLD_SIZE / LIGHT_FRUSTUM_WIDTH)
    
static float2 poissonDisk[16] = {
    float2( -0.94201624, -0.39906216 ),
    float2( 0.94558609, -0.76890725 ),
    float2( -0.094184101, -0.92938870 ),
    float2( 0.34495938, 0.29387760 ),
    float2( -0.91588581, 0.45771432 ),
    float2( -0.81544232, -0.87912464 ),
    float2( -0.38277543, 0.27676845 ),
    float2( 0.97484398, 0.75648379 ),
    float2( 0.44323325, -0.97511554 ),
    float2( 0.53742981, -0.47373420 ),
    float2( -0.26496911, -0.41893023 ),
    float2( 0.79197514, 0.19090188 ),
    float2( -0.24188840, 0.99706507 ),
    float2( -0.81409955, 0.91437590 ),
    float2( 0.19984126, 0.78641367 ),
    float2( 0.14383161, -0.14100790 )
};

float PenumbraSize(float zReceiver, float zBlocker) //Parallel plane estimation
{
    return (zReceiver - zBlocker) / zBlocker;
}

// NOTE: zReceiver is in positive view space
void FindBlocker(Texture2D shadowMapTex, SamplerState point_sampler, float4 projection_info, out float avgBlockerDepth, out float numBlockers, float2 uv, float zReceiver )
{
    //This uses similar triangles to compute what
    //area of the shadow map we should search
    float searchWidth = LIGHT_SIZE_UV * (zReceiver - NEAR_PLANE) / zReceiver;
    //searchWidth = LIGHT_SIZE_UV;
    searchWidth = max(searchWidth, 0.0);
    float blockerSum = 0;
    numBlockers = 0;

    for( int i = 0; i < BLOCKER_SEARCH_NUM_SAMPLES; ++i )
    {
        float shadowMapDepth = shadowMapTex.SampleLevel(point_sampler, uv + poissonDisk[i] * searchWidth, 0).r;
        shadowMapDepth = linearize_depth(shadowMapDepth, projection_info);
        if ( shadowMapDepth < zReceiver ) 
        {
            blockerSum += shadowMapDepth;
            numBlockers++;
        }
    }
    avgBlockerDepth = blockerSum / numBlockers;
}

float PCF_Filter( Texture2D shadowMapTex, SamplerComparisonState pcf_sampler, float2 uv, float zReceiver, float filterRadiusUV )
{
    float sum = 0.0f;
    for ( int i = 0; i < PCF_NUM_SAMPLES; ++i )
    {
        float2 offset = poissonDisk[i] * filterRadiusUV;
        sum += shadowMapTex.SampleCmpLevelZero(pcf_sampler, uv + offset, zReceiver);
    }
    return sum / PCF_NUM_SAMPLES;
}

float PCSS ( Texture2D shadowMapTex, SamplerState point_sampler, SamplerComparisonState pcf_sampler, float4 projection_info, float4 coords )
{
    float2 uv = coords.xy;
    float compare_value = coords.z;
    float linearized_z = linearize_depth(coords.z, projection_info);

    // STEP 1: blocker search
    float avgBlockerDepth = 0;
    float numBlockers = 0;
    FindBlocker( shadowMapTex, point_sampler, projection_info, avgBlockerDepth, numBlockers, uv, linearized_z );
    if( numBlockers < 1 ) //There are no occluders so early out (this saves filtering)
        return 1.0f;
    // STEP 2: penumbra size
    float penumbraRatio = PenumbraSize(linearized_z, avgBlockerDepth);
    float filterRadiusUV = penumbraRatio * LIGHT_SIZE_UV * NEAR_PLANE / linearized_z;// * linearized_z / NEAR_PLANE;
    //filterRadiusUV = 0;

    // STEP 3: filtering
    return PCF_Filter( shadowMapTex, pcf_sampler, uv, compare_value, filterRadiusUV );
}