#pragma once

#include "defines.h"
#include <cstdint>

extern "C" {
	EXPORT void calc_image(
		double xstart, double ystart, int xpoints, int ypoints, double pixel_size,
		const double*  waveVector1, const double*  center1, double radius1, const double* beamImage1,
		double length1, int nPoints1, double sigma1x, double sigma1y, double beam1Ampl, double beam1Rotation,
        const double*  waveVector2, const double*  center2, double radius2, const double* beamImage2,
        double length2, int nPoints2, double sigma2x, double sigma2y, double beam2Ampl, double beam2Rotation,
        double r_curvature, int nForwardFrames, int nBackwardFrames, double lambda, double omega, bool hasInterference,
        double noiseCoeff, double amplStd, double phaseStd, int nThreads, uint8_t* image, double* total_intens);
}

