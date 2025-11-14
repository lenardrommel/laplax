#include "interflib.h"

#include "utils.h"

#include <vector>
#include <math.h>
#include <future>
#include <algorithm>
#include <random>
#include <iostream>
#include <mutex>
#include <fstream>


namespace {

struct Wave {
	double ampl;
	double phase;
};

void calcImage(
		double xstart, double ystart, int xpoints, int ypoints, double pixel_size,
		const Vector& wave_vector1, const Vector& center1, double radius1, const double* beamImage1,
		double length1, int nPoints1, double sigma1x, double sigma1y, double beam1Ampl, double beam1Rotation,
        const Vector& wave_vector2, const Vector& center2, double radius2, const double* beamImage2,
        double length2, int nPoints2, double sigma2x, double sigma2y, double beam2Ampl, double beam2Rotation,
        double r_curvature, int nForwardFrames, int nBackwardFrames, double lambda, double omega, bool hasInterference,
        double noiseCoeff, double amplStd, double phaseStd, int nThreads, uint8_t* image, double* totIntens)
{
    std::random_device rnd;
    std::mt19937 generator(rnd());
    std::uniform_real_distribution<> distrib(0, noiseCoeff);
    std::uniform_real_distribution<> phaseDistrib(0, 2 * M_PI);
    std::uniform_real_distribution<> amplDistrib(0.0, amplStd);
    std::normal_distribution<> piezoDistrib{0, phaseStd};

    // const double maxIntens = beam1Ampl * beam1Ampl + beam2Ampl * beam2Ampl + 2 * beam1Ampl * beam2Ampl;
    const double maxIntens = 4;

    auto noise = [&]() {
        return distrib(generator);
    };

    auto rndPhase = [&] {
        return phaseDistrib(generator);
    };

    auto amplNoise = [&] {
        return 1.0 - amplDistrib(generator);
    };

    auto piezoNoise = [&](){
        return piezoDistrib(generator);
    };

	const double k = 2 * M_PI / lambda;
	const double cosBeam1Rot = cos(beam1Rotation);
	const double sinBeam1Rot = sin(beam1Rotation);
	const double cosBeam2Rot = cos(beam2Rotation);
	const double sinBeam2Rot = sin(beam2Rotation);

    auto calcWave1 = [&](double z, double x, double y) {
        if (beamImage1) {
            const double step1 = length1 / nPoints1;
            auto clipPixels = [&](int value) {
                return std::min(std::max(0, value), nPoints1 - 1);
            };
            const int nPixelsX = clipPixels((x - center1[0]) / step1 + nPoints1 / 2);
            const int nPixelsY = clipPixels((y - center1[1]) / step1 + nPoints1 / 2);
            double ampl = beamImage1[nPixelsX * nPoints1 + nPixelsY];

            return  Wave{ampl * amplNoise() * beam1Ampl, z * k};
        }

        // rotation
        const double xPrime = x * cosBeam1Rot - y * sinBeam1Rot;
        const double yPrime = x * sinBeam1Rot + y * cosBeam1Rot;
        const double xCenterPrime = center1[0] * cosBeam1Rot - center1[1] * sinBeam1Rot;
        const double yCenterPrime = center1[0] * sinBeam1Rot + center1[1] * cosBeam1Rot;

        const double r2 = sigma1x * (xPrime - xCenterPrime) * (xPrime - xCenterPrime) + sigma1y * (yPrime - yCenterPrime) * (yPrime - yCenterPrime);
    	return Wave{std::exp(-r2 / (radius1 * radius1)) * amplNoise() * beam1Ampl, z * k};
    	//return Wave{beam1Ampl * (r2 <= radius1 * radius1), z * k};
    };

    auto calcWave2 = [&](double z, double x, double y) {
        if (beamImage2) {
            const double step2 = length2 / nPoints2;
            auto clipPixels = [&](int value) {
                return std::min(std::max(0, value), nPoints2 - 1);
            };
            const int nPixelsX = clipPixels((x - center2[0]) / step2 + nPoints2 / 2);
            const int nPixelsY = clipPixels((y - center2[1]) / step2 + nPoints2 / 2);
            double ampl = beamImage2[nPixelsX * nPoints2 + nPixelsY];

            // TODO add lense here
            return  Wave{ampl * amplNoise() * beam2Ampl, z * k};
        }

        // rotation
        const double xPrime = x * cosBeam2Rot - y * sinBeam2Rot;
        const double yPrime = x * sinBeam2Rot + y * cosBeam2Rot;
        const double xCenterPrime = center2[0] * cosBeam2Rot - center2[1] * sinBeam2Rot;
        const double yCenterPrime = center2[0] * sinBeam2Rot + center2[1] * cosBeam2Rot;

        const double r2 = sigma2x * (xPrime - xCenterPrime) * (xPrime - xCenterPrime) + sigma2y * (yPrime - yCenterPrime) * (yPrime - yCenterPrime);
    	return Wave{std::exp(-r2 / (radius2 * radius2)) * amplNoise() * beam2Ampl, z * k  + 0.5 * k * r2  / r_curvature};
    	//return Wave{beam2Ampl * (r2 <= radius2 * radius2), z * k};
    };


    auto calcIntens = [&](double a1, double a2, double deltaPhi) {
        const auto i1 = a1 * a1;
        const auto i2 = a2 * a2;
        double result = 0;

        if (hasInterference) {
            result = i1 + i2 + 2 * sqrt(i1 * i2) * cos(deltaPhi);
        } else {
            result = i1 + i2;
        }

        return result;
    };

    const int totalPoints = xpoints * ypoints;
    std::vector<double> ampl1(totalPoints);
    std::vector<double> ampl2(totalPoints);
    std::vector<double> deltaPhase(totalPoints);

	auto worker = [&](int kStart, int kEnd) {
		for (int k = kStart; k < kEnd; ++k) {
			int i = k / ypoints;
			int j = k % ypoints;
			const Vector point = {xstart + i * pixel_size, ystart + j * pixel_size, 0};

			const Vector source2 = utils::backTrack(point, wave_vector2, center2);
	        const double dist2 = utils::dist(point, source2);
	        auto w2 = calcWave2(dist2, source2[0], source2[1]);

	        const Vector source1 = utils::backTrack(point, wave_vector1, center1);
	        const double dist1 = utils::dist(point, source1);
	        auto w1 = calcWave1(dist1, source1[0], source1[1]);

            ampl1[k] = w1.ampl;
            ampl2[k] = w2.ampl;
            deltaPhase[k] = w1.phase - w2.phase;
		}
	};

	const int pointsPerThread = totalPoints / nThreads;
	std::vector<std::future<void>> futures;

	for (int iThread = 0; iThread < nThreads; ++iThread) {
		int kStart = pointsPerThread * iThread;
		int kEnd = kStart + pointsPerThread;
		futures.push_back(std::async(std::launch::async, worker, kStart, kEnd));
	}

    // wait for intens1, intens2, deltaPhase
	for (const auto& f : futures) {
		f.wait();
	}

    auto imageWorker = [&](uint8_t* img, double time, double& integIntens) {
        for (int k = 0; k < totalPoints; ++k) {
            const double intens = calcIntens(
                ampl1[k], 
                ampl2[k], 
                deltaPhase[k] - omega * time
            );
            const double rescaledIntens = std::min(intens / maxIntens, 1.0);
            const double intensWithNoise = (rescaledIntens + noise()) / (1 + noiseCoeff);
            img[k] = static_cast<uint8_t>(255 * intensWithNoise);
            integIntens += intens;
        }
    };

    std::vector<std::future<void>> imageFutures;

    auto startPhase = rndPhase();
    for (int iFrame = 0; iFrame < nForwardFrames; ++iFrame) {
        double time = startPhase + 2 * M_PI * iFrame / nForwardFrames + piezoNoise();
        int ind = iFrame * totalPoints;
        uint8_t* img = image + ind;
        imageFutures.push_back(std::async(
            std::launch::async, imageWorker, img, time,
            std::ref(totIntens[iFrame])));
    }

    startPhase = rndPhase();
    for (int iFrame = 0; iFrame < nBackwardFrames; ++iFrame) {
        double time = startPhase + 2 * M_PI * (1.0 - static_cast<double>(iFrame) / nBackwardFrames) + piezoNoise();
        int ind = (iFrame + nForwardFrames) * totalPoints;
        uint8_t* img = image + ind;
        imageFutures.push_back(std::async(
            std::launch::async, imageWorker, img, time,
            std::ref(totIntens[iFrame + nForwardFrames])));
    }

    // wait for frames
    for (const auto& f : imageFutures) {
        f.wait();
    }
}

} // namespace


void calc_image(
		double xstart, double ystart, int xpoints, int ypoints, double pixel_size,
		const double* vector1, const double*  cnt1, double radius1, const double* beamImage1,
		double length1, int nPoints1, double sigma1x, double sigma1y, double beam1Ampl, double beam1Rotation,
        const double* vector2, const double*  cnt2, double radius2, const double* beamImage2,
        double length2, int nPoints2, double sigma2x, double sigma2y, double beam2Ampl, double beam2Rotation,
        double r_curvature, int nForwardFrames, int nBackwardFrames, double lambda, double omega, bool hasInterference,
        double noiseCoeff, double amplStd, double phaseStd, int nThreads, uint8_t* image, double* totIntens)
{
	auto wave_vector1 = Vector{vector1[0], vector1[1], vector1[2]};
	auto wave_vector2 = Vector{vector2[0], vector2[1], vector2[2]};
	auto center1 = Vector{cnt1[0], cnt1[1], cnt1[2]};
	auto center2 = Vector{cnt2[0], cnt2[1], cnt2[2]};

    calcImage(
        xstart, ystart, xpoints, ypoints, pixel_size,
        wave_vector1, center1, radius1, beamImage1, length1, nPoints1, sigma1x, sigma1y, beam1Ampl, beam1Rotation,
        wave_vector2, center2, radius2, beamImage2, length2, nPoints2, sigma2x, sigma2y, beam2Ampl, beam2Rotation,
        r_curvature, nForwardFrames, nBackwardFrames, lambda, omega, hasInterference,
        noiseCoeff,  amplStd, phaseStd, nThreads, image, totIntens);
}
