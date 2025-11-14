import pycuda.autoinit
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray
from pycuda.compiler import SourceModule
import numpy as np
import torch

mod = SourceModule("""
#include <cmath>
#include <cstdint>


using Vector = double[3];

__device__ auto dist(const Vector& vector1, const Vector& vector2)
{
    double result2;
    for (int i = 0; i < 3; ++i) {
        result2 += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i]);
    }
    return sqrt(result2);
}

__device__ void backTrack(
    const Vector& target, 
    const Vector& normalVector, 
    const Vector& center,
    Vector& result)
{
    Vector targetCenter;
    targetCenter[0] = target[0] - center[0];
    targetCenter[1] = target[1] - center[1];
    targetCenter[2] = target[2] - center[2];

    const double dotScalar = 
        targetCenter[0] * normalVector[0] + 
        targetCenter[1] * normalVector[1] + 
        targetCenter[2] * normalVector[2];

    result[0] = target[0] - dotScalar * normalVector[0];
    result[1] = target[1] - dotScalar * normalVector[1];
    result[2] = target[2] - dotScalar * normalVector[2];
}

__device__ struct Wave {
    double ampl;
    double phase;
};

__device__ void calcImage(
    double start, double end, int nPoints,
    const Vector& wave_vector1, const Vector& center1, double radius1,
    const Vector& wave_vector2, const Vector& center2, double radius2,
    int nFrames, double lambda, double omega, int hasInterference,
    uint8_t* image, double* totIntens)
{
    const auto kVector = 2 * M_PI / lambda;

    auto calcWave1 = [&](double z, double x, double y) {
        const auto r2 = (x - center1[0]) * (x - center1[0]) + (y - center1[1]) * (y - center1[1]);
        return Wave{std::exp(-r2 / (radius1 * radius1)), z * kVector};
    };

    auto calcWave2 = [&](double z, double x, double y) {
        const auto r2 = (x - center2[0]) * (x - center2[0]) + (y - center2[1]) * (y - center2[1]);
        return Wave{std::exp(-r2 / (radius2 * radius2)), z * kVector};
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
        //return static_cast<uint8_t>(255.0 * result / 4.0);
    };

    double ampl1;
    double ampl2;
    double deltaPhase;

    const int k = blockIdx.x * blockDim.x + threadIdx.x;  

    {
        int i = k / nPoints;
        int j = k - i * nPoints;
        const auto step = (end - start) / nPoints;
        const Vector point = {start + i * step, start + j * step, 0};

        Vector source2;
        {
            backTrack(point, wave_vector2, center2, source2);
        }
        const auto dist2 = dist(point, source2);
        auto w2 = calcWave2(dist2, source2[0], source2[1]);

        Vector source1;
        {
            backTrack(point, wave_vector1, center1, source1);
        }
        const auto dist1 = dist(point, source1);
        auto w1 = calcWave1(dist1, source1[0], source1[1]);

        ampl1 = w1.ampl;
        ampl2 = w2.ampl;
        deltaPhase = w1.phase - w2.phase;
    }

    const int totalPoints = nPoints * nPoints;
    for (int iFrame = 0; iFrame < nFrames; ++iFrame) {
        double time = 2 * M_PI * iFrame / nFrames;
        int ind = iFrame * totalPoints;
        auto img = image + ind;
        const double intens = calcIntens(
            ampl1, 
            ampl2, 
            deltaPhase + omega * time
        );
        
        atomicAdd(totIntens + iFrame, intens);
        img[k] = static_cast<uint8_t>(255.0 * intens / 4.0);
    }
}

__global__ void calc_image(
    double start, double end, int n_steps,
    const double* vector1, const double*  cnt1, double radius1,
    const double* vector2, const double*  cnt2, double radius2,
    int nFrames, double lambda, double omega, int hasInterference,
    uint8_t* image, double* totIntens)
{
    Vector v1 = {vector1[0], vector1[1], vector1[2]};
    Vector c1 = {cnt1[0], cnt1[1], cnt1[2]};
    Vector v2 = {vector2[0], vector2[1], vector2[2]};
    Vector c2 = {cnt2[0], cnt2[1], cnt2[2]};

    calcImage(
      start, end, n_steps,
      v1, c1, radius1,
      v2, c2, radius2,
      nFrames, lambda, omega, hasInterference,
      image, totIntens);
}
""", arch='sm_70')


def calc_image(
        start, end, n_points,
        wave_vector1, center1, radius1,
        wave_vector2, center2, radius2,
        n_frames, lamb, omega, has_interf,
        noise_coef, block_size=64):  # number of threads per block

    result = np.zeros(n_frames * n_points * n_points, dtype=np.uint8)
    tot_intens = np.zeros(n_frames, dtype=np.float64)
    n = n_points ** 2
    n_blocks = int(n / block_size)  # value determine by block size and total work

    impl = mod.get_function("calc_image")

    assert noise_coef == 0, 'not implemented'
    # see https://stackoverflow.com/questions/46169633/how-to-generate-random-number-inside-pycuda-kernel/46181257#46181257

    impl(
        np.float64(start), np.float64(end), np.int32(n_points),
        drv.In(wave_vector1), drv.In(center1), np.float64(radius1),
        drv.In(wave_vector2), drv.In(center2), np.float64(radius2),
        np.int32(n_frames), np.float64(lamb), np.float64(omega), np.int32(has_interf),
        drv.Out(result), drv.Out(tot_intens),
        block=(block_size, 1, 1), grid=(n_blocks, 1))

    result = result.reshape(n_frames, n_points, n_points)

    # to uint8
    # im_min, im_max = 0, 4

    # result = 255.0 * (result - im_min) / (im_max - im_min)
    # result = result.astype(np.uint8)

    return result, tot_intens

