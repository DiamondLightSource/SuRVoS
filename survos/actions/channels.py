

import numpy as np
import logging as log

from scipy import ndimage as ndi
from itertools import combinations_with_replacement

from ..core import DataModel
from ..lib.convolutions import gconvssh, make_gaussian_1d
from ..lib._preprocess import tvdenoising
from ..lib._channels import symmetric_eigvals3S_gpu

### RAW CLAMPED DATA

def compute_threshold(data=None, params=None):
    return data

def compute_inv_threshold(data=None, params=None):
    return -data

### DENOISING

def compute_gaussian(data=None, params=None):
    sz, sy, sx = params['Sigma']
    kz = make_gaussian_1d(sz, order=0, trunc=3)
    ky = make_gaussian_1d(sy, order=0, trunc=3)
    kx = make_gaussian_1d(sx, order=0, trunc=3)

    log.info('+ Padding data')
    d, h, w = kz.size//2, ky.size//2, kx.size//2
    data = np.pad(data, ((d,d),(h,h),(w,w)), mode='reflect')

    log.info('   - Computing gaussian (radius={})'.format((d, h, w)))
    gauss = gconvssh(data, kz, ky, kx)

    return gauss

def compute_tvdenoise(data=None, params=None):
    return tvdenoising(data, params['Lambda'], spacing=params['Spacing'],
                       max_iter=params['Max Iter'])

### LOCAL STATISTICS

def compute_local_mean(data=None, params=None):
    log.info('+ Padding data')
    d, h, w = params['Radius']
    data = np.pad(data, ((d,d),(h,h),(w,w)), mode='reflect')

    log.info('+ Computing local mean features (radius={})'.format((d,h,w)))
    kernelz = np.ones(d * 2 + 1, np.float32)
    kernelz /= kernelz.sum()
    kernely = np.ones(h * 2 + 1, np.float32)
    kernely /= kernely.sum()
    kernelx = np.ones(w * 2 + 1, np.float32)
    kernelx /= kernelx.sum()

    features = gconvssh(data, kernelz, kernely, kernelx)

    return features

def compute_local_std(data=None, params=None):
    log.info('+ Padding data')
    d, h, w = params['Radius']
    data = np.pad(data, ((d,d),(h,h),(w,w)), mode='reflect')

    log.info('+ Computing local mean features (radius={})'.format((d,h,w)))
    kernelz = np.ones(d * 2 + 1, np.float32)
    kernelz /= kernelz.sum()
    kernely = np.ones(h * 2 + 1, np.float32)
    kernely /= kernely.sum()
    kernelx = np.ones(w * 2 + 1, np.float32)
    kernelx /= kernelx.sum()

    log.info('   - Computing local mean squared')
    meansq = gconvssh(data, kernelz, kernely, kernelx)
    meansq **= 2

    log.info('   - Computing local squared mean')
    data **= 2
    sqmean = gconvssh(data, kernelz, kernely, kernelx)
    sqmean -= meansq
    np.maximum(sqmean, 0, out=sqmean)
    np.sqrt(sqmean, sqmean) # inplace

    return sqmean

def compute_local_centering(data=None, params=None):
    mean = compute_local_mean(data=data, params=params)
    return data - mean

def compute_local_norm(data=None, params=None):
    mean = compute_local_mean(data=data, params=params)
    std = compute_local_std(data=data, params=params)
    mask = ~np.isclose(std, 0)
    result = np.zeros_like(data)
    result[mask] = (data[mask] - mean[mask]) / std[mask]
    return result

def compute_local_magnitude(data=None, params=None):
    gz, gy, gx = np.gradient(data)
    mag = np.sqrt(gz**2 + gy**2 + gx**2)
    mean = compute_local_mean(data=mag, params=params)
    return mean

### GAUSSIAN

def compute_gaussian_centering(data=None, params=None):
    g1 = compute_gaussian(data=data, params=params)
    return data - g1

def compute_gaussian_normalization(data=None, params=None):
    g1 = compute_gaussian(data=data, params=params)
    num = data - g1
    g2 = compute_gaussian(data=num**2, params=params)
    den = np.sqrt(g2)
    mask = ~np.isclose(den, 0)
    result = np.zeros_like(data)
    result[mask] = num[mask] / den[mask]
    return result

def compute_gaussian_magnitude(data=None, params=None):
    sz, sy, sx = params['Sigma']
    out = []

    for oz, oy, ox in [(1,0,0),(0,1,0),(0,0,1)]:
        szz = sz if oz == 0 else sz * 3
        syy = sy if oy == 0 else sy * 3
        sxx = sx if ox == 0 else sx * 3
        kz = make_gaussian_1d(szz, order=oz, trunc=3)
        ky = make_gaussian_1d(syy, order=oy, trunc=3)
        kx = make_gaussian_1d(sxx, order=ox, trunc=3)

        log.info('+ Padding data')
        d, h, w = kz.size//2, ky.size//2, kx.size//2
        tmp = np.pad(data, ((d,d),(h,h),(w,w)), mode='reflect')

        log.info('   - Computing convolutions (radius={})'.format((d, h, w)))
        gauss = gconvssh(tmp, kz, ky, kx)
        out.append(gauss)

    out = np.sqrt(out[0]**2 + out[1]**2 + out[2]**2)

    return out

### BLOB DETECTION

def compute_difference_gaussians(data=None, params=None):
    sigma1 = params['Sigma']
    sigma2 = tuple(np.asarray(sigma1) * params['Sigma Ratio'])
    if 'Response' in params and params['Response'] == 'Dark':
        data *= -1
    g1 = compute_gaussian(data=data, params=dict(Sigma=sigma1))
    g2 = compute_gaussian(data=data, params=dict(Sigma=sigma2))
    response = g1 - g2
    if 'Threshold' in params and params['Threshold']:
        response[response < 0] = 0
    return response


def compute_laplacian_gaussian(data=None, params=None):
    sz, sy, sx = params['Sigma']
    out = np.zeros_like(data)

    for i, (oz, oy, ox) in enumerate([(2,0,0),(0,2,0),(0,0,2)]):
        kz = make_gaussian_1d(sz, order=oz, trunc=3)
        ky = make_gaussian_1d(sy, order=oy, trunc=3)
        kx = make_gaussian_1d(sx, order=ox, trunc=3)

        if 'Response' in params and params['Response'] == 'Bright':
            if i == 0: kz *= -1
            if i == 1: ky *= -1
            if i == 2: kx *= -1

        log.info('+ Padding data')
        d, h, w = kz.size//2, ky.size//2, kx.size//2
        tmp = np.pad(data, ((d,d),(h,h),(w,w)), mode='reflect')

        log.info('   - Computing convolutions (radius={})'.format((d, h, w)))
        gauss = gconvssh(tmp, kz, ky, kx)
        out += gauss

    if 'Threshold' in params and params['Threshold']:
        out[out < 0] = 0

    return out


### HESSIAN

def compute_hessian(data=None, params=None):
    log.info('+ Computing Hessian Matrix')
    gaussian_filtered = compute_gaussian(data=data, params=params)
    gradients = [np.gradient(gaussian_filtered, axis=i) for i in range(3)]
    axes = range(data.ndim)
    H_elems = [np.gradient(gradients[2-ax0], axis=2-ax1)
               for ax0, ax1 in combinations_with_replacement(axes, 2)]
    sigma = max(params['Sigma'])
    if sigma > 1:
        H_elems = [elem * sigma**2 for elem in H_elems]
    return H_elems

def compute_hessian_determinant(data=None, params=None):
    Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = compute_hessian(data=data, params=params)

    log.info('+ Computing Hessian Determinant')
    det =  Hxx * (Hyy*Hzz - Hyz*Hyz) \
         - Hxy * (Hxy*Hzz - Hyz*Hxz) \
         + Hxz * (Hxy*Hyz - Hyy*Hxz)
    if 'Response' in params and params['Response'] == 'Bright':
        return -det
    return det

def hessian_eigvals(data=None, params=None, correct=False, doabs=False): # TODO: GPU THIS
    H = compute_hessian(data=data, params=params)
    if correct:
        s = max(params['Sigma'])**2
        Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = [h * s for h in H]
    else:
        Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = H
    log.info('+ Computing Hessian Eigenvalues')
    R = symmetric_eigvals3S_gpu(Hxx, Hxy, Hxz, Hyy, Hyz, Hzz, doabs=doabs)
    return R

def compute_hessian_eigvals(data=None, params=None, correct=False):
    R = hessian_eigvals(data=data, params=params, correct=correct)
    return R[..., params['Eigen Value']].copy()

### STRUCTURE TENSOR

def compute_structure_tensor(data=None, params=None):
    log.info('+ Computing Structure Tensor')
    p1 = dict(Sigma=params['Sigma Deriv'])
    p2 = dict(Sigma=params['Sigma Area'])

    gradients = np.gradient(data)
    H_elems = [compute_gaussian(gradients[2-ax0] * gradients[2-ax1], params=p2)
               for ax0, ax1 in combinations_with_replacement(range(3), 2)]
    return H_elems


def compute_structure_tensor_determinant(data=None, params=None):
    Sxx, Sxy, Sxz, Syy, Syz, Szz = compute_structure_tensor(data=data, params=params)

    log.info('+ Computing Structure Tensor Determinant')
    return  Sxx * (Syy*Szz - Syz*Syz) \
          - Sxy * (Sxy*Szz - Syz*Sxz) \
          + Sxz * (Sxy*Syz - Syy*Sxz)


def compute_structure_tensor_eigvals(data=None, params=None):
    Sxx, Sxy, Sxz, Syy, Syz, Szz = compute_structure_tensor(data=data, params=params)
    log.info('+ Computing Structure Tensor Eigenvalues')
    R = symmetric_eigvals3S_gpu(Sxx, Sxy, Sxz, Syy, Syz, Szz)
    return R[..., params['Eigen Value']].copy()


def compute_gaussian_scale_invariant(data=None, params=None):
    sigma = np.asarray(params['Init Sigma'])
    ratio = params['Sigma Ratio']
    num_scales = params['Num Scales']
    response = params['Response']

    result = compute_gaussian(data=data, params=dict(Sigma=sigma))

    for i in range(num_scales):
        sigma *= ratio
        tmp = compute_gaussian(data=data, params=dict(Sigma=sigma))
        if response == 'Min':
            result = np.minimum(result, tmp)
        elif response == 'Max':
            result = np.maximum(result, tmp)
        else:
            result += tmp

    if response == 'Avg':
        result /= num_scales

    return result

### SCALE INVARIANT DoG

def compute_sidog(data=None, params=None):
    sigma = np.asarray(params['Init Sigma'], dtype=np.float32)
    max_sigma = params['Max Sigma']
    sratio = params['Sigma Ratio']
    response = params['Response']

    result = np.full(data.shape, -np.inf, data.dtype)

    if 'Response' in params and params['Response'] == 'Dark':
        data *= -1

    gauss = compute_gaussian(data, params=dict(Sigma=sigma))

    while max(sigma) * sratio < max_sigma:
        prev_sigma = sigma.copy()

        sigma *= sratio
        gauss1 = compute_gaussian(data, params=dict(Sigma=sigma))

        tmp = (gauss - gauss1) * max(prev_sigma)
        np.maximum(result, tmp, out=result)
        gauss = gauss1

    if 'Threshold' in params and params['Threshold']:
        result[result < 0] = 0

    return result

### SCALE INVARIANT LAPLACIAN

def compute_silaplacian(data=None, params=None):
    sigma = np.asarray(params['Init Sigma'], dtype=np.float32)
    max_sigma = params['Max Sigma']
    sincr = params['Sigma Incr']

    result = np.full(data.shape, -np.inf, data.dtype)

    while max(sigma) < max_sigma:
        params = dict(Sigma=sigma, Response=params['Response'])
        tmp = compute_laplacian_gaussian(data=data, params=params) * max(sigma)**2
        np.maximum(result, tmp, out=result)
        sigma += sincr

    if 'Threshold' in params and params['Threshold']:
        result[result < 0] = 0

    return result

### SCALE INVARIANT HESSIAN DETERMINANT

def compute_sihessian_det(data=None, params=None):
    sigma = np.asarray(params['Init Sigma'], dtype=np.float32)
    max_sigma = params['Max Sigma']
    sincr = params['Sigma Incr']

    result = np.full(data.shape, -np.inf, data.dtype)

    while max(sigma) < max_sigma:
        params = dict(Sigma=sigma, Response=params['Response'])
        tmp = compute_hessian_determinant(data=data, params=params)
        np.maximum(result, tmp, out=result)
        sigma += sincr

    return result

### FRANGI FILTER

def compute_frangi(data=None, params=None):
    sigma = np.asarray(params['Init Sigma'], dtype=np.float32)
    max_sigma = params['Max Sigma']
    sincr = params['Sigma Incr']

    result = None

    while max(sigma) < max_sigma:
        R = hessian_eigvals(data=data, params=dict(Sigma=sigma),
                            correct=True, doabs=True)
        e1 = R[..., 0]
        e2 = R[..., 1]
        e3 = R[..., 2]

        ae1 = np.abs(e1)
        ae2 = np.abs(e2)
        ae3 = np.abs(e3)

        ae1sq = ae1 * ae1
        ae2sq = ae2 * ae2
        ae3sq = ae3 * ae3

        Ra = (ae2sq / ae3sq)
        Rb = (ae1sq / (ae2 * ae3))
        S = ae1sq + ae2sq + ae3sq

        A = B = 2 * (params['Lamda']**2)
        C = 2 * S.max()

        expRa = 1 - np.exp(-Ra / A)
        expRb =     np.exp(-Rb / B)
        expS  = 1 - np.exp(-S  / C)

        tmp = expRa * expRb * expS

        if params['Response'] == 'Dark':
            tmp[e2 < 0] = 0; tmp[e3 < 0] = 0;
        else:
            tmp[e2 > 0] = 0; tmp[e3 > 0] = 0;

        tmp[np.isnan(tmp)] = 0

        if result is None:
            result = tmp
        else:
            np.maximum(result, tmp, out=result)

        sigma += sincr

    return result


def compute_relu(data=None, params=None):
    rtype = params['Type']
    alpha = params['Alpha']

    if rtype == 'Standard':
        np.maximum(0, data, out=data)
    elif rtype == 'Noisy':
        gnoise = np.random.randn(*data.shape).astype(np.float32)
        np.maximum(0, data + alpha * gnoise, out=data)
    else:
        data[data < 0] *= alpha

    return data

###############################################################################
# MAIN FUNC
###############################################################################

fnmap = {
    'thresh'            : compute_threshold,
    'inv_thresh'        : compute_inv_threshold,

    'gauss'             : compute_gaussian,
    'tv'                : compute_tvdenoise,

    'local_mean'        : compute_local_mean,
    'local_std'         : compute_local_std,
    'local_center'      : compute_local_centering,
    'local_norm'        : compute_local_norm,
    'local_mag'         : compute_local_magnitude,

    'gauss_center'      : compute_gaussian_centering,
    'gauss_norm'        : compute_gaussian_normalization,
    'gauss_mag'         : compute_gaussian_magnitude,

    'dog'               : compute_difference_gaussians,
    'log'               : compute_laplacian_gaussian,

    'hessian_det'       : compute_hessian_determinant,
    'hessian_eig'       : compute_hessian_eigvals,

    'structure_det'     : compute_structure_tensor_determinant,
    'structure_eig'     : compute_structure_tensor_eigvals,

    'gauss_scale3d'     : compute_gaussian_scale_invariant,
    'si_dog'            : compute_sidog,
    'si_laplacian'      : compute_silaplacian,
    'si_hessian_det'    : compute_sihessian_det,
    'frangi'            : compute_frangi,

    'relu'              : compute_relu
}

def compute_channel(source=None, clamp=None, feature=None,
                    idx=None, name=None, params=None, out=None):
    result = None

    log.info('+ Loading data into memory')
    DM = DataModel.instance()
    data = DM.load_slices(source)

    if clamp is not None:
        log.info('+ Clamping data')
        t1, t2 = clamp
        data[data < t1] = t1
        data[data > t2] = t2
        data -= t1
        data /= (t2 - t1)
    elif feature != 'relu':
        log.info('+ Rescaling data')
        data -= data.min()
        data /= data.max()

    if feature in fnmap:
        log.info('+ Computing feature {}'.format(feature))
        result = fnmap[feature](data=data, params=params)

        log.info('+ Normalizing feature')
        result -= result.mean()
        std = result.std()
        if std >= 1e-5:
            result /= std

        log.info('+ Calculating feature statistics')
        amin, amax = result.min(), result.max()
        evmin, evmax = np.percentile(result, (1., 99.))

        log.info('+ Saving feature into disk {}'.format(result.shape))
        params['source'] = source
        params['feature_idx'] = idx
        params['feature_name'] = name
        params['feature_type'] = feature
        params['vmin'] = amin
        params['vmax'] = amax
        params['evmin'] = evmin
        params['evmax'] = evmax
        params['default_evmin'] = evmin
        params['default_evmax'] = evmax
        params['active'] = True

        DM.write_slices(out, result, params=params)

        return out, idx, params

    return None


def compute_all_channel(features=None):
    output = []
    for feature in features:
        log.info('\n## Computing Feature {}'.format(feature['feature']))
        output.append(compute_channel(**feature))
    return output
