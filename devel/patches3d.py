import numpy as np
from matplotlib import pyplot as plt
import skimage
import scipy
from loguru import logger
import datetime
import copy
import imma
import io3d


def cart2pol(x, y, x0=0, y0=0):
    rho = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    phi = np.arctan2((y - y0), (x - x0))
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


# %%

# %%


# patch_background_density_hu =

# patch = np.zeros(patch_size_px)


# center1 = (100, 256)
# radius1 = 20
def add_circle(img, center1, radius1, intensity=1):
    x, y = np.indices(img.shape)
    mask = ((x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2).astype(bool)
    img[mask == True] = intensity
    return img


def add_circle_mm(img, voxelsize_mm_scalar, center_mm, radius_mm, intensity_hu):
    return add_circle(img, np.asarray(center_mm) / voxelsize_mm_scalar, radius_mm / voxelsize_mm_scalar, intensity_hu)


def make_patch(voxelsize_mm_scalar=.1, patch_size_mm=[40, 40], patch_object_density_hu=400,
               patch_object_center_density_hu=800, background_density_hu=-1024):
    voxelsize_mm = [voxelsize_mm_scalar] * 3
    patch_size_px = (np.asarray(patch_size_mm) / voxelsize_mm_scalar).astype(int)

    patch = np.ones(patch_size_px) * background_density_hu
    patch = add_circle_mm(patch, voxelsize_mm_scalar, (12.0, 10.0), 5.0, patch_object_density_hu)
    patch = add_circle_mm(patch, voxelsize_mm_scalar, (12.0, 10.0), 2.0, patch_object_center_density_hu)
    patch = add_circle_mm(patch, voxelsize_mm_scalar, (12.0, 30.0), 5.0, patch_object_density_hu)
    patch = add_circle_mm(patch, voxelsize_mm_scalar, (12.0, 30.0), 2.0, patch_object_center_density_hu)
    patch = add_circle_mm(patch, voxelsize_mm_scalar, (28.0, 20.0), 5.0, patch_object_density_hu)
    patch = add_circle_mm(patch, voxelsize_mm_scalar, (28.0, 20.0), 2.0, patch_object_center_density_hu)
    return patch


# %%

# %%
# voxelsize_mm = [.5, .5, .5]

# colon_radius

# patch_
# patch3d = np.ones(patch_3ď_shape) * background_density_hu
# for phi in range(0, patch.shape[0]):
#     x, y = pol2cart(patch_radius, phi)
#     patch3d[int(x), int(y), :] = patch[phi,:]

# patch3d.max()
# %%
# for i in range(patch.shape[0]):
#     patch[i, :] = i
# %%


def place_patch_on_tube(patch, patch_thickness_mm, tube_radius_mm, voxelsize_mm_scalar, patch3d=None,
                        background_density_hu=-1024):
    voxelsize_mm = [voxelsize_mm_scalar] * 3
    patch_pixelsize_mm = voxelsize_mm[:2]
    color_radius = tube_radius_mm
    rhos = []
    phis = []

    if not patch3d:
        patch_3ď_shape = [patch.shape[0], patch.shape[0], patch.shape[1]]
        patch3d = np.ones(patch_3ď_shape) * background_density_hu

    for x in range(0, patch3d.shape[0]):
        for y in range(0, patch3d.shape[1]):

            rho, phi = cart2pol(x * voxelsize_mm[0], y * voxelsize_mm[1], patch.shape[0] * voxelsize_mm[0] / 2,
                                patch.shape[1] * voxelsize_mm[1] / 2)
            phi = phi + np.pi

            rhos.append(rho)
            phis.append(phi)

            if (rho > (tube_radius_mm - patch_thickness_mm)) and (rho <= tube_radius_mm):
                # patch3d[x,y,:] = 1

                patch_row_px = int(tube_radius_mm * phi / patch_pixelsize_mm[0])
                if (patch_row_px >= 0 and patch_row_px < patch.shape[0]):
                    patch3d[x, y, :] = patch[patch_row_px, :]

    return patch3d


#             # phi*colon_radius_mm
#             # 3
#
#             # print(patch_row_px, end="")
#             if (patch_row_px >= 0 and patch_row_px < patch_size_mm[1]):
#                 # patch3d[x, y,:] = patch[int(phi/(2*np.pi) * patch.shape[1]),:]

#                 patch3d[x, y,:] = patch[patch_row_px,:]
#                 # print(" ok")
#             else:
#                 # print("")
#                 pass

# # %%
# !pip
# install
# io3d
# pydicom
# imma
# SimpleITK
# # %%

# io3d.write(().astype(int),)
# %%
# !pip install ndnoise
# %%
# import ndnoise


# %%
def noises(shape, sample_spacing=None, exponent=0, lambda0=0, lambda1=1, method="space", **kwargs):
    """ Create noise based on space paramters.

    :param shape:
    :param sample_spacing: in space units like milimeters
    :param exponent:
    :param lambda0: wavelength of first noise
    :param lambda1: wavelength of last noise
    :param method: use "space" or "freq" method. "freq" is more precise but slower.
    :param kwargs:
    :return:
    """
    kwargs1 = dict(
        shape=shape,
        sample_spacing=sample_spacing,
        exponent=exponent,
        lambda0=lambda0,
        lambda1=lambda1,
        **kwargs
    )

    if method is "space":
        noise = noises_space(**kwargs1)
    elif method is "freq":
        noise = noises_freq(**kwargs1)
    else:
        logger.error("Unknown noise method `{}`".format(method))

    return noise


def ndimage_normalization(data, std_factor=1.0):
    t0 = datetime.datetime.now()
    data0n = (data - np.mean(data)) * 1.0 / (std_factor * np.var(data) ** 0.5)
    logger.debug(f"t_norm={datetime.datetime.now() - t0}")

    return data0n


def gaussian_filter_fft(image, sigma):
    input_ = np.fft.fftn(image)
    result = scipy.ndimage.fourier_gaussian(input_, sigma=sigma)
    result = np.fft.ifftn(result)
    return np.abs(result)


def noises_space(
        shape,
        sample_spacing=None,
        exponent=0.0,
        lambda0=0,
        lambda1=1,
        random_generator_seed=None,
        use_fft="auto",
        **kwargs
):
    """
    use_fft: ("auto", "lambda0", "lambda1", "both", "none") auto: use fft if lambda is > 5
    """

    data0 = 0
    data1 = 0
    w0 = 0
    w1 = 0
    lambda0_px = None
    lambda1_px = None
    if random_generator_seed is not None:
        np.random.seed(seed=random_generator_seed)

    use_fft_l0 = use_fft == "lambda0" or use_fft == "both"
    use_fft_l1 = use_fft == "lambda1" or use_fft == "both"
    if use_fft == "auto":
        use_fft_l0 = lambda0 > 5
        use_fft_l1 = lambda1 > 5

    # lambda1 = lambda_stop * np.asarray(sample_spacing)
    t0 = datetime.datetime.now()

    if lambda0 is not None:
        lambda0_px = lambda0 / np.asarray(sample_spacing)
        data0 = np.random.rand(*shape)
        if use_fft_l0:
            data0 = gaussian_filter_fft(data0, sigma=lambda1_px)
            pass
        else:
            data0 = scipy.ndimage.filters.gaussian_filter(data0, sigma=lambda0_px)
        data0 = ndimage_normalization(data0)
        w0 = np.exp(exponent * lambda0)
    logger.debug(f"t_l0={datetime.datetime.now() - t0}")
    t0 = datetime.datetime.now()
    if lambda1 is not None:
        lambda1_px = lambda1 / np.asarray(sample_spacing)
        data1 = np.random.rand(*shape)
        if use_fft_l1:
            data1 = gaussian_filter_fft(data1, sigma=lambda1_px)
        else:
            data1 = scipy.ndimage.filters.gaussian_filter(data1, sigma=lambda1_px)

        data1 = ndimage_normalization(data1)
        w1 = np.exp(exponent * lambda1)
    logger.debug(f"t_l1={datetime.datetime.now() - t0}")
    t0 = datetime.datetime.now()
    logger.debug("lambda_px {} {}".format(lambda0_px, lambda1_px))
    logger.debug(f"use_fft lambda 0 and 1 {use_fft_l0} {use_fft_l1}")
    wsum = w0 + w1
    if wsum > 0:
        w0 = w0 / wsum
        w1 = w1 / wsum

    # print w0, w1
    # print np.mean(data0), np.var(data0)
    # print np.mean(data1), np.var(data1)

    data = (data0 * w0 + data1 * w1)
    logger.debug("w0, w1 {} {}".format(w0, w1))

    # plt.figure()
    # plt.imshow(data0[:,:,50], cmap="gray")
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(data1[:,:,50], cmap="gray")
    # plt.colorbar()
    return data




# %%
def random_direction_vector(return_angles=False):
    """
    Get random direction vector
    :param return_angles: gives also the angles
    :return:
        vector
        or
        vector, theta, phi
    """
    xi1 = np.random.rand()
    xi2 = np.random.rand()

    # theta = np.arccos(np.sqrt(1.0-xi1))
    theta = np.arccos(1.0 - (xi1 * 1))
    phi = xi2 * 2 * np.pi

    xs = np.sin(theta) * np.cos(phi)
    ys = np.sin(theta) * np.sin(phi)
    zs = np.cos(theta)

    vector = np.asarray([xs, ys, zs])
    if return_angles:
        return vector, theta, phi
    return vector


# %%
def random_rotate_volume(data3d, background_density_hu=-1024):
    vec, theta, phi = random_direction_vector(return_angles=True)
    alpha = np.random.random() * 360
    theta = theta * 360 / (2 * np.pi)
    phi = phi * 360 / (2 * np.pi)

    data3d = scipy.ndimage.rotate(data3d, alpha, axes=(1, 0), reshape=False, order=0, mode='constant',
                                  cval=background_density_hu, prefilter=True)
    data3d = scipy.ndimage.rotate(data3d, theta, axes=(1, 2), reshape=False, order=0, mode='constant',
                                  cval=background_density_hu, prefilter=True)
    data3d = scipy.ndimage.rotate(data3d, phi, axes=(0, 2), reshape=False, order=0, mode='constant',
                                  cval=background_density_hu, prefilter=True)

    return data3d


# %%
# random_direction_vector(return_angles=True)
# %% md
# Prepare augmented patch
# %%

def insert_patch_into_volume(
        datap, patch,
        colon_radius_mm=10,
        patch_thickness_mm=1.0,
        voxelsize_mm_scalar=.1,
        background_density_hu = -1024,
        noise_intensity_hu=50
):
    """
    Put 2D patch on tube and put it into volumetric image.

    :param datap: volumetric image with structure
    :param patch:
    :param colon_radius_mm:
    :param patch_thickness_mm:
    :param voxelsize_mm_scalar:
    :param background_density_hu: is used as threshold value for additional noise
    :param noise_intensity_hu: intensity of noise added to non-background of 3D patch
    :return:
    """
    voxelsize_mm = [voxelsize_mm_scalar] * 3

    angle = np.random.random() * 360
    patch = skimage.transform.rotate(patch, angle, order=0, preserve_range=True, cval=background_density_hu)
    patch3d = place_patch_on_tube(patch, patch_thickness_mm, colon_radius_mm, voxelsize_mm_scalar)

    # noise2d = noises(
    #     [301, 302],
    #     sample_spacing=[1, 1],
    #     random_generator_seed=5,
    #     lambda0=1,
    #     lambda1=16,
    #     exponent=0,
    #     method="space"
    # )



    noise = noises(
        patch3d.shape,
        sample_spacing=voxelsize_mm,
        random_generator_seed=None,
        lambda0=0.1,
        lambda1=10,
        exponent=-1.5,
        method="space",
        # method="freq"
    )

    patch3d_noise = patch3d
    patch3d_noise[patch3d > background_density_hu] += (noise[patch3d > background_density_hu] * 50)
    patch3d_noise[patch3d_noise < background_density_hu] = background_density_hu

    # plt.figure(figsize=(10, 10))
    # plt.imshow(np.max(patch3d_noise, axis=1))
    # plt.colorbar()

    patch3d[patch3d > background_density_hu] += (noise[patch3d > background_density_hu] * noise_intensity_hu)

    patch3d = random_rotate_volume(patch3d)
    # plt.figure()
    # plt.imshow(np.max(patch3d, axis=1))

    patch3dr = imma.image.resize_to_mm(patch3d, voxelsize_mm, datap.voxelsize_mm, order=2, preserve_range=True)
    patch3dr.shape

    indst = (np.random.rand(3) * (np.asarray(datap.data3d.shape) - patch3dr.shape - 1)).astype(int)
    indsp = indst + patch3dr.shape
    slices = [slice(indst[0], indsp[0]), slice(indst[2], indsp[2]), slice(indst[2], indsp[2])]

    datap = copy.deepcopy(datap)
    datap.data3d[slices] = np.maximum(datap.data3d[slices], patch3dr)

    return datap, slices

