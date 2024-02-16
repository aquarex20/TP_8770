import numpy as np
from ImageKL import ImageKL

class ImageKLFeat:
    def __init__(self, path_code_fit_image: str, path_code_test_image: str):
        self.__compress_code = None
        self.__quantifier = ()
        self.code = ImageKL(path_code_fit_image)
        self.test_code = ImageKL(path_code_test_image)

    def psnr(self):
        return 10 * np.log10(self.__mse()) + 20 * np.log10(self.__msi())

    def ssim(self):
        covariance_images = np.cov(np.concatenate((np.reshape(self.test_code.code, (-1, 1)), np.reshape(self.__compress_code, (-1, 1))), axis=1).T)[0,1]
        variance_base = np.cov(np.reshape(self.test_code.code, (-1, 1)).T)
        variance_compressed = np.cov(np.reshape(self.__compress_code, (-1, 1)).T)
        return (2 * np.mean(self.test_code.code) * np.mean(self.__compress_code) + 0.01) * (2 * covariance_images + 0.03) / ((np.mean(self.test_code.code) ** 2 + np.mean(self.__compress_code) ** 2 + 0.01) * (variance_base + variance_compressed + 0.03))

    def encode(self, quantifier=(8, 8, 8), is_yuv=False):
        self.__quantifier = quantifier
        if is_yuv:
            self.code = self.rgb2yuv(self.code)
            self.test_code = self.rgb2yuv(self.test_code)
        kl_transform = self.code.kl_transform()
        self.__compress_code = self.test_code.kl_transform_inv(self.test_code.reconstruct(self.code.quantify(quantifier, kl_transform), kl_transform))
        return self.__compress_code
    
    def __msi(self):
        return max(np.power(2, self.__quantifier))

    def __mse(self):
        return 1 / np.mean((self.__compress_code - self.test_code.code) ** 2)
