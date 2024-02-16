from os import path
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

class ImageKL:
    def __init__(self, path_code: str):
        self.__width = 0
        self.__height = 0
        self.__quantifier = ()
        self.__compress_code = None
        self.code = self.tilde_image(path_code)

    def tilde_image(self, image_path: str):
        print("Reading....... ", image_path )
        image = plt.imread(image_path)
        self.__width = image.shape[1]
        self.__height = image.shape[0]
        return rearrange(image, 'h w c -> (h w) c')

    def psnr(self):
        return 10 * np.log10(self.__mse()) + 20 * np.log10(self.__msi())

    def ssim(self):
        covariance_images = np.cov(np.concatenate((np.reshape(self.code, (-1, 1)), np.reshape(self.__compress_code, (-1, 1))), axis=1).T)[0,1]
        variance_base = np.cov(np.reshape(self.code, (-1, 1)).T)
        variance_compressed = np.cov(np.reshape(self.__compress_code, (-1, 1)).T)
        return (2 * np.mean(self.code) * np.mean(self.__compress_code) + 0.01) * (2 * covariance_images + 0.03) / ((np.mean(self.code) ** 2 + np.mean(self.__compress_code) ** 2 + 0.01) * (variance_base + variance_compressed + 0.03))

    def compression_ratio(self):
        code_color_space = np.power(2, 8 * 3)
        compress_code_color_space = np.power(2, self.__quantifier[0]) * np.power(2, self.__quantifier[1]) * np.power(2, self.__quantifier[2])
        code_length = np.array(self.code).shape[0] * np.array(self.code).shape[1]
        compress_code = np.array(self.__compress_code).shape[0] * np.array(self.__compress_code).shape[1]
        return compress_code * compress_code_color_space / (code_length * code_color_space)

    def rgb2yuv(self, rgb):
        return np.dot(rgb, np.array([[1/4, 1/2, 1/4], [0, -1, 1], [1, -1, 0]]))

    def yuv2rgb(self, yuv):
        g = np.dot(yuv, np.array([1, -1, -1/4]))
        r = np.dot(yuv, np.array([0, 0, 1])) + g
        b = np.dot(yuv, np.array([0, 1, 0])) + g
        return np.transpose([r, g, b])

    def encode(self, quantifier=(8, 8, 8), is_yuv=False):
        if is_yuv:
            self.code = self.rgb2yuv(self.code)
        kl_transform = self.kl_transform()
        self.__quantifier = quantifier
        self.__compress_code = self.kl_transform_inv(self.reconstruct(self.quantify(quantifier, kl_transform), kl_transform))
        if is_yuv:
            self.__compress_code = self.yuv2rgb(self.__compress_code)
        return rearrange(np.clip(self.__compress_code, 0, 1), '(h w) c -> h w c', h=self.__height, w=self.__width)

    def __average(self, code):
        mean = np.mean(code, axis=0)
        average = np.zeros(code.shape)
        for i in range(code.shape[1]):
            average[:, i] = mean[i]
        return average

    def __covariant(self, code):
        return np.cov(code, rowvar=False)

    def __eigenvector(self, code):
        return np.linalg.eig(self.__covariant(code))[1]

    def __inverse_eigenvector(self, code):
        return np.linalg.inv(self.__eigenvector(code))

    def kl_transform(self):
        return np.transpose(np.matmul(self.__eigenvector(self.code), np.transpose(np.subtract(self.code, self.__average(self.code)))))

    def quantify(self, quantifier, kl_transform_outpout):
        canals = [kl_transform_outpout[:, i] for i in range(3)]
        return  [np.linspace(np.min(canals[i]), np.max(canals[i]), num=2**quantifier[i]) for i in range(len(canals))]

    def reconstruct(self, quantifier, kl_transform_outpout):
        canals = [kl_transform_outpout[:, i] for i in range(3)]
        compress_canals = [[quantifier[i][np.abs(quantifier[i] - data).argmin()] for data in canals[i]] for i in range(len(quantifier))]
        return np.transpose(np.array(compress_canals))

    def kl_transform_inv(self, kl_transform):
        print(self.__average(self.code).shape)
        return np.transpose(np.matmul(np.transpose(self.__inverse_eigenvector(self.code)), np.transpose(np.add(kl_transform, self.__average(self.code)))))

    def __msi(self):
        return max(np.power(2, self.__quantifier))

    def __mse(self):
        return 1 / np.mean((self.__compress_code - self.code) ** 2)
