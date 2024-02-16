import os
from ImageKL import ImageKL

if __name__ == '__main__':
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    image_files = [os.path.join(script_path, "../data", f) for f in os.listdir(os.path.join(script_path, "../data")) if os.path.isfile(os.path.join(script_path, "../data", f))]
    quantifiers = [(8, 8, 8), (8, 8, 4), (8, 8, 0), (8, 4, 0)]
    for image_file in image_files:
        print("---------------------------------------------------")
        image = ImageKL(image_file)
        for quantifier in quantifiers:
            print("-----------------------")
            for is_yuv in [True, False]:
                print("-----------")
                print("Quantifier: ", quantifier, " is_yuv: ", is_yuv)
                coded_image = image.encode(quantifier, is_yuv)
                print("PSNR: ", image.psnr())
                print("SSIM: ", image.ssim())
                print("Compression ratio: ", image.compression_ratio())
                print("-----------")
            print("-----------------------")
        print("---------------------------------------------------")