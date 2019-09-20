from __future__ import division
import os
import numpy as np
from skimage import io, morphology, exposure, transform, color
from skimage.measure import label, regionprops
import random
import math
import matplotlib.pyplot as plt
import time

class efros:
    def __init__(self, window_size):
        self.window_size = window_size
        self.half_window = window_size // 2
        self.pixels_window = window_size ** 2
        self.seed_size = 3
        self.max_error_threshold = 0.3
        self.error_threshold = 0.1
        self.sigma = self.window_size / 6.4
        self.output_path = "./output/"
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)

    def synthsis(self, img_path, new_img_row, new_img_col, output):
        start_time = time.time()

        img_data = io.imread(img_path)
        img_data = img_data / 255
        img_row, img_col = np.shape(img_data)

        seed_row = random.randint(0, img_row - self.seed_size)
        seed_col = random.randint(0, img_col - self.seed_size)
        seed_data = img_data[seed_row:seed_row + self.seed_size, seed_col:seed_col + self.seed_size]

        new_img = np.zeros((new_img_row, new_img_col))
        number_pixel = new_img_row * new_img_col
        interval = number_pixel // 100
        new_img[new_img_row//2:new_img_row//2 + self.seed_size, new_img_col//2:new_img_col//2 + self.seed_size] = seed_data

        number_filled = self.seed_size ** 2
        is_filled = np.zeros((new_img_row, new_img_col))
        is_filled[new_img_row//2:new_img_row//2 + self.seed_size, new_img_col//2:new_img_col//2 + self.seed_size] = np.ones((self.seed_size, self.seed_size))

        all_windows = self.samples(img_data)
        new_image_padded = np.lib.pad(new_img, self.half_window, 'constant', constant_values = 0)
        is_filled_padded = np.lib.pad(is_filled, self.half_window, 'constant', constant_values = 0)

        gaussi_mask = self.gaussiMask()

        while number_filled < number_pixel:
            progress = 0
            next_row, next_col = np.nonzero(morphology.binary_dilation(is_filled) - is_filled)
            neighbor = []
            for i in range(len(next_row)):
                row = next_row[i]
                col = next_col[i]
                neighbor.append(np.sum(is_filled[row - self.half_window:row + self.half_window + 1, col - self.half_window:col + self.half_window + 1]))
            arrange = np.argsort(-np.array(neighbor, dtype = int))
            for x, index in enumerate(arrange):
                row = next_row[index]
                col = next_col[index]
                best_matches = self.find_matches(new_image_padded[row:row + 2 * self.half_window + 1, col:col + 2 * self.half_window + 1],
                all_windows,
                is_filled_padded[row:row + 2 * self.half_window + 1, col:col + 2 * self.half_window + 1],
                gaussi_mask)
                pick = random.randint(0, len(best_matches) - 1)
                if best_matches[pick][0] <= self.max_error_threshold:
                    new_image_padded[row + self.half_window][col + self.half_window] = best_matches[pick][1]
                    new_img[row][col]=best_matches[pick][1]
                    is_filled_padded[row + self.half_window][col + self.half_window] = 1
                    is_filled[row][col] = 1
                    number_filled += 1
                    if number_filled % interval == 0:
                        print("Pixels filled " + str(number_filled) + "/" + str(number_pixel) + " | " + str(number_filled/interval) + "% | Time = " + str(time.time() - start_time) + " seconds")
                    progress = 1
            if progress == 0:
                self.max_error_threshold *= 1.1
        io.imsave(self.output_path + output, new_img)

    def inpainting(self, img_path, output):
        start_time = time.time()
        img_data = io.imread(img_path)
        img_row, img_col = np.shape(img_data)
        number_pixel = img_row * img_col
        interval = number_pixel // 100

        img_data = img_data / 255
        is_filled = np.ceil(img_data)
        number_filled = np.sum(is_filled)

        all_windows = self.samples_black(img_data, is_filled)
        new_img = img_data
        new_image_padded = np.lib.pad(new_img, self.half_window, 'constant', constant_values = 0)
        is_filled_padded = np.lib.pad(is_filled, self.half_window, 'constant', constant_values = 0)

        gaussi_mask = self.gaussiMask()
        while number_filled < number_pixel:
            progress = 0
            next_row, next_col = np.nonzero(morphology.binary_dilation(is_filled) - is_filled)
            neighbor = []
            for i in range(len(next_row)):
                row = next_row[i]
                col = next_col[i]
                neighbor.append(np.sum(is_filled[row - self.half_window:row + self.half_window + 1, col - self.half_window:col + self.half_window + 1]))
            arrange = np.argsort(-np.array(neighbor, dtype = int))
            for x, index in enumerate(arrange):
                row = next_row[index]
                col = next_col[index]
                best_matches = self.find_matches(new_image_padded[row:row + 2 * self.half_window + 1, col:col + 2 * self.half_window + 1],
                all_windows,
                is_filled_padded[row:row + 2 * self.half_window + 1, col:col + 2 * self.half_window + 1],
                gaussi_mask)
                pick = random.randint(0, len(best_matches) - 1)
                if best_matches[pick][0] <= self.max_error_threshold:
                    new_image_padded[row + self.half_window][col + self.half_window] = best_matches[pick][1]
                    new_img[row][col]=best_matches[pick][1]
                    is_filled_padded[row + self.half_window][col + self.half_window] = 1
                    is_filled[row][col] = 1
                    number_filled += 1
                    if number_filled % interval == 0:
                        print("Pixels filled " + str(number_filled) + "/" + str(number_pixel) + " | " + str(number_filled/interval) + "% | Time = " + str(time.time() - start_time) + " seconds")
                    progress = 1
            if progress == 0:
                self.max_error_threshold *= 1.1
        io.imsave(self.output_path + output, new_img)
        
    def removal(self, img_path, mask_path, output):
        start_time = time.time()
        img_data = io.imread(img_path)
        img_data = color.rgb2gray(img_data)
        img_data = img_data / 255
        img_row, img_col = np.shape(img_data)
        number_pixel = img_row * img_col
        interval = number_pixel // 100
        
        mask_data = io.imread(mask_path)
        mask_data = color.rgb2gray(mask_data)
        mask_data = mask_data / 255
        mask_data = np.ceil(mask_data)
        is_filled = np.ones((img_row, img_col)) - mask_data
        number_filled = np.sum(is_filled)
        
        all_windows = self.samples_black(img_data, is_filled)
        new_img = img_data
        new_image_padded = np.lib.pad(new_img, self.half_window, 'constant', constant_values = 0)
        is_filled_padded = np.lib.pad(is_filled, self.half_window, 'constant', constant_values = 0)
        
        gaussi_mask = self.gaussiMask()
        
        while number_filled < number_pixel:
            progress = 0
            next_row, next_col = np.nonzero(morphology.binary_dilation(is_filled) - is_filled)
            neighbor = []
            for i in range(len(next_row)):
                row = next_row[i]
                col = next_col[i]
                neighbor.append(np.sum(is_filled[row - self.half_window:row + self.half_window + 1, col - self.half_window:col + self.half_window + 1]))
            arrange = np.argsort(-np.array(neighbor, dtype = int))
            for x, index in enumerate(arrange):
                row = next_row[index]
                col = next_col[index]
                best_matches = self.find_matches(new_image_padded[row:row + 2 * self.half_window + 1, col:col + 2 * self.half_window + 1],
                all_windows,
                is_filled_padded[row:row + 2 * self.half_window + 1, col:col + 2 * self.half_window + 1],
                gaussi_mask)
                pick = random.randint(0, len(best_matches) - 1)
                if best_matches[pick][0] <= self.max_error_threshold:
                    new_image_padded[row + self.half_window][col + self.half_window] = best_matches[pick][1]
                    new_img[row][col]=best_matches[pick][1]
                    is_filled_padded[row + self.half_window][col + self.half_window] = 1
                    is_filled[row][col] = 1
                    number_filled += 1
                    print number_filled
                    if number_filled % interval == 0:
                        print("Pixels filled " + str(number_filled) + "/" + str(number_pixel) + " | " + str(number_filled/interval) + "% | Time = " + str(time.time() - start_time) + " seconds")
                    progress = 1
            if progress == 0:
                self.max_error_threshold *= 1.1
        io.imsave(self.output_path + output, new_img)
        

    def samples_black(self, img_data, is_filled):
        window_matrix = []
        for i in range(self.half_window, img_data.shape[0] - self.half_window - 1):
            for j in range(self.half_window, img_data.shape[1] - self.half_window - 1):
                if 0 not in is_filled[i - self.half_window:i + self.half_window + 1, j - self.half_window: j + self.half_window + 1]:
                    window_matrix.append(np.reshape(img_data[i - self.half_window:i + self.half_window + 1, j - self.half_window: j + self.half_window + 1], self.window_size ** 2))
        return np.double(window_matrix)


    def samples(self, img_data):
        window_matrix = []
        for i in range(self.half_window, img_data.shape[0] - self.half_window - 1):
            for j in range(self.half_window, img_data.shape[1] - self.half_window - 1):
                window_matrix.append(np.reshape(img_data[i - self.half_window:i + self.half_window + 1, j - self.half_window: j + self.half_window + 1], self.window_size ** 2))
        return np.double(window_matrix)

    def gaussiMask(self):
        shape = (self.window_size, self.window_size)
        m, n = [(ss - 1.) / 2 for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp( -(x * x + y * y) / (2. * self.sigma * self.sigma) )
        h[ h < np.finfo(h.dtype).eps * h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def find_matches(self, template, sample_image, valid_mask, gaussi_mask):
        template = np.reshape(template, self.pixels_window)
        valid_mask = np.reshape(valid_mask, self.pixels_window)
        gaussi_mask = np.reshape(gaussi_mask, self.pixels_window)
        total_weight = np.sum(np.multiply(gaussi_mask, valid_mask))
        distance = (sample_image - template) ** 2
        ssd = np.sum((distance * gaussi_mask * valid_mask) / total_weight, axis = 1)
        min_error = min(ssd)
        return [[err, sample_image[i][self.pixels_window // 2]] for i, err in enumerate(ssd) if err <= min_error * (1 + self.error_threshold)]
