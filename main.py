from efros import efros
from exemplarBasedInpainting import Removal
from skimage import io, morphology, exposure, transform, color
import time

def img_synthsis():
    img_names = ["T1.gif", "T2.gif", "T3.gif", "T4.gif", "T5.gif"]
    path = "./images/"
    result = open("Synthsis.txt", "wb")
    for img in img_names:
        for i in [5, 9, 11, 15, 21]:
            start = time.time()
            img_synthsis = efros(i)
            img_synthsis.synthsis(str(path + img), 200, 200, img.split('.')[0]+"_"+str(i)+".gif")
            end = time.time()
            result.write("Texture :" + img + " Windows Size :" + str(i) + " Time :" + str(end - start)+ " seconds")
            print "Finished in "+str(end-start)+" seconds"
    result.close()

def img_inpainting():
    img_names = ["test_im1.bmp", "test_im2.bmp"]
    path = "./images/"
    result = open("Inpainting.txt", "wb")
    for img in img_names:
        for i in [5, 9, 11]:
            start = time.time()
            img_inpainting = efros(i)
            img_inpainting.inpainting(str(path+img), img.split('.')[0]+"_"+str(i)+".bmp")
            end = time.time()
            result.write("Image :" + img + " Windows Size :" + str(i) + " Time :" + str(end - start)+ " seconds")
            print "Finished in "+str(end-start)+" seconds"
    result.close()

def exemplar_inpainting():
    img = "test_im3.jpg"
    input_path = "./images/"
    mask_path = "./images/test_im3_mask.jpg"
    img_data = io.imread(input_path + img)
    mask_data = io.imread(mask_path)
    mask_data = color.rgb2gray(img_data)
    start = time.time()
    object_removal = Removal(img_data, mask_data, 4)
    if object_removal.checkValidInputs()== object_removal.CHECK_VALID:
        object_removal.removal()
        io.imsave("./images/test_im3_removal.jpg", object_removal.result)
    else:
        print 'Error: invalid parameters.'
    end = time.time()
    print "Finished in "+str(end-start)+" seconds"
    
    start = time.time()
    img_removal = efros(9)
    img_removal.removal(input_path + img, mask_path, img.split('.')[0]+"_"+str(9)+".bmp")
    end = time.time()
    print "Finished in "+str(end-start)+" seconds"
    

def main():
    img_synthsis()
    img_inpainting()
    exemplar_inpainting()


if __name__ == "__main__":
    main()
