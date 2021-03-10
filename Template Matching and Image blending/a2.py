from PIL import Image, ImageDraw
import numpy as np
import math
import ncc
import scipy as sp

# Part 1 Q2
def MakeGaussianPyramid(image, scale, minsize):
    x = image.size[0]
    y = image.size[1]
    retval = [image]

    # while it's bigger than min_size
    while int(min(x, y) * scale) > minsize:
        # turn image into np array
        image_arr = np.asarray(image)
        # convert to float
        # image_arr = image_arr.astype()

        # apply gaussian first
        if image_arr.ndim == 2:
            # greyscale case
            smooth = sp.ndimage.gaussian_filter(image_arr, 1/(2*scale))
        elif image_arr.ndim == 3:
            # RGB case
            r = image_arr[:,:,0]
            g = image_arr[:,:,1]
            b = image_arr[:,:,2]
            # convert them to float first
            r = r.astype('float64')
            g = g.astype('float64')
            b = b.astype('float64')

            r = sp.ndimage.gaussian_filter(r, 1/(2*scale))
            g = sp.ndimage.gaussian_filter(g, 1/(2*scale))
            b = sp.ndimage.gaussian_filter(b, 1/(2*scale))

            # clip them
            #r = np.clip(r, 0, 255)
           # g = np.clip(g, 0, 255)
            #b = np.clip(b, 0, 255)

            # convert it back to uint8
            r = r.astype('uint8')
            g = g.astype('uint8')
            b = b.astype('uint8')
            smooth = np.copy(image_arr)

            smooth[:,:,0] = r
            smooth[:,:,1] = g
            smooth[:,:,2] = b
        else:
            # error
            return retval

        # now the smooth is smoothed numpy ndarray, convert it back to PIL image
        image = Image.fromarray(smooth)

        # shrink the image and add
        image = image.resize((int(x*scale), int(y*scale)), Image.BICUBIC)
        retval.append(image)

        # calculate new x and y
        x = image.size[0]
        y = image.size[1]

    return retval

# Part 1 Q3
def ShowGaussianPyramid(pyramid):
    # get the length
    length = len(pyramid)
    # height is just the max height which is first image
    width = 0
    height = pyramid[0].size[1] # this is the final height

    # add up the width
    for im in pyramid:
        width = width + im.size[0]

    # init an image
    image = Image.new("RGB", (width, height))

    offset_x = 0
    offset_y = 0

    # paste the image
    for im in pyramid:
        image.paste(im, (offset_x, offset_y))
        # update offset x, y need not change
        offset_x = offset_x + im.size[0]

    # done 
    image.show()
    return image


# Part 1 Q4
def FindTemplate(pyramid, template, threshold):
    all_array = []
    for image in pyramid:
        arr = ncc.normxcorr2D(image, template)
        all_array.append(arr)
    scale = 0.75
    original = pyramid[0]
    # convert it to rgb
    original = original.convert('RGB')
    draw = ImageDraw.Draw(original)

    # template width and height for drawing box
    temp_x = template.size[0]
    temp_y = template.size[1]
    temp_x_half = int(temp_x/2)
    temp_y_half = int(temp_y/2)
    # each element in all_array is result for each template. 
    for i in range(len(all_array)):
        corr = all_array[i] # need to scale this

        multi = 1/(scale**i)
        
        for j in range(corr.shape[0]):
            for k in range(corr.shape[1]):
                # remember, j is y and k is x
                if corr[j][k] > threshold:
                    # if it's over threshold, draw the stupid box
                    x1 = k - temp_x_half
                    x2 = k + temp_x_half
                    y1 = j - temp_y_half
                    y2 = j + temp_y_half

                    # make sure to scale
                    x1 = int(x1 * multi)
                    x2 = int(x2 * multi)
                    y1 = int(y1 * multi)
                    y2 = int(y2 * multi)

                    # draw
                    draw.line((x1, y1, x1, y2), fill='red', width=2)
                    draw.line((x2, y1, x2, y2), fill='red', width=2)
                    draw.line((x1, y1, x2, y1), fill='red', width=2)
                    draw.line((x1, y2, x2, y2), fill='red', width=2)
                    

    # now it's done.
    del draw
    original.show()
    return original
        

# Part 2 Q2
# returns: 3d numpy arrays
def MakeLaplacianPyramid(image, scale, minsize):
    # create gauss pyramid first
    gauss_pyramid = MakeGaussianPyramid(image, scale, minsize)

    # convert to RGB if need be
    for i in range(len(gauss_pyramid)):
        if gauss_pyramid[i].mode == 'L':
            gauss_pyramid[i] = gauss_pyramid[i].convert('RGB')

    # convert to RGB if need be
    if image.mode == 'L':
        image = image.convert('RGB')

    # also create pyramid without using any gaussian filtering
    pyramid = []

    for g_im in gauss_pyramid:
        pyramid.append(image.resize(g_im.size, Image.BICUBIC))

    # lets convert them to nparray
    gauss_pyramid_np = []
    for im in gauss_pyramid:
        gauss_pyramid_np.append(np.asarray(im))

    pyramid_np = []
    for im in pyramid:
        pyramid_np.append(np.asarray(im))

    retval = []

    # now for each original from pyramid and smoothed from gauss_pyramid at each levels
    for i in range(len(pyramid_np) - 1):
        #retval.append(pyramid_np[i] - gauss_pyramid_np[i])
        p = pyramid_np[i]
        g = gauss_pyramid_np[i]
        p_r = p[:,:,0]
        p_g = p[:,:,1]
        p_b = p[:,:,2]
        g_r = g[:,:,0]
        g_g = g[:,:,1]
        g_b = g[:,:,2]

        new_r = p_r - g_r
        new_g = p_g - g_g
        new_b = p_b - g_b
        

        new = np.copy(p)
        new[:,:,0] = new_r
        new[:,:,1] = new_g
        new[:,:,2] = new_b
        retval.append(new)

    retval.append(gauss_pyramid_np[len(gauss_pyramid_np) - 1])
    
    return retval

# Part 2 Q3
def ShowLaplacianPyramid(pyramid):
    # offset by 128 for all but last
    for i in range(len(pyramid) - 1):
        pyramid[i] = pyramid[i] + 128
    
    # turn them into PIL image
    new_pyr = []
    for ia in pyramid:
        new_pyr.append(Image.fromarray(ia))
    
    # now the new_pyr is just like the output of Part 1
    # get the length
    length = len(new_pyr)
    # height is just the max height which is first image
    width = 0
    height = new_pyr[0].size[1] # this is the final height

    # add up the width
    for im in new_pyr:
        width = width + im.size[0]

    # init an image
    image = Image.new("RGB", (width, height))

    offset_x = 0
    offset_y = 0

    # paste the image
    for im in new_pyr:
        image.paste(im, (offset_x, offset_y))
        # update offset x, y need not change
        offset_x = offset_x + im.size[0]

    image.show()

# Part 2 Q4
def ReconstructGaussianFromLaplacianPyramid(lPyramid):
    # last one goes as is
    retval = [lPyramid[len(lPyramid) - 1]]
    current = retval[0]
    # retval.insert(0, )
    # from the last one
    for i in range(len(lPyramid)-1):
        index = len(lPyramid) - 1 - i
        next_index = index - 1
        #current = lPyramid[index]
        nex = np.copy(lPyramid[next_index])
        # upsample the image
        im_current = Image.fromarray(current)
        im_next = Image.fromarray(nex)

        current = np.array(im_current.resize(im_next.size, Image.BICUBIC))
        # current is now upsampled
        current_r = current[:,:,0]
        current_g = current[:,:,1]
        current_b = current[:,:,2]
        nex_r = nex[:,:,0]
        nex_g = nex[:,:,1]
        nex_b = nex[:,:,2]
        
        new_r = current_r + nex_r
        new_g = current_g + nex_g
        new_b = current_b + nex_b
        #new_r = np.clip(new_r, 0, 255)
        #new_g = np.clip(new_g, 0, 255)
        #new_b = np.clip(new_b, 0, 255)

        new = np.copy(nex)
        new[:,:,0] = new_r
        new[:,:,1] = new_g
        new[:,:,2] = new_b

        retval.insert(0, new)
        current = new
    
    # convert to PIL
    for i in range(len(retval)):
        retval[i] = Image.fromarray(retval[i])

    return retval


def TestPyr(image_path):
    image = Image.open(image_path)
    li = MakeGaussianPyramid(image, 0.75, 20)
    ShowGaussianPyramid(li)
