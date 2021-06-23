import torch
from torchvision.transforms import ToTensor
import cv2
import numpy as np
from nets.inpaint_networks import Generator
from helper import load_scan, return_scan_to_orig
import nibabel as nib
from scipy.ndimage.morphology import binary_opening, binary_closing
from matplotlib import pyplot as plt
from helper import get_vert_range
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder

class Inpaint(object):

    def __init__(self, img_scan_path, mask_path, vertebra_id, inpainted_img_path = 'inpainted_img.nii.gz', inpainted_mask_path = 'inpainted_mask.nii.gz', vertebra_range=None):
        # load the img and mask
        self.orig_img3d, _, _, _ = load_scan(img_scan_path, resample=True)
        self.orig_mask3d, self.mask_header, self.mask_affine, self.zooms = load_scan(mask_path, resample=True)
        self.orig_mask3d = self.orig_mask3d.astype(np.uint8)
        #self.zooms = self.mask_header.get_zooms() #RIP so lateral, axial, coronal
        
        # crop scan to include only five vertebrae
        self.vertebra_id = vertebra_id
        self.vertebra_range = vertebra_range
        self.crop_to_five()

        categories = np.arange(6)
        self.one_hot_encoder = OneHotEncoder(categories=[categories])

        self.orig_ax_length = self.img3d.shape[1] # axial
        self.orig_cor_length = self.img3d.shape[2] 
        self.orig_lat_length = self.img3d.shape[0] 

        # set saved model path, device, transforms
        '''
        if self.vertebra_range.index(self.vertebra_id) == 2:
            lat_model_path = 'models/inpaintLatMid.pt'
            cor_model_path = 'models/inpaintCorMid.pt'            
        else:
        '''
        lat_model_path = 'models/inpaint_modelLat.pt'
        cor_model_path = 'models/inpaint_modelCor.pt'

        if torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
        self.toTensor = ToTensor()
        self.resize_size = (128, 128)
        # network parameters
        netG_params = {'input_dim': 4, 'ngf': 16}
        self.netGlat = Generator(netG_params, self.use_cuda)
        self.netGcor = Generator(netG_params, self.use_cuda)
        if self.use_cuda:
            self.netGlat.cuda()
            self.netGcor.cuda()
        # load weights
        if not self.use_cuda:
            state_dict_lat = torch.load(lat_model_path, map_location='cpu')
            state_dict_cor = torch.load(cor_model_path, map_location='cpu')
        else:
            state_dict_lat = torch.load(lat_model_path)
            state_dict_cor = torch.load(cor_model_path)
        self.netGlat.load_state_dict(state_dict_lat['netG_state_dict'])
        self.netGcor.load_state_dict(state_dict_cor['netG_state_dict'])
        self.inpainted_img_path = inpainted_img_path
        self.inpainted_mask_path = inpainted_mask_path
        self.softmax = torch.nn.Softmax(dim=1)

    def crop_to_five(self):
        '''
        Crop the scan and mask to include five vertebrae and pad if necessary so along all axes we have same size
        '''
        if not self.vertebra_range:
            self.vertebra_range = get_vert_range(self.orig_mask3d, self.vertebra_id)
        #print("Vertebra range taken for inpainting: ", self.vertebra_range)
        topx, topy, topz = np.where(self.orig_mask3d==self.vertebra_range[0])
        x, y, z = np.where(self.orig_mask3d==self.vertebra_id)
        bottomx, bottomy, bottomz = np.where(self.orig_mask3d==self.vertebra_range[-1]) 
        xarray = np.concatenate([topx,bottomx])
        yarray = np.concatenate([topy,bottomy])
        zarray = np.concatenate([topz,bottomz])
        #RIP so lateral, axial, coronal
        self.ymax, self.ymin = np.max(yarray), np.min(yarray)
        zmax, zmin = np.max(zarray), np.min(zarray)
        yrange = self.ymax-self.ymin
        zrange = self.orig_img3d.shape[2] #max - zmin
        xrange = self.orig_img3d.shape[0]
        # crop or pad coronal dims to match axial
        if yrange < zrange:
            self.padz = False
            # crop directly since coronal always larger than yrange (=initial axial size) 
            zmid = np.min(z) + (np.max(z) - np.min(z))//2
            self.zcrop1 = zmid - yrange//2
            self.zcrop2 = zmid + yrange//2 + yrange%2
            if self.zcrop1  < 0 or self.zcrop2 > self.orig_img3d.shape[2]:
                    self.zcrop1, self.zcrop2 = 0, yrange
            self.img3d = np.copy(self.orig_img3d[:,self.ymin:self.ymax,self.zcrop1:self.zcrop2])
            self.mask3d = np.copy(self.orig_mask3d[:,self.ymin:self.ymax,self.zcrop1:self.zcrop2])
        else:
            self.padz = True
            pad_size = yrange-zrange
            pad_size_mod = pad_size%2
            img_crop = np.copy(self.orig_img3d[:,self.ymin:self.ymax,:])
            mask_crop = np.copy(self.orig_mask3d[:,self.ymin:self.ymax,:])
            self.zcrop1, self.zcrop2 = pad_size//2, pad_size//2+pad_size_mod
            self.img3d = np.pad(img_crop, ((0,0),(0,0), (self.zcrop1, self.zcrop2)), 'constant', constant_values=(-1024))
            self.mask3d = np.pad(mask_crop, ((0,0),(0,0), (self.zcrop1, self.zcrop2)), 'constant', constant_values=(0))
        # now same for lateral
        if yrange < xrange:
            self.padx = False
            # crop directly since coronal always larger than yrange (=initial axial size) 
            xmid = np.min(x) + (np.max(x) - np.min(x))//2
            self.xcrop1 = xmid - xrange//2
            self.xcrop2 = xmid + xrange//2 + xrange%2
            if self.xcrop1  < 0 or self.xcrop1 > self.orig_img3d.shape[0]:
                    self.xcrop1, self.xcrop2 = 0, xrange
            self.img3d = self.img3d[self.xcrop1:self.xcrop2,:,:]
            self.mask3d = self.img3d[self.xcrop1:self.xcrop2,:,:]
        else:
            self.padx = True
            pad_size = yrange-xrange
            pad_size_mod = pad_size%2
            self.xcrop1, self.xcrop2 = pad_size//2, pad_size//2+pad_size_mod
            self.img3d = np.pad(self.img3d, ((self.xcrop1, self.xcrop2),(0,0),(0,0)), 'constant', constant_values=(-1024))
            self.mask3d = np.pad(self.mask3d, ((self.xcrop1, self.xcrop2),(0,0),(0,0)), 'constant', constant_values=(0))

    def fill_vert(self, mask):
        """
        This function takes a segmentation mask as input and fills the inpainted 
        vertebra wherever not filled (i.e. not vertebra_id everywhere inside). 
        Parameters
        ----------
            mask: numpy array 
                The mask whose components we want to fill
        Returns
        -------
            mask_filled: numpy array
                The mask with the components filled
        """
        im_floodfill = np.copy(mask)
        im_floodfill[im_floodfill!=self.vertebra_id] = 0
        im_floodfill[im_floodfill==self.vertebra_id] = 255
        im_floodfill_copy = np.copy(im_floodfill)
        # The size needs to be 2 pixels larger than the image.
        h, w = im_floodfill.shape[:2]
        mask4mask = np.zeros((h+2, w+2), np.uint8)
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask4mask, (0,0), 255)
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # Combine the two images to get the foreground.
        im_floodfill_inv = im_floodfill_inv | im_floodfill_copy
        im_floodfill_inv[im_floodfill_inv==255] = self.vertebra_id
        mask_filled = mask | im_floodfill_inv
        return mask_filled
    
    def inpaint(self, img_slice, mask_slice, min_x, max_x, min_y, max_y, views='lateral'):
        """
        This function inpaints a 2D image and mask
        Parameters
        ----------
            img_slice: numpy array
                The 2D image we wish to inpaint
            mask: numpy array 
                The 2D mask we wish to inpaint
            min_x: int
                The minimum row index of vertbra to be inpainted 
            max_x: int
                The maximum row index of vertbra to be inpainted
            min_y: int
                The minimum column index of vertbra to be inpainted
            max_y: int
                The maximum column index of vertbra to be inpainted
            views: string
                If lateral then lateral model is used for inpainting, else the coronal
        Returns
        -------
            inpainted_img: numpy array
                The inpainted image
            inpainted_mask: numpy array
                The inpainted mask
            mask_binary: numpy array
                The binary mask used for inpainting
        """
        # create binary mask
        mask = np.zeros(img_slice.shape)
        mask[min_x:max_x, min_y:max_y] = 1
        # keep a copy of original to have background later 
        img_orig = np.copy(img_slice)
        mask_binary = np.copy(mask)

        # rotate image if coronal
        if views=='coronal':
            img_slice = np.rot90(img_slice, axes=(1, 0)) # image is from lat,ax -> ax,lat
            mask_slice = np.rot90(mask_slice, axes=(1, 0))
            mask = np.rot90(mask, axes=(1, 0))
       
        # prepare binary mask for net
        mask = cv2.resize(mask, self.resize_size, interpolation=cv2.INTER_NEAREST)
        mask = torch.Tensor(mask) # gives dtype float32
        mask = mask.unsqueeze(0)
        mask = mask.unsqueeze(0)

        # prepare seg mask for net
        mask_slice[mask_slice==self.vertebra_id] = 0
        # resize to network size
        mask_seg = cv2.resize(mask_slice, self.resize_size, interpolation=cv2.INTER_NEAREST)
        mask_seg = np.uint8(np.round(mask_seg)) # just to be sure

        mask_seg = self.map_vert_to_class(mask_seg)
        mask_seg = torch.Tensor(mask_seg) # gives dtype float32
        mask_seg_one_hot = torch.nn.functional.one_hot(mask_seg.long(), num_classes=6)
        mask_seg_one_hot = mask_seg_one_hot.permute(2,0,1)
        mask_seg_one_hot = mask_seg_one_hot.unsqueeze(0)
        mask_seg = mask_seg.unsqueeze(0)
        mask_seg = mask_seg.unsqueeze(0)

        # prepare img for net   
        img_slice = cv2.resize(img_slice, self.resize_size)
        img_slice = np.clip(img_slice, -1024, 3071) # clip to HU units
        img_slice = np.uint8(255*(img_slice+1024)/4095) # normalize to range 0-255 
        img_slice = img_slice[:,:, None]
        img_slice = self.toTensor(img_slice)
        img_slice = img_slice.unsqueeze(0)
        corrupt_img = (1-mask)*img_slice

        if self.use_cuda:
            mask = mask.cuda()
            mask_seg = mask_seg.cuda()
            corrupt_img = corrupt_img.cuda() 

        # inpaint
        if views=='lateral':
            netG = self.netGlat
        elif views=='coronal':
            netG = self.netGcor

        # get prediction
        with torch.no_grad():
            _, inpainted_mask, inpainted_img = netG(corrupt_img, mask_seg, mask)
        inpainted_mask = self.softmax(inpainted_mask)

        #inpainted_mask = torch.argmax(inpainted_mask, dim=1)
        inpainted_img = inpainted_img * mask + corrupt_img * (1. - mask)
        inpainted_mask = inpainted_mask * mask + mask_seg_one_hot * (1. - mask)
        #inpainted_mask = self.map_class_to_vert(inpainted_mask)

        # set img back to how it was
        inpainted_img = inpainted_img.squeeze().detach().cpu().numpy()
        inpainted_img = (inpainted_img)*4095 - 1024 # normalize back to HU units 
        inpainted_img = cv2.resize(inpainted_img, (self.orig_ax_length, self.orig_ax_length))
        # set mask back
        inpainted_mask = inpainted_mask.squeeze().detach().cpu().numpy()
        inpainted_mask_resized = np.zeros((6, self.orig_ax_length, self.orig_ax_length))
        for i in range(6):
            if views=='coronal':
                inpainted_mask_resized[i,:,:] = np.rot90(cv2.resize(inpainted_mask[i,:,:], (self.orig_ax_length, self.orig_ax_length))) #, interpolation=cv2.INTER_NEAREST)
            else:
                inpainted_mask_resized[i,:,:] = cv2.resize(inpainted_mask[i,:,:], (self.orig_ax_length, self.orig_ax_length)) #, interpolation=cv2.INTER_NEAREST)
        inpainted_mask = inpainted_mask_resized
        
        if views=='coronal':
            inpainted_img = np.rot90(inpainted_img) #, axes=(1, 0))

        return inpainted_img, inpainted_mask, mask_binary 

    def map_vert_to_class(self, mask_seg):
        """
        Since the inpainting models have been trained with vertebrae labels 1-5 this functions maps the 
        five vertebrae to labels 1-5
        in the image
        Parameters
        ----------
            mask_seg: numpy array 
                The mask we want to inpaint with unknown vertebrae labels
        Returns
        -------
            mask_seg: numpy array
                The mask with the vertebrae labels mapped to 1-5
        """
        mask_seg[mask_seg==self.vertebra_range[0]] = 1
        mask_seg[mask_seg==self.vertebra_range[1]] = 2
        mask_seg[mask_seg==self.vertebra_range[2]] = 3
        mask_seg[mask_seg==self.vertebra_range[3]] = 4
        mask_seg[mask_seg==self.vertebra_range[4]] = 5
        vert_values=np.arange(1,6)
        found_vert = np.in1d(mask_seg, vert_values)
        found_vert = np.reshape(found_vert, mask_seg.shape)
        mask_seg[found_vert==False] = 0
        return mask_seg
    
    def map_class_to_vert(self, inpainted_mask):
        """
        The reverse operation of map_vert_to_class. Here the mask includes labels 1-5 which are mapped back
        to the original labels in the scan
        Parameters
        ----------
            inpainted_mask: numpy array 
                The inpainted mask with the vertebrae labels 1-5
        Returns
        -------
            inpainted_mask: numpy array
                The inpainted mask with original vertebrae labels
        """
        inpainted_mask[inpainted_mask==1] = self.vertebra_range[0]
        inpainted_mask[inpainted_mask==2] = self.vertebra_range[1]
        inpainted_mask[inpainted_mask==3] = self.vertebra_range[2]
        inpainted_mask[inpainted_mask==4] = self.vertebra_range[3]
        inpainted_mask[inpainted_mask==5] = self.vertebra_range[4]
        return inpainted_mask

    def get_one_hot(self, mask_slice):
        """
        This function converts a segmentation mask to a one-hot encoded version
        Parameters
        ----------
            mask_slice: numpy array 
                The 2D mask
        Returns
        -------
            mask_one_hot: numpy array
                The one hot encoding of the mask with 6 labels (including background)
        """
        mask_one_hot = self.map_vert_to_class(mask_slice).reshape(-1, 1)
        self.one_hot_encoder.fit(mask_one_hot)
        mask_one_hot = self.one_hot_encoder.transform(mask_one_hot).toarray()
        # set 1st dim to num classes
        mask_one_hot = np.transpose(mask_one_hot, (1, 0))
        mask_one_hot = mask_one_hot[:,:, None]
        mask_one_hot = mask_one_hot.reshape((6, mask_slice.shape[0], mask_slice.shape[1]))
        return mask_one_hot
        
    def apply(self, mode='lateral'):
        """
        This function applies inpainting on a 3D scan and segmentation mask
        Parameters
        ----------
            mode: string 
                If set to 'lateral' then only the lateral model is used for the inpainting
                If set to 'coronal' then only the coronal model is used for the inpainting
                If set to 'fuse' then both the lateral and coronal models are used for the inpainting
        """
        num_lat_slices = self.img3d.shape[0]
        num_cor_slices = self.img3d.shape[2]
        bin_mask = np.zeros(self.mask3d.shape)
        x,y,z = np.where(self.mask3d==self.vertebra_id)
        bin_mask[np.min(x):np.max(x), np.min(y):np.max(y), np.min(z):np.max(z)] = 1
        if mode=='lateral' or mode=='fuse':
            mask_lat = np.zeros((6, self.mask3d.shape[0], self.mask3d.shape[1], self.mask3d.shape[2]))
            img_lat = np.zeros(self.img3d.shape)
            binary_lat = np.zeros(self.mask3d.shape)
            # for each lateral slice
            for idx in range(num_lat_slices):
                img_slice, mask_slice = np.copy(self.img3d[idx, :, :]), np.copy(self.mask3d[idx, :, :])
                xloc, yloc = np.where(mask_slice==self.vertebra_id)
                # check if vertebra is present in image
                if xloc.shape[0]==0:
                    # if not keep mask as it is
                    mask_lat[:,idx, :, :] = self.get_one_hot(mask_slice)
                    img_lat[idx, :, :] = img_slice
                else:
                    min_x, max_x = np.min(xloc), np.max(xloc)
                    min_y, max_y = np.min(yloc), np.max(yloc)
                    inpainted_img, inpainted_mask, binary_mask = self.inpaint(img_slice, mask_slice, min_x, max_x, min_y, max_y)
                    mask_lat[:,idx, :, :] = inpainted_mask
                    img_lat[idx,:, :] = inpainted_img
                    binary_lat[idx,:,:] = binary_mask


        if mode=='coronal' or mode=='fuse':
            mask_cor = np.zeros((6, self.mask3d.shape[0], self.mask3d.shape[1], self.mask3d.shape[2]))
            img_cor = np.zeros(self.img3d.shape)
            binary_cor = np.zeros(self.mask3d.shape)
            # for each coronal slice
            for idx in range(num_cor_slices):
                img_slice, mask_slice = np.copy(self.img3d[:, :, idx]), np.copy(self.mask3d[:, :, idx])
                xloc, yloc = np.where(mask_slice==self.vertebra_id)
                # check if vertebra is present in image
                if xloc.shape[0]==0:
                    # if not keep mask as it is
                    mask_cor[:, :, :, idx] = self.get_one_hot(mask_slice)
                    img_cor[:, :, idx] = img_slice
                else:
                    min_x, max_x = np.min(xloc), np.max(xloc)
                    min_y, max_y = np.min(yloc), np.max(yloc)
                    # else remove fractured vertebra and inpaint
                    inpainted_img, inpainted_mask, binary_mask = self.inpaint(img_slice, mask_slice, min_x, max_x, min_y, max_y, 'coronal')
                    mask_cor[:, :, :, idx] = inpainted_mask
                    img_cor[:, :, idx] = inpainted_img
                    binary_cor[:,:,idx] = binary_mask
        
        # return to a one channel mask and convert labels back
        if mode=='lateral':
            mask_lat = np.argmax(mask_lat, axis=0)
            mask_lat = self.map_class_to_vert(mask_lat)
            self.mask3d = mask_lat
            self.img3d = img_lat
        elif mode=='coronal':
            mask_cor = np.argmax(mask_cor, axis=0)
            mask_cor = self.map_class_to_vert(mask_cor)
            self.mask3d = mask_cor
            self.img3d = img_cor
        elif mode=='fuse':
            mask_fuse = mask_cor*0.5+mask_lat*0.5
            mask_fuse = np.argmax(mask_fuse, axis=0)
            mask_fuse = self.map_class_to_vert(mask_fuse)
            self.mask3d = mask_fuse
            self.img3d = (img_lat+img_cor)/2
    
        # save result
        self.mask3d = self.mask3d.astype(np.uint8)
        self.img3d = self.img3d.astype(np.float32)
        
        # put back if we padded and cropped
        if self.padz and self.padx:
            self.orig_img3d[:,self.ymin:self.ymax, :] = self.img3d[self.xcrop1:-self.xcrop2,:,self.zcrop1:-self.zcrop2]
            self.orig_mask3d[:,self.ymin:self.ymax, :] = self.mask3d[self.xcrop1:-self.xcrop2,:,self.zcrop1:-self.zcrop2]
        elif self.padz and not self.padx:
            self.orig_img3d[self.xcrop1:self.xcrop2,self.ymin:self.ymax, :] = self.img3d[:,:,self.zcrop1:-self.zcrop2]
            self.orig_mask3d[self.xcrop1:self.xcrop2,self.ymin:self.ymax, :] = self.mask3d[:,:,self.zcrop1:-self.zcrop2]
        elif not self.padz and self.padx:
            self.orig_img3d[:,self.ymin:self.ymax, self.zcrop1:self.zcrop2] = self.img3d[self.xcrop1:-self.xcrop2,:,:]
            self.orig_mask3d[:,self.ymin:self.ymax, self.zcrop1:self.zcrop2] = self.mask3d[self.xcrop1:-self.xcrop2,:,:]
        else:
            self.orig_img3d[self.xcrop1:self.xcrop2,self.ymin:self.ymax, self.zcrop1:self.zcrop2] = self.img3d
            self.orig_mask3d[self.xcrop1:self.xcrop2,self.ymin:self.ymax, self.zcrop1:self.zcrop2] = self.mask3d
        
        img = return_scan_to_orig(self.orig_img3d, self.mask_affine, self.mask_header, self.zooms)
        nib.save(img, self.inpainted_img_path)

        mask_fuse = return_scan_to_orig(self.orig_mask3d, self.mask_affine, self.mask_header, self.zooms, np.uint8)
        nib.save(mask_fuse, self.inpainted_mask_path)
        print('Inpaint mask and image saved at: ', self.inpainted_mask_path, self.inpainted_img_path)
