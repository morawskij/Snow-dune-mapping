import numpy as np
import matplotlib.pyplot as plt
import rasterio
import matplotlib.colors as clr
import matplotlib.cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from astropy.convolution import convolve
import scipy.ndimage as nd
from skimage import morphology
from apply_unet import predict_image

def treshold_data_criterion(treshold):
    return lambda x: x>=treshold

def change_cmap(cmap,a,b,n=8,boolean=False,treshold=0.5,interv=None):
    new_colors=[]
    if boolean:
        for x in np.linspace(a,b,2**n):
            if boolean:
                if x<treshold:
                    new_colors.append(cmap(0))
                else:
                    new_colors.append(cmap(1.0))
    elif interv is not None:
        if b>a:
            for x in np.linspace(a,b,2**n):
                new_colors.append(cmap(np.floor((x-a)/interv)*interv/(b-a)))
        else:
            for x in np.linspace(a,b,2**n):
                new_colors.append(cmap(1.0))
    else:
        for x in np.linspace(a,b,2**n):
            if x<0:
                new_colors.append(cmap(0.5*(a-x)/a))
            else:
                new_colors.append(cmap(0.5*(1+x/b)))
    return clr.ListedColormap(new_colors)

def model_for_trough_skeletonization(VO_path=None,VOP_path=None,LoG_path=None,dune_mask_path=None,VOVOP_treshold=0.05,base_path=None,idnum=None):
    if base_path is not None:
        VOP_path=base_path+r'\VOP_'+str(idnum)+'.tif'
        VO_path=base_path+r'\VO_'+str(idnum)+'.tif'
        LoG_path=base_path+r'\LoG_'+str(idnum)+'.tif'
        dune_mask_path=base_path+r'\dune_buffer_mask_'+str(idnum)+'.tif'
    VO = raster_data(from_path=True,input_path=VO_path,nodata_value=-10,name="VO")
    VOP = raster_data(from_path=True,input_path=VOP_path,nodata_value=-10,name="VOP")
    VOVOP=VO.multiply_by(-1).add(1).multiply_by(VOP.multiply_by(-1).add(1),name=r'$(1-$'"VO"r'$)\bullet(1-$'"VOP"r'$)$'" (dunes)").normalize_positive().add(-VOVOP_treshold).multiply_by(1/(1-VOVOP_treshold))
    LoG_full = raster_data(from_path=True,input_path=LoG_path,nodata_value=-1,name="LoG",data_criterion=treshold_data_criterion(-1))
    LoG_trough_detector=LoG_full.multiply_by(-1,name="-LoG").normalize_positive()
    mask = raster_data(from_path=True,input_path=dune_mask_path,nodata_value=-10,name="Buffered dune_mask",data_criterion=treshold_data_criterion(0.5))
    return mask.multiply_by(VOVOP.add(LoG_trough_detector).multiply_by(0.5),name="Trough detector").setup_for_skeletonization()

def model_for_crest_skeletonization(VO_path=None,VON_path=None,LoG_path=None,dune_mask_path=None,VOVON_treshold=0.05,base_path=None,idnum=None):
    if base_path is not None:
        VON_path=base_path+r'\VON_'+str(idnum)+'.tif'
        VO_path=base_path+r'\VO_'+str(idnum)+'.tif'
        LoG_path=base_path+r'\LoG_'+str(idnum)+'.tif'
        dune_mask_path=base_path+r'\dune_buffer_mask_'+str(idnum)+'.tif'
    VO = raster_data(from_path=True,input_path=VO_path,nodata_value=-10,name="VO")
    VON = raster_data(from_path=True,input_path=VON_path,nodata_value=-10,name="VON")
    VOVON=VO.multiply_by(VON.multiply_by(-1).add(1),name="VO"r'$\bullet(1-$'"VON"r'$)$'" (dunes)").normalize_positive().add(-VOVON_treshold).multiply_by(1/(1-VOVON_treshold))
    LoG_full = raster_data(from_path=True,input_path=LoG_path,nodata_value=-1,name="LoG",data_criterion=treshold_data_criterion(-1))
    LoG_crest_detector=LoG_full.normalize_positive()
    mask = raster_data(from_path=True,input_path=dune_mask_path,nodata_value=-10,name="Buffered dune_mask",data_criterion=treshold_data_criterion(0.5))
    return mask.multiply_by(VOVON.add(LoG_crest_detector).multiply_by(0.5),name="Crest detector").setup_for_skeletonization()


class padded_array:

    def __init__(self,arr,x,y,pad_mode='edge'):
        self.data=np.pad(arr,1,pad_mode)
        self.x=x
        self.y=y
    
    def shift(self,i,j):
        return self.data[1+i:1+self.x+i,1+j:1+self.y+j] 

class raster_data:

    def __init__(self,from_path=False,input_path="",raster=None,profile=None,nodata_value=-1000,data_criterion=None,name="Raster data",treshold_val=None):

        self.nodata=nodata_value
        if data_criterion is None:
            data_criterion=treshold_data_criterion(self.nodata)
        self.name=name
        self.treshold_val=treshold_val
        
        if from_path:
            with rasterio.open(input_path,mode='r+',driver='GTiff') as src:
                self.data = src.read()[0,:,:]
                self.profile = src.profile

            if self.nodata is not None:
                self.data=np.where(data_criterion(self.data),self.data,self.nodata)

        else:
            self.data=raster
            self.profile=profile

        self.x,self.y = self.data.shape
        self.max=np.nanmax(self.data)
        self.min=np.nanmin(self.data)
        self.range=self.max-self.min

        self.transform=self.profile['transform']
        self.res=self.transform[0]

    def convolve(self,conv_filter):
        return self.__class__(raster=convolve(self.data,conv_filter,boundary='extend',normalize_kernel=False,nan_treatment='fill'),profile=self.profile,name=self.name,treshold_val=self.treshold_val,nodata_value=self.nodata)

    def smoothen(self,filter_size=3,object_output=True):
        filtered=nd.gaussian_filter(self.data, (filter_size-1)/2) 
        if object_output:
            return self.__class__(raster=filtered,profile=self.profile,name=self.name,treshold_val=self.treshold_val,nodata_value=self.nodata) 
        else:
            return filtered

    def smooth_median(self,filter_size=3,object_output=True):
        filtered=nd.median_filter(self.data, size=filter_size) 
        if object_output:
            return self.__class__(raster=filtered,profile=self.profile,name=self.name,treshold_val=self.treshold_val,nodata_value=self.nodata) 
        else:
            return filtered
    
    def set_name(self,name):
        self.name=name

    def crop(self,crop_x,crop_y):

        x0=int(crop_x[0]*self.x)
        x1=int(crop_x[1]*self.x)
        y0=int(crop_y[0]*self.y)
        y1=int(crop_y[1]*self.y)

        return self.__class__(raster=self.data[x0:x1,y0:y1],profile=self.profile,name=self.name,treshold_val=self.treshold_val,nodata_value=self.nodata)
    
    def tresholded(self,treshold,name=None,boolean=True,zeros_as_nodata=False,as_predictions=False,extra_demand=0,extra_demand_buff=0):

        if name is None:
            name=self.name
        
        treshold_val=treshold
        if treshold>np.max(self.data):
            treshold_val=None

        if boolean:
            for_matching=1
        else:
            for_matching=self.data

        if zeros_as_nodata:
            non_matching=self.nodata
        else:
            non_matching=0

        if as_predictions:
            return_cls = predictions
        else:
            return_cls=self.__class__

        # To bypass anomalies of LoG at the edges due to padding
        if extra_demand>0.0:
            trsh=np.ones_like(self.data)*extra_demand
            trsh[extra_demand_buff:-extra_demand_buff-1,extra_demand_buff:-extra_demand_buff-1]=0
            trsh=nd.gaussian_filter(trsh, extra_demand_buff/2) 
        else:
            trsh=np.zeros_like(self.data)
        return return_cls(raster=np.where(self.data>self.nodata,np.where(self.data>trsh+treshold,for_matching,non_matching),self.nodata),profile=self.profile,name=name,treshold_val=treshold_val,nodata_value=self.nodata)

    def padded(self,pad_width=1,pad_mode='constant',pad_vals=(0,0),name=None):

        if name is None:
            name=self.name

        return self.__class__(raster=np.pad(self.data,pad_width,pad_mode,constant_values=pad_vals),profile=self.profile,name=name,treshold_val=self.treshold_val,nodata_value=self.nodata)

    
    def visualize(self,cmap=matplotlib.cm.viridis,crop_x=[0,1],crop_y=[0,1],fig=None,ax=None,figsize=(10,10),title=None,boolean=False,discrete=None,alpha=1,cbar=True,draw_pixel_grid=False,grid_color="white",y=1.03):

        if fig is None or ax is None:
            fig,ax=plt.subplots(figsize=figsize)

        if title is None:
            title=self.name

        cropped=self.crop(crop_x,crop_y)

        cropped=self.__class__(raster=np.where(cropped.data>cropped.nodata,cropped.data,np.nan),profile=self.profile,name=self.name,treshold_val=self.treshold_val)

        change=boolean or (discrete is not None)
        
        if change:
            cmap=change_cmap(cmap,cropped.min,cropped.max,treshold=cropped.treshold_val,boolean=boolean,interv=discrete)

        sc=ax.imshow(cropped.data,cmap=cmap,alpha=alpha)
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0.05)
            if discrete is not None:
                pos=np.arange(int(cropped.range/discrete)+1)*discrete+cropped.min
                bar = fig.colorbar(sc,cax=cax,orientation='horizontal',spacing='proportional',ticks=discrete/2+pos,boundaries=np.append(pos,pos[-1]+discrete))     
                bar.ax.set_xticklabels((pos).astype(type(discrete)))
            else:
                bar = fig.colorbar(sc,cax=cax,orientation='horizontal')
        if draw_pixel_grid: 
            ax.set_yticks(np.arange(-0.5, x1-x0, 1), minor=True)
            ax.set_xticks(np.arange(-0.5, y1-y0, 1), minor=True)
            ax.grid(color=grid_color,lw=1,which='minor')
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(title,y=y)

    def multiply_by(self,fac,name=None):

        if name is None:
            name=self.name
        
        if isinstance(fac,raster_data):
            return self.__class__(raster=np.where(self.data>self.nodata,fac.data*self.data,self.nodata),profile=self.profile,name=name,treshold_val=self.treshold_val,nodata_value=self.nodata)
        else:
            return self.__class__(raster=np.where(self.data>self.nodata,fac*self.data,self.nodata),profile=self.profile,name=name,treshold_val=self.treshold_val,nodata_value=self.nodata)
                                      
    def add(self,cst,name=None):

        if name is None:
            name=self.name
        
        if isinstance(cst,raster_data):
            return self.__class__(raster=np.where(self.data>self.nodata,cst.data+self.data,self.nodata),profile=self.profile,name=name,treshold_val=self.treshold_val,nodata_value=self.nodata)  
        else:
            return self.__class__(raster=np.where(self.data>self.nodata,cst+self.data,self.nodata),profile=self.profile,name=name,treshold_val=self.treshold_val,nodata_value=self.nodata)

    def equal_to(self,val=1,name=None):

        if name is None:
            name=self.name

            return self.__class__(raster=np.where(self.data==val,1,0),profile=self.profile,name=name,treshold_val=self.treshold_val)

    def normalize_positive(self):
        
        dummy = self.tresholded(0,boolean=False)
        return dummy.multiply_by(1.0/dummy.max)

    def write_single_band_tif(self, outfile, dtype=None):

        if dtype is None:
            if np.isclose(self.max,np.floor(self.max)):
                dtype=rasterio.int16
            else:
                dtype=rasterio.float32
        self.profile.update(dtype=dtype,count=1,nodata=self.nodata)
        with rasterio.open(outfile, 'w', **self.profile) as dst:
            dst.write(self.data.astype(dtype),1)

    def setup_for_skeletonization(self,name=None):
        if name is None:
            name=self.name
        return skeletonizable_raster(from_path=False,input_path="",raster=self.data,profile=self.profile,nodata_value=self.nodata,data_criterion=treshold_data_criterion(self.nodata),name=name,treshold_val=self.treshold_val)
        
class skeletonizable_raster(raster_data):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.prepare_neighbors()
    
    def prepare_neighbors(self):
        indxs=np.arange(-1,2)
        indxs=list(map(lambda x: x.flatten(),np.meshgrid(indxs,indxs)))
        neighbors=np.array(list(zip(indxs[0],indxs[1])))
        neighbors=np.delete(neighbors,4,axis=0)
        neighbors_order=np.array([0,1,2,4,7,6,5,3])
        self.neighbors=neighbors[neighbors_order]
    
    def skeletonize_B(self,boolean=True,last_check=False):
        arr=padded_array(self.data,self.x,self.y)
        B=sum(map(lambda n: arr.shift(n[0],n[1]),self.neighbors))
        if last_check:
            B=np.where(B>=7,0,1)
        elif boolean:
            B=np.where((B-2)*(B-6)<=0,1,0)
        return skeletonizable_raster(raster=B,profile=self.profile,name="B",treshold_val=0.5)

    def skeletonize_A(self,boolean=True):
        arr=padded_array(self.data,self.x,self.y)
        A_prep=np.zeros_like(self.data)
        A=sum(map(lambda i: np.where(arr.shift(self.neighbors[i][0],self.neighbors[i][1])==0,1,0)*np.where(arr.shift(self.neighbors[(i+1)%8][0],self.neighbors[(i+1)%8][1])>0,1,0),np.arange(8)))        
        if boolean:
            A=np.where(A==1,1,0)
        return skeletonizable_raster(raster=A,profile=self.profile,name="A",treshold_val=0.5)
        
    def skeletonize_D(self,boolean=True,iter2=False):
        arr=padded_array(self.data,self.x,self.y)
        if iter2:
            D=arr.shift(-1,0)*arr.shift(0,1)*arr.shift(0,-1)
        else:
            D=arr.shift(-1,0)*arr.shift(0,1)*arr.shift(1,0)
        if boolean:
            D=np.where(D!=1,1,0)
        return skeletonizable_raster(raster=D,profile=self.profile,name="D",treshold_val=0.5)

    def skeletonize_E(self,boolean=True,iter2=False):
        arr=padded_array(self.data,self.x,self.y)
        if iter2:
            E=arr.shift(0,-1)*arr.shift(-1,0)*arr.shift(1,0)
        else:
            E=arr.shift(0,-1)*arr.shift(0,1)*arr.shift(1,0)
        if boolean:
            E=np.where(E!=1,1,0)
        return skeletonizable_raster(raster=E,profile=self.profile,name="E",treshold_val=0.5)

    def skeletonize_step(self,iter2=False,change=0):

        to_be_erased = self.equal_to().multiply_by(self.skeletonize_A()).multiply_by(self.skeletonize_B()).multiply_by(self.skeletonize_D(iter2=iter2)).multiply_by(self.skeletonize_E(iter2=iter2))
        change+=np.sum(to_be_erased.data)
        return self.multiply_by(to_be_erased.multiply_by(-1).add(1)), change
    
    def skeletonization(self,levels=None,skeleton=None,verbose=True,verbose_concerned=False,tck=None,stage=1,visual_control=True,crop_x=[0,0.1],crop_y=[0,0.1],boolean=False,cmap1=plt.cm.viridis,cmap2=plt.cm.plasma,cmap3=plt.cm.gray):

        if levels is None:
            levels=[(1-i/4.0)*self.max+self.min*i/4.0 for i in range(1,5)]
        nlevels=len(levels)
        
        if skeleton is None:
            skeleton=self.tresholded(levels[0],name="Skeleton")
            
        else: 
            skeleton=skeleton.add(self.tresholded(levels[0]))
        
        if visual_control:
            fig,ax = plt.subplots(ncols=2,figsize=(15,7))
            skeleton.visualize(fig=fig,ax=ax[0],crop_x=crop_x,crop_y=crop_y,cmap=cmap2,discrete=1,title="Entering skeleton work {}/{}".format(stage,nlevels+stage-1))
        
        if verbose:
            if tck is None:
                print("Starting skeletonization")
                tck=time.time()
            print("Entering stage {}/{}".format(stage,nlevels+stage-1))


        
        while True:
            change=0
            if verbose_concerned:
                print("Iteration 1")
            skeleton, change = skeleton.skeletonize_step(change=change)
            if verbose_concerned:
                print("Iteration 2")
            skeleton, change = skeleton.skeletonize_step(iter2=True,change=change)
            if change==0:
                break

        #skeleton=skeleton.multiply_by(skeleton.skeletonize_B(last_check=True))
        
        if visual_control:
            self.visualize(fig=fig,ax=ax[1],crop_x=crop_x,crop_y=crop_y,cmap=cmap1)
            if nlevels==1:
                skeleton_progress=skeletonizable_raster(raster=np.where(skeleton.data>0,skeleton.data,-10),profile=self.profile,nodata_value=-10,name=self.name+" final skeleton")
            else:
                skeleton_progress=skeletonizable_raster(raster=np.where(skeleton.data>0,skeleton.data,-10),profile=self.profile,nodata_value=-10,name=self.name+" partial skeleton")
            skeleton_progress.visualize(fig=fig,ax=ax[1],crop_x=crop_x,crop_y=crop_y,cmap=cmap3,discrete=1,cbar=False)
        
        if nlevels==1:
            if verbose:
                print("Skeletonization completed in {:.2f} s".format(time.time()-tck))
            if boolean:
                skeleton.data=np.where(skeleton.data>0,1,-10)
            else:
                skeleton.data=np.where(skeleton.data>0,skeleton.data,-10)
            skeleton.nodata=-10
            return skeleton
        else:
            return self.skeletonization(levels=levels[1:],skeleton=skeleton,verbose=verbose,verbose_concerned=verbose_concerned,tck=tck,stage=stage+1,visual_control=visual_control,boolean=boolean,crop_x=crop_x,crop_y=crop_y,cmap1=cmap1,cmap2=cmap2)

    def thin_skeleton(self,verbose=True):
        if verbose:
            tck2=time.time()
            print("Starting thinning of the skeleton")
            skeleton_thin=skeletonizable_raster(raster=self.data,nodata_value=-20,profile=self.profile,name=self.name+" thinned",data_criterion=treshold_data_criterion(-20))
            skeleton_thin.data=morphology.thin(skeleton_thin.tresholded(0.5).data).astype(int)
        if verbose:
            print("Thinning the skeleton performed in {:.2f} s".format(time.time()-tck2))
        return skeleton_thin


class DEM(raster_data):

    def __init__(self,wavelength_of_interest=15,RR_filter=None,VO_radius=None,VO_VE=1000,starting_visual=False,starting_visual_figsize=(15,6),precompute=False,precompute_VO=True,verbose=True,presmoothing=True,presmoothing_filter=3,paths_to_precomputed={},**kwargs):
        super().__init__(**kwargs)
        self.WOI=wavelength_of_interest
        if RR_filter is None:
            RR_filter=self.WOI

        if presmoothing:
            self.data=self.smooth_median(filter_size=presmoothing_filter,object_output=False)
        if precompute:
            if 'RR' in paths_to_precomputed.keys():
                self.RR=DEM(from_path=True,input_path=paths_to_precomputed['RR'],name=self.name+" RR",treshold_val=self.treshold_val,nodata_value=self.nodata)
            else:
                self.RR=self.ResidualRelief(RR_filter,verbose=verbose)
            if 'slopes' in paths_to_precomputed.keys():
                self.slopes=raster_data(from_path=True,input_path=paths_to_precomputed['slopes'],name=self.name+" RR slopes",treshold_val=self.treshold_val,nodata_value=self.nodata)
            else:
                self.slopes=self.RR.slope_map(verbose=verbose)
            if 'LoG' in paths_to_precomputed.keys():
                self.LoG=raster_data(from_path=True,input_path=paths_to_precomputed['LoG'],name=self.name+" RR LoG",treshold_val=self.treshold_val,nodata_value=self.nodata)
            else:
                self.LoG=self.RR.compute_LoG(verbose=verbose)
            if precompute_VO:
                if 'VO' in paths_to_precomputed.keys():
                    self.VO=raster_data(from_path=True,input_path=paths_to_precomputed['VO'],name=self.name+" RR VO",treshold_val=self.treshold_val,nodata_value=self.nodata)
                else:
                    self.VO=self.RR.compute_VO(radius=VO_radius,VE=VO_VE,verbose=verbose)
                if 'VOP' in paths_to_precomputed.keys():
                    self.VOP=raster_data(from_path=True,input_path=paths_to_precomputed['VOP'],name=self.name+" RR VOP",treshold_val=self.treshold_val,nodata_value=self.nodata)
                else:
                    self.VOP=self.RR.compute_VOP(radius=VO_radius,VE=VO_VE,verbose=verbose)
                if 'VON' in paths_to_precomputed.keys():
                    self.VON=raster_data(from_path=True,input_path=paths_to_precomputed['VON'],name=self.name+" RR VON",treshold_val=self.treshold_val,nodata_value=self.nodata)
                else:
                    self.VON=self.RR.compute_VON(radius=VO_radius,VE=VO_VE,verbose=verbose)
        if starting_visual:
            fig, ax = plt.subplots(nrows=2,ncols=3,figsize=starting_visual_figsize)
            self.visualize(fig=fig,ax=ax[0,0],y=1)
            self.RR.visualize(fig=fig,ax=ax[0,1],cmap=change_cmap(plt.cm.coolwarm,self.RR.min,self.RR.max),y=1)
            self.LoG.visualize(fig=fig,ax=ax[0,2],cmap=change_cmap(plt.cm.Spectral,self.LoG.min,self.LoG.max),y=1)
            self.VO.visualize(fig=fig,ax=ax[1,0],y=1)
            self.VOP.visualize(fig=fig,ax=ax[1,1],y=1)
            self.VON.visualize(fig=fig,ax=ax[1,2],y=1)

    def ResidualRelief(self,filter_size,name=None,smoothen=False,verbose=True,additional_smoothing=False):

        if name is None:
            name=self.name+" RR"
        tck=time.time()
        if verbose:
            print("Starting RR calculation")
        filtered = self.data - nd.uniform_filter(self.data, size=filter_size) 
        if verbose:
            print("RR calculation completed in {:.2f} s".format(time.time()-tck))
        RR_map=DEM(raster=filtered,profile=self.profile,name=name,treshold_val=self.treshold_val,nodata_value=self.nodata,presmoothing=additional_smoothing)
        if smoothen:
            RR_map=RR_map.smoothen()
        return RR_map
    
    def slope_map(self,name=None,smoothen=True,verbose=True):

        if name is None:
            name=self.name+" slopes"

        tck=time.time()
        if verbose:
            print("Starting slope calculation")
        
        x, y = np.gradient(self.data, self.res, self.res)
        
        slope = np.sqrt(x**2 + y**2)
        slope = np.degrees(np.arctan(slope))

        slope_map=raster_data(raster=slope,profile=self.profile,name=name,treshold_val=self.treshold_val,nodata_value=self.nodata)
        if smoothen:
            slope_map=slope_map.smoothen()
        if verbose:
            print("Slope calculation completed in {:.2f} s".format(time.time()-tck))
        return slope_map

    def compute_LoG(self,name=None,filter_size=5,verbose=True):

        if name is None:
            name=self.name+" LoG"
        tck=time.time()
        if verbose:
            print("Starting LoG calculation")
        
        LoG = self.smoothen(filter_size=filter_size).convolve(np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]))
        LoG.set_name(name)
        if verbose:
            print("LoG calculation completed in {:.2f} s".format(time.time()-tck))
        
        return LoG

    def compute_VO (self,VE=1000,name=None,radius=None,P=False,N=False,verbose=True):


        #Based on the source code from Rolland et al. 2022 paper "Volumetric Obscurance as a New Tool to Better Visualize Relief from Digital Elevation Models"


        expl_str=" VO{}".format(int(P)*"P"+int(N)*"N")
        if name is None:
            name=self.name+expl_str
    
        if radius is None:
            radius=self.WOI

        tck=time.time()
        if verbose:
            print("Starting"+expl_str+" calculation")

        x = np.arange(-radius, radius + 1, dtype = int)  
        y = np.arange(-radius, radius + 1, dtype = int)  
        grid = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2) 
        distance = np.sqrt(grid[:,0]**2 + grid[:,1]**2) 
        spherical_mask = distance < radius
        distance = distance[spherical_mask] 
        grid = grid[spherical_mask]
        vert_cor=np.sqrt(radius**2 - distance**2) * self.res / VE
        vo = np.zeros((self.x, self.y))
        ds=distance.size
        sm = np.sum(vert_cor)*(2-int(P or N))
        for i in range(0,ds):
            M = (np.roll(self.data, [grid[i, 0], grid[i, 1]], axis = [0, 1]) - self.data)
            vo_upd = vert_cor[i] - M   
            upper_trsh = (2-int(P)) * vert_cor[i]
            vo_upd[vo_upd > upper_trsh] = upper_trsh
            lower_trsh = int(N) * vert_cor[i]
            vo_upd[vo_upd < lower_trsh] = lower_trsh
            vo+=vo_upd*(1-2*int(N))+2*int(N)*vert_cor[i]
        vo/=sm
        vo_obj=raster_data(raster=vo,profile=self.profile,name=name,treshold_val=self.treshold_val,nodata_value=self.nodata)
        if verbose:
            print(expl_str[1:]+" calculated in {:.2f} s".format(time.time()-tck))
        return vo_obj

    def compute_VOP (self,VE=1000,name=None,radius=None,verbose=True):
        return self.compute_VO (VE=VE,name=name,radius=radius,P=True,verbose=verbose)

    def compute_VON (self,VE=1000,name=None,radius=None,verbose=True):
        return self.compute_VO (VE=VE,name=name,radius=radius,N=True,verbose=verbose)
    
    def dune_predictions_raw(self,method='slope and LoG treshold',AI_treshold=0.75,AI_model=None,AI_raw_output=None,AI_buffer=5,AI_low_limit=-1.0411863,AI_high_limit=0.98570275,AI_treshold_edge_buff=10,slope_treshold=0.1,slope_treshold_edge_extra=0.1,slope_treshold_edge_buff=10,LoG_treshold_edge_extra=0.04,LoG_treshold_edge_buff=10,LoG_treshold=0.01,name=None,visual_control=False,visual_control_figsize=(15,10),verbose=True,output_to_file=None):

        if name is None:
            name=self.name+" dune predictions"

        tck=time.time()
        if verbose:
            print("Starting {} based prediction".format(method))
        if(method=='AI'):
            if verbose:
                AI_verb=0
            else:
                AI_verb=None
            prediction_raster=predictions(profile=self.profile,raster=predict_image(self.LoG.data,AI_model,buffer=AI_buffer,verbose=AI_verb,low_limit=AI_low_limit,high_limit=AI_high_limit,report_frequency=10**(1+2*int(not verbose))))
            if AI_raw_output is not None:
                prediction_raster.write_single_band_tif(AI_raw_output)
            prediction_raster=prediction_raster.tresholded(AI_treshold,extra_demand=1-AI_treshold,extra_demand_buff=AI_treshold_edge_buff)
        else:
            prediction_raster=self.slopes.tresholded(slope_treshold,extra_demand=slope_treshold_edge_extra,extra_demand_buff=slope_treshold_edge_buff).add(self.LoG.tresholded(LoG_treshold,extra_demand=LoG_treshold_edge_extra,extra_demand_buff=LoG_treshold_edge_buff)).tresholded(0.5,name="Raw treshold based predictions",as_predictions=True)
        if visual_control:
            prediction_raster.visualize(figsize=visual_control_figsize)
        if verbose:
            print("{} based prediction calculated in {:.2f} s".format(method, time.time()-tck))
        return prediction_raster

    def dune_predictions_raw_cleaned(self,method='slopes',AI_treshold=0.75,AI_model=None,AI_raw_output=None,AI_buffer=5,AI_low_limit=-1.0411863,AI_high_limit=0.98570275,slope_treshold=0.1,LoG_treshold=0.01,size_treshold=50,hole_treshold=None,name=None,visual_control=False,visual_control_figsize=(15,10),verbose=True,output_to_file=None,AI_raw_precomputed=None):
      
        if name is None:
            name=self.name
            
        if AI_raw_precomputed is None:
            prediction_raster = self.dune_predictions_raw(method=method,AI_treshold=AI_treshold,AI_model=AI_model,AI_raw_output=AI_raw_output,AI_buffer=AI_buffer,AI_low_limit=AI_low_limit,AI_high_limit=AI_high_limit,slope_treshold=slope_treshold,LoG_treshold=LoG_treshold,name=name,visual_control=visual_control,visual_control_figsize=visual_control_figsize,verbose=verbose,output_to_file=output_to_file)
        else:
            if verbose:
                print("Loading pre-computed raw AI output")
            prediction_raster=predictions(input_path=AI_raw_precomputed,from_path=True).tresholded(AI_treshold)
        prediction_raster.cleaning_up_small_mess(size_treshold=size_treshold,hole_treshold=hole_treshold)
        if output_to_file is not None:
            prediction_raster.write_single_band_tif(output_to_file)
        return prediction_raster
            
    def dune_predictions(self,method='slopes',AI_treshold=0.9,AI_model=None,AI_raw_output=None,AI_buffer=24,AI_low_limit=-1.0411863,AI_high_limit=0.98570275,slope_treshold=0.1,LoG_treshold=0.01,size_treshold=None,hole_treshold=None,name=None,visual_control=False,visual_control_figsize=(15,10),verbose=True,skeleton_visual_control=False,crop_x=[0.1,0.2],crop_y=[0.1,0.2],path_to_skeleton=None,save_crests=None,save_skeleton=None,output_to_file=None,output_precutting=None,AI_raw_precomputed=None):

        if name is None:
            name=self.name

        if size_treshold is None:
            size_treshold=self.WOI**2
        
        tck=time.time()
        if verbose:
            print("Starting {} based model".format(method))
        prediction_raster=self.dune_predictions_raw_cleaned(method=method,AI_treshold=AI_treshold,AI_model=AI_model,AI_raw_output=AI_raw_output,AI_buffer=AI_buffer,AI_low_limit=AI_low_limit,AI_high_limit=AI_high_limit,slope_treshold=slope_treshold,LoG_treshold=LoG_treshold,name=name,visual_control=visual_control,visual_control_figsize=visual_control_figsize,verbose=verbose,output_to_file=output_precutting,AI_raw_precomputed=AI_raw_precomputed)
        if path_to_skeleton is None:
            buffered=prediction_raster.buffered(self.WOI)
            if verbose:
                print("Trough skeletonization")
            for_skeletonization=self.trough_detector(buffered,low_LoG_limit=(-1)*AI_low_limit)
            skeleton=for_skeletonization.skeletonization(levels=[0.5,0.2,0.1,0],boolean=True,visual_control=skeleton_visual_control,crop_x=crop_x,crop_y=crop_y)
            if save_skeleton is not None:
                skeleton.write_single_band_tif(save_skeleton)
        else:
            skeleton=skeletonizable_raster(from_path=True,input_path=path_to_skeleton,nodata_value=-10)
        skeleton=skeleton.thin_skeleton()
        prediction_raster=prediction_raster.add(skeleton.multiply_by(-1)).tresholded(0.5)
        prediction_raster.cleaning_up_small_mess(size_treshold=size_treshold,hole_treshold=hole_treshold,holes_too=False)
        if output_to_file is not None:
            prediction_raster.write_single_band_tif(output_to_file)
        if verbose:
            print("Treshold based model evaluated in {:.2f} s".format(time.time()-tck))
        if save_crests is not None:
            if verbose:
                print("Crest skeletonization")
            for_skeletonization=self.crest_detector(prediction_raster)
            skeleton=for_skeletonization.skeletonization(levels=[0.5,0.2,0.1,0.05],boolean=True,visual_control=skeleton_visual_control,crop_x=crop_x,crop_y=crop_y)
            skeleton=skeleton.thin_skeleton()
            skeleton.nodata=0
            skeleton.write_single_band_tif(save_crests)
            return prediction_raster,skeleton
        else:
            return prediction_raster

    def trough_detector(self,mask,pad_mode='edge',VOVOP_treshold=0.03,low_LoG_limit=1.0411863):
        VOVOP=self.VO.multiply_by(-1).add(1).multiply_by(self.VOP.multiply_by(-1).add(1),name=r'$(1-$'"VO"r'$)\bullet(1-$'"VOP"r'$)$'" (dunes)").normalize_positive().add(-VOVOP_treshold).multiply_by(1/(1-VOVOP_treshold))
        LoG_trough_detector=self.LoG.multiply_by(-1,name="-LoG")
        LoG_trough_detector.data=np.where(LoG_trough_detector.data<low_LoG_limit,LoG_trough_detector.data,low_LoG_limit)
        LoG_trough_detector=LoG_trough_detector.normalize_positive()
        return mask.multiply_by(VOVOP.add(LoG_trough_detector).multiply_by(0.5),name="Trough detector").setup_for_skeletonization()

    def crest_detector(self,mask,pad_mode='edge',VOVON_treshold=0.05):
        VOVON=self.VO.multiply_by(self.VON.multiply_by(-1).add(1),name="VO"r'$\bullet(1-$'"VON"r'$)$'" (dunes)").normalize_positive().add(-VOVON_treshold).multiply_by(1/(1-VOVON_treshold))
        LoG_crest_detector=self.LoG.normalize_positive()
        return mask.multiply_by(VOVON.add(LoG_crest_detector).multiply_by(0.5),name="Crest detector").setup_for_skeletonization()

class predictions(raster_data):
    
    def cleaning_up_small_mess(self,size_treshold=225,hole_treshold=None,verbose=True,name=None,holes_too=True):
        
        if hole_treshold is None:
            hole_treshold=size_treshold 
        tck=time.time()
        if verbose:
            print("Starting cleaning up")
        self.data=morphology.remove_small_objects(self.data.astype(bool),size_treshold).astype(int)
        if holes_too:
            self.data=morphology.remove_small_holes(self.data.astype(bool),hole_treshold).astype(int)
        if verbose:
            print("Cleaning up completed in {:.2f} s".format(time.time()-tck))

    def buffered(self,buffer_size,verbose=True, name=None):

        if name is None:
            name=self.name+" buffered"
    
        tck=time.time()
        if verbose:
            print("Starting buffering")
        rect_size=int((2*buffer_size+1)/(2**0.5))
        buffered = predictions(from_path=False,input_path="",raster=morphology.dilation(self.data,morphology.rectangle(rect_size,rect_size)),profile=self.profile,nodata_value=self.nodata,name=name,treshold_val=self.treshold_val)
        if verbose:
            print("Buffering completed in {:.2f} s".format(time.time()-tck))
        return buffered

        
