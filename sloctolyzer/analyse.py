import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pandas as pd
import numpy as np
import cv2
import copy
from PIL import Image, ImageOps
from pathlib import Path, WindowsPath, PosixPath
from sloctolyzer import utils
from sloctolyzer.measure import get_vessel_coords, feature_measurement
from sloctolyzer.segment import slo_inference, avo_inference, fov_inference

# Function to analyse a single IR-SLO image
def analyse(path, 
            save_path=None, 
            scale=None,
            location=None,
            eye=None,
            slo_model=None, 
            avo_model=None, 
            fov_model=None,
            save_results=True,
            save_images=False,
            collate_segmentations=True,
            compute_metrics=True,
            verbose=True,
            segmentation_dict={},
            demo_return=False):
    """
    Inner function to analyse an individual IR-SLO img, given options for scaling/location/eye.

    Inputs:
    -------------------
    path (str) : Path to image/vol file.

    save_path (str) : Path to save results to. Will be created if doesn't exist

    scale (str) : Microns-per-pixel conversion factor from pixel space to physical space.
    
    location (str) : Either 'Macular' or 'Optic disc' centred. 
    
    Eye (str) : Either 'Right' or 'Left'.

    slo_model, avo_model, fov_model : binary, artery-vein-optic disc, and fovea detection models.

    save_results (bool) : Flag to save log and xlsx output file with feature measurements and metadata.

    save_images (bool) : Flag to save segmentation masks and segmentations superimposed onto SLO.

    collate_segmentations (bool) : Flag to save the superimposed segmentation image to a global directory, 
                                   assuming batch analysis.

    compute_metrics (bool) : Flag to compute feature measurements of the retinal vessels.

    segmentation_dict (dict) : Dictionaries with new segmentation to recompute measurements. By default left empty UNLESS
                               correcting manual segmentations.
    """
    # Initialise list of messages to save
    logging_list = []
    metadata = {}

    # load image, check to make sure is path
    if isinstance(path, (str, WindowsPath, PosixPath)):

        # check if vol file, otherwise is regular image file
        ftype = str(path).split('.')[-1]
        if ftype.lower() == 'vol':
            slo, meta, log = utils.load_volfile(path, verbose=verbose, logging=[])
            eye = meta['eye']
            scale = meta['scale']
            location = meta['location']
            metadata = copy.deepcopy(meta)
            logging_list.extend(log)
        else:
            slo = np.array(ImageOps.grayscale(Image.open(path)))

    elif isinstance(path, np.ndarray):
        slo = path.copy()
        path = save_path   
    else:
        msg = "Unknown filetype, must be either string/filepath/numpy array."
        logging_list.append(msg)
        if verbose:
            print(msg)
        return
    img_shape = slo.shape

    # accuounting for image files saved from HEYEX with info bar at bottom of image which is 100 rows.
    if slo.shape[0]==1636 or slo.shape[0]==868:
        slo = slo[:img_shape[0]-100]
        img_shape = slo.shape
    _, N = img_shape

    # Collect metadata for recomputing measurements when a manual annotation is provided
    segmented_already = False
    if 'metadata' in segmentation_dict:
        segmented_already = True
        metadata = copy.deepcopy(segmentation_dict['metadata'])
        fovea = np.array([metadata['fovea_x'], metadata['fovea_y']]).astype(int)
        location = metadata['location']
        eye = metadata['eye']
        
        # Extract segmentation masks, recompute OD centre and OD radius
        slo_avimout = segmentation_dict['avod_map']
        slo_vbinmap = segmentation_dict['binary_map']
        od_mask = slo_avimout[...,1]
        od_centre = avo_inference._get_od_centre(od_mask)
        if location == 'Optic disc':
            od_radius, od_boundary = utils._process_opticdisc(od_mask)
            metadata['optic_disc_x'] = od_centre[0]
            metadata['optic_disc_y'] = od_centre[1]
            metadata['optic_disc_radius_px'] = od_radius
        else:
            od_centre = None

    # compatability with mac, linux and windows
    dirpath = save_path
    if isinstance(Path(path), PosixPath):
        split_path = str(path).split('/')
    elif isinstance(Path(path), WindowsPath):
        split_path = str(path).split('\\')
    fname_type = split_path[-1]
        
    # Get filename and log to user SLO is being analysed
    fname = fname_type.split(".")[0]
    metadata['Filename'] = fname_type
    msg = f"\nAnalysing {fname}."
    logging_list.append(msg)
    if verbose:
        print(msg)

    # Error handle scale
    if scale is not None:
        if not isinstance(scale, (float, int)):
            msg = f"Pixel lengthscale {scale} should be a float or integer. Ignoring scale and measuring in pixels."
            logging_list.append(msg)
            if verbose:
                print(msg)
            scale = None
        if (scale > 20) or (scale < 3):
            msg = f"Pixel lengthscale {scale} should be in [3,20] microns-per-pixel. Is your scale in mm-per-pixel?. Ignoring scale and measuring in pixels."
            logging_list.append(msg)
            if verbose:
                print(msg)
            scale = None

    # Error handle save_path
    if (save_results+save_images)>0:
        if save_path is None:
            msg = f"Path {save_path} is not specified, but option to save is flagged. Creating directory 'output' in current working directory."
            logging_list.append(msg)
            if verbose:
                print(msg)
            save_path = "output"
            os.mkdir(save_path)
              
        else: 
            if not os.path.exists(save_path):
                msg = f"Path {save_path} does not exist. Creating directory."
                logging_list.append(msg)
                if verbose:
                    print(msg)
                os.mkdir(save_path)

        # Create fname directory
        save_path = os.path.join(save_path, fname)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
                
    # Save out SLO image
    if slo.max() == 1:
        slo_save = (255*slo).astype(np.uint8)
    else:
        slo_save = slo.copy()
    if save_images:
        cv2.imwrite(os.path.join(save_path,f"{fname}_slo.png"), slo_save)

    # SEGMENTING
    if not segmented_already:
        msg = "\nSEGMENTING..."
        
        logging_list.append(msg)
        if verbose:
            print(msg)
    
        # Forcing model instantiation if unspecified
        # SLO segmentation models
        if slo_model is None or type(slo_model) != slo_inference.SLOSegmenter:
            msg = "Loading models..."
            logging_list.append(msg)
            if verbose:
                print(msg)
            slo_model = slo_inference.SLOSegmenter()
         # SLO segmentation models
        if fov_model is None or type(fov_model) != fov_inference.FOVSegmenter:
            fov_model = fov_inference.FOVSegmenter()
        # AVO segmentation models
        if avo_model is None or type(avo_model) != avo_inference.AVOSegmenter:
            avo_model = avo_inference.AVOSegmenter()

        # binary vessel detection
        msg = "    Segmenting binary vessels from SLO image."
        logging_list.append(msg)
        if verbose:
            print(msg)
        slo_vbinmap = slo_model.predict_img(slo)

        # fovea detection
        msg = "    Segmenting fovea from SLO image."
        logging_list.append(msg)
        if verbose:
            print(msg)
        fmask, fovea = fov_model.predict_img(slo)
        if save_images:
            cv2.imwrite(os.path.join(save_path,f"{fname}_slo_fovea_map.png"), 
                        (255*fmask).astype(np.uint8))

        # artery-vein-optic disc detection, using binary vessel detector as original reference
        # We also reassigh the binary vessel map as artery+vein maps
        msg = "    Segmenting artery-vein vessels and optic disc from SLO image."
        logging_list.append(msg)
        if verbose:
            print(msg)
        slo_avimout, od_centre = avo_model.predict_img(slo, location=location)#, slo_vbinmap)
        if od_centre is None:
            msg = 'WARNING: Optic disc not detected. Please check image.'
            logging_list.append(msg)
            if verbose:
                print(msg)
        od_mask = slo_avimout[...,1]

    # Attempt to resolve location if not inputted
    msg = "\n\nInferring image metadata..."
    logging_list.append(msg)
    if verbose:
        print(msg)
    metadata["location"] = location
    if location is None:
        
        # We check whether x-location of optic disc centre is in the main
        # area of the image, i.e. in [0.1N, 0.9N] where N is image width
        location = "Macula"
        if od_centre is not None:
            if 0.1*img_shape[1] < od_centre[0] < 0.9*img_shape[1]:
                location = "Optic disc"
        msg = f"    No location specified. Detected SLO image to be {location.lower()}-centred."
        logging_list.append(msg)
        if verbose:
            print(msg)
        metadata["location"] = location
    else:
        msg = f"    Location is specified as {location.lower()}-centred."
        logging_list.append(msg)
        if verbose:
            print(msg)
            
    # If eye unspecified, try to infer
    if eye is None:
        # If fovea known, detect which eye based on position of OD and fovea as first port of call
        if fovea.sum() != 0 and od_centre is not None:
            if fovea[0] < od_centre[0]:
                eye = "Right"
            else:
                eye = "Left"
            msg = f"    Using the position of the fovea and optic disc, we infer it is the {eye} eye."
              
        # if the fovea is not detected (predictions at origin), we must only use optic disc centre
        # positioning wrt image We check whether the optic disc x-location is less than half the image width
        elif od_centre is not None and fovea.sum() == 0:
            ratio = 0.05
            if od_centre[0] < (0.5-ratio)*img_shape[1]:
                eye = "Left"
                msg = f"    The optic disc is nearer the left of the image, and so the SLO image is assumed to be the {eye} eye. Please check."
            elif od_centre[0] > (0.5+ratio)*img_shape[1]:
                eye = "Right"
                msg = f"    The optic disc is nearer the right of the image, and so the SLO image is assumed to be the {eye} eye. Please check."
            elif (0.5-ratio)*img_shape[1] < od_centre[0] < (0.5+ratio)*img_shape[1]:
                msg = "    The optic disc is near the centre of the image, "
                if fovea.sum() == 0 and od_centre[0] < 0.5*img_shape[1]:
                    eye = "Left"
                    msg += f" and no fovea is detected. The SLO image is assumed as the {eye} eye. Please check."
                elif fovea.sum() == 0 and od_centre[0] > 0.5*img_shape[1]:
                    eye = "Right"
                    msg += f" and no fovea is detected. The SLO image is assumed as the {eye} eye. Please check."   
        
        # If all else fails, we check where the largest amount of vasculature is - i.e. should be around the disc which is
        # typically off-centre which tells us which eye it might be. Last chance saloon.
        elif od_centre is None:
            msg = '    Detecting eye based on which half of image has highest proportion of vessel pixels.'
            if slo_vbinmap[:, :N//2].sum() >= slo_vbinmap[:, N//2:].sum():
                eye = 'Left'
            else:
                eye = 'Right'
            msg += f" Thus, the SLO image is assumed as the {eye} eye. Please check."
        logging_list.append(msg)
        if verbose:
            print(msg)   

    # Eye provided in the metadata
    else:
        msg = f"    Eye type is specified as the {eye} eye."
        logging_list.append(msg)
        if verbose:
            print(msg)
    metadata["eye"] = eye
    metadata['manual_annotation'] = segmented_already

    # catch any potential fovea detections at origin, i.e. model did not predict fovea anywhere
    fovea_missing = False
    if fovea.sum() == 0:
        # alert user that fovea is missing
        fovea_missing = True
        msg = "Fovea was not detected. Please double-check image."
        logging_list.append(msg)
        if verbose:
            print(msg)  
    metadata["fovea_x"] = fovea[0]
    metadata["fovea_y"] = fovea[1]
    metadata["missing_fovea"] = fovea_missing

    # store optic disc centre,
    # The latter is only stored for for an optic disc-centred scan
    # macular-centred SLO do not show the optic disc entirel
    od_radius, od_boundary = utils._process_opticdisc(od_mask)
    if location == "Optic disc":
        metadata["optic_disc_x"] = od_centre[0]
        metadata["optic_disc_y"] = od_centre[1]
        metadata["optic_disc_radius_px"] = od_radius  
    else:
        od_radius = None
    if scale is None:
        metadata["measurement_units"] = "px"
    else:
        metadata["scale"] = scale
        metadata["measurement_units"] = "microns"
        metadata["scale_units"] = "microns-per-pixel"
    msg = f"Measurements which have units are in {metadata['measurement_units']} units. Otherwise they are non-dimensional."
    logging_list.append(msg)
    if verbose:
        print(msg)

    # Binary vessels are:
    #        pixels detected by the binary vessel detector
    #        - any detected in the optic disc by the AVOD-model
    #        + any missed pixels identified by the AVOD-map.
    # We purposely choose not to fill in any missing AV-pixels using the binary vessel detector as this
    # can lead to many missclassified pixels due to the AV-model's uncertainty.
    # Therefore, the binary vessel map will ALWAYS contain more pixels detected.
    slo_vbinmap = (((slo_vbinmap + (slo_avimout[...,[0,2]].sum(axis=-1) > 0)).astype(bool)) * (1-od_mask)).astype(int)

    # option to save out segmentation masks
    if save_images:
        avoimout_save = 191*slo_avimout[...,0] + 127*slo_avimout[...,2] + 255*slo_avimout[...,1]
        Image.fromarray((255*slo_vbinmap).astype(np.uint8)).save(os.path.join(save_path,f"{fname}_slo_binary_map.png"))
        Image.fromarray((avoimout_save).astype(np.uint8)).save(os.path.join(save_path,f"{fname}_slo_avod_map.png"))

    # FEATURE MEASUREMENTS
    # - macula-centred SLO: Whole image
    # - optic disc-centred SLO: Zone B and C (0.5D-1D, 0D-2D annulus) around optic disc, and whole image
    msg = "\n\nFEATURE MEASUREMENT..."
    logging_list.append(msg)
    if verbose:
        print(msg)
    
    if compute_metrics:

        # Logging
        msg = f"\nMeasuring en-face vessel metrics on {location.lower()}-centred SLO image"
        
        # if scale is None:
        msg += " using the whole image."
        logging_list.append(msg)
        if verbose:
            print(msg)

        if location == 'Optic disc':
            msg = "We will also measure Zones B (0.5-1 OD diameter) and C (2 OD diameter) from optic disc margin."
            logging_list.append(msg)
            if verbose:
                print(msg)

        # Compute vessel metrics
        slo_dict = {}
        slo_keys = ["binary", "artery", "vein"]
        artery_vbinmap, vein_vbinmap = slo_avimout[...,0], slo_avimout[...,2]
        masks = utils.generate_zonal_masks((N,N), od_radius, od_centre, location)

        for v_map, v_type in zip([slo_vbinmap, artery_vbinmap, vein_vbinmap], slo_keys):

            # detect individual vessels, similar to skelentonisation but detects individual vessels, and
            # splits them at any observed intersection
            vcoords = get_vessel_coords.generate_vessel_skeleton(v_map, od_mask, od_centre, min_length=10)
                
            # log to user 
            msg = f"    Measuring {v_type} vessel map"
            logging_list.append(msg)
            if verbose:
                print(msg)

            # Compute features 
            slo_dict[v_type] = feature_measurement.vessel_metrics(v_map,
                                                                vcoords,
                                                                masks,
                                                                scale=scale,
                                                                vessel_type=v_type)

        # Plot the segmentations superimposed onto the SLO
        if save_images or collate_segmentations:

            save_info = [fname, save_path, dirpath, save_images, collate_segmentations]
            zonal_masks = list(masks.values())
            utils.superimpose_segmentation(slo, slo_vbinmap, slo_avimout, 
                                        od_mask, od_centre, 
                                        fovea, location, zonal_masks, save_info)
            

        # Organise measurements of SLO into dataframe
        slo_df = utils.nested_dict_to_df(slo_dict).reset_index()
        slo_df = slo_df.rename({"level_0":"vessel_map", "level_1":"zone"}, axis=1, inplace=False)
        reorder_cols = ["vessel_map", "zone", "fractal_dimension", "vessel_density", "average_global_calibre", 
                        "average_local_calibre", "tortuosity_density", "tortuosity_distance", "CRAE_Knudtson", "CRVE_Knudtson"]
        slo_df = slo_df[reorder_cols]

        # Compute AVR
        slo_df["AVR"] = -1
        all_grids = np.array(list(slo_df.zone.drop_duplicates()))
        avrs = []
        for z in all_grids:
            crae = slo_df[(slo_df.zone==z) & (slo_df.vessel_map=="artery")].iloc[0].CRAE_Knudtson
            crve = slo_df[(slo_df.zone==z) & (slo_df.vessel_map=="vein")].iloc[0].CRVE_Knudtson
            if crae==-1 or crve ==-1:
                avrs.append(-1)
            else:
                avrs.append(crae/crve)
        avrs = np.array(avrs)

        # Outputting warning to user if AVR exceeds 1
        warning_zones = all_grids[avrs > 1]
        if warning_zones.shape[0] > 0:
            if location == "Macula":
                msg = f"WARNING: AVR value exceeds 1, please check artery-vein segmentation."
            elif location == "Optic disc":
                msg = f"WARNING: AVR value exceeds 1 for zones "
                for z in warning_zones[:-1]:
                    msg += f"{z}, "
                msg += f"and {warning_zones[-1]}. Please check artery-vein segmentation."
            if verbose:
                print(msg)
        logging_list.append(msg)

        # add AVR to measurement dataframe
        null_dict = {key:len(all_grids)*[-1] for key in reorder_cols[2:]}
        avr_dict = {**{"vessel_map":len(all_grids)*["artery-vein"], "zone":all_grids},**null_dict, **{"AVR":avrs}}
        avr_df = pd.DataFrame(avr_dict)
        slo_df = pd.concat([slo_df, avr_df], axis=0).reset_index(drop=True)

        # Collect dataframes per zone
        slo_df.loc[slo_df.zone.isin(["B", "C"]), ["fractal_dimension", "vessel_density", "average_global_calibre"]] = -1
        slo_dfs = []
        for z in all_grids:
            df = slo_df[slo_df.zone == z].reset_index(drop=True)
            df = df.iloc[[1,0,2,3]].reset_index(drop=True)
            slo_dfs.append(df)
        
    else:
        msg = f"\n\nSkipping metric calculation."
        if verbose:
            print(msg)
        logging_list.append(msg)
        slo_dfs = None

    # Save out measurements and segmentations
    meta_df = pd.DataFrame(metadata, index=[0])
    if save_results:
        with pd.ExcelWriter(os.path.join(save_path, f'{fname}_output.xlsx')) as writer:
            # write metadata
            meta_df.to_excel(writer, sheet_name='metadata', index=False)
    
            # write SLO measurements
            if slo_dfs is not None: 
                for (df, z) in zip(slo_dfs, all_grids):
                    df.to_excel(writer, sheet_name=f'slo_measurements_{z}', index=False)
    
        msg = f"\n\nSaved out metadata, measurements and segmentations."
        logging_list.append(msg)
        if verbose:
            print(msg)
    
        with open(os.path.join(save_path, f"{fname}_log.txt"), "w") as f:
            for line in logging_list:
                f.write(line+"\n")

    # final log
    msg = f"\n\nCompleted analysis of {fname}.\n\n"
    logging_list.append(msg)
    if verbose:
        print(msg)

    # Return metadata, SLO image, measurements, segmentations and logging
    segmentations = [slo_avimout, slo_vbinmap]
    return meta_df, slo_dfs, slo, segmentations, logging_list
