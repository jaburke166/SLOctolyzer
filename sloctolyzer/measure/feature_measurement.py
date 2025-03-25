import numpy as np
import matplotlib.pyplot as plt
import cv2
import bottleneck as bn
import numba
from scipy.spatial import KDTree
from skimage import morphology
from skimage.draw import polygon

def boxcount(Z, k):
    S = np.add.reduceat(np.add.reduceat(Z,
                        np.arange(0, Z.shape[0], k), axis=0),
                        np.arange(0, Z.shape[1], k), axis=1)
    return len(np.where((S > 0) & (S < k*k))[0])



def fractal_dimension(Z):

    p = min(Z.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]



def global_metrics(vessel_, skeleton):

    vessel_ = (vessel_ > 0).astype(int)
    skeleton = (skeleton > 0).astype(int)
        
    fractal_d = fractal_dimension(vessel_)
    global_width = np.sum(vessel_)/np.sum(skeleton)
    
    return fractal_d, global_width



def Knudtson_cal(w1,w2):
    w_artery = 0.88*np.sqrt(np.square(w1) + np.square(w2)) 
    w_vein = 0.95*np.sqrt(np.square(w1) + np.square(w2)) 
    return w_artery, w_vein



def _distance_2p(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5



def _curve_length(x, y):
    return np.sum(((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2) ** 0.5)



def _chord_length(x, y):
    return _distance_2p(x[0], y[0], x[len(x) - 1], y[len(y) - 1])



def _detect_inflection_points(x, y):
    cf = np.convolve(y, [1, -1])
    inflection_points = []
    for i in range(2, len(x)):
        if np.sign(cf[i]) != np.sign(cf[i - 1]):
            inflection_points.append(i - 1)
    return inflection_points



def tortuosity_density(x, y, curve_length):
    inflection_points = _detect_inflection_points(x, y)
    n = len(inflection_points)
    if not n:
        return 0
    starting_position = 0
    sum_segments = 0
    
    # we process the curve dividing it on its inflection points
    for in_point in inflection_points:
        segment_x = x[starting_position:in_point]
        segment_y = y[starting_position:in_point]
        seg_curve = _curve_length(segment_x, segment_y)
        seg_chord = _chord_length(segment_x, segment_y)
        if seg_chord:
            sum_segments += seg_curve / seg_chord - 1
        starting_position = in_point

    # return ((n - 1)/curve_length)*sum_segments  # This is the proper formula
    return (n - 1)/n + (1/curve_length)*sum_segments # This is not



def _refine_coords(coords: list[np.ndarray], dtype: type = np.int16):
    return [_refine_path(c).astype(dtype) for c in coords]



def _refine_path(data: np.ndarray, window: int = 4):
    # Simple moving average
    return bn.move_mean(data, window=window, axis=0, min_count=1)


    
def _compute_vessel_edges(coords: list[np.ndarray], dist_map: np.ndarray):
    edges1 = []
    edges2 = []
    for path in coords:
        x, y = path[:,0], path[:,1]
        delta = np.gradient(path, axis=0)
        angles = np.arctan2(delta[:,1], delta[:,0])
        d = dist_map[x, y]
        offset_x = d * np.cos(angles + np.pi/2)
        offset_y = d * np.sin(angles + np.pi/2)
        x_edge1 = x + offset_x
        y_edge1 = y + offset_y
        x_edge2 = x - offset_x
        y_edge2 = y - offset_y
        edges1.append(np.stack([x_edge1, y_edge1], axis=1))
        edges2.append(np.stack([x_edge2, y_edge2], axis=1))
        
    return edges1, edges2


    
def _calculate_vessel_widths(mask, coords):
    
    # Refine coordinates
    coords_refined = _refine_coords(coords) # dtype = np.int16
    
    # Distance transform
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # Get diameter of refined vessel skeleton
    vessel_map = np.zeros_like(dist, dtype=bool)
    c_r = np.concatenate(coords_refined, axis=0)
    vessel_map[c_r[:,0], c_r[:,1]] = True
    vessel_map = dist * vessel_map + 2.0
    
    # Calculate edges of the vessels
    edges1, edges2 = _compute_vessel_edges(coords_refined, vessel_map)
    
    # Binary image of pixels within the edges
    mask_edges = np.zeros_like(mask, dtype=bool)
    for edge1, edge2 in zip(edges1, edges2):
        combined = np.vstack((edge1, edge2[::-1]))
        rr, cc = polygon(combined[:, 0], combined[:, 1], shape=mask.shape)
        mask_edges[rr, cc] = True
        
    # AND with segmentation mask
    mask_edges = mask_edges & (mask > 0)
    
    # Identify the edges of the vessels (Canny edge detection)
    mask_edges = cv2.Canny(mask_edges.astype(np.uint8), 0, 1)
    
    # Locate edges in the original image for each vessel
    on_pixels = np.argwhere(mask_edges).astype(np.float32)
    tree = KDTree(on_pixels)
    edges1 = [on_pixels[tree.query(e)[1]] for e in edges1]
    edges2 = [on_pixels[tree.query(e)[1]] for e in edges2]
    
    # Calculate vessel width at each point + average width
    widths = [np.linalg.norm(e1 - e2, axis=1) for e1, e2 in zip(edges1, edges2)]
    avg_width = [np.mean(w, dtype=float) for w in widths]
    
    return avg_width




def vessel_metrics(vessels,
                   vessel_coords,
                   roi_masks,
                   scale,
                   min_pixels_per_vessel: int = 10, 
                   vessel_type: str = "binary"):
    """
    Re-write of tortuosity_measures.evaluate_window() to include only necessary code.
    """
    # collect outputs in a dictionary
    vessels = (vessels > 0).astype(np.uint8)
    slo_dict = {'whole':{}, 'B':{}, 'C':{}}
    logging_list = []

    # Number of vessel pixels
    vessel_total_count = np.sum(vessels==1) 
    pixel_total_count = vessels.shape[0]*vessels.shape[1]
    vessel_density = vessel_total_count / pixel_total_count

    # Compute FD, VD and Average width over whole image
    skeleton = morphology.skeletonize(vessels)
    fractal_dimension, average_width_all = global_metrics(vessels, skeleton) 
    slo_dict['whole']["fractal_dimension"] = fractal_dimension
    slo_dict['whole']["vessel_density"] = vessel_density
    slo_dict['whole']["average_global_calibre"] = average_width_all*scale
    
    for roi_type, mask in roi_masks.items():

        # Get zonal vessel map
        zonal_mask = (vessels * mask).astype(np.uint8)
        
        # Loop over windows
        tcurve = 0
        tcc = 0
        td = 0
        
        # Initialise vessel widths and count lists
        vessel_count = 0
        zonal_vessels = []
        for i, vessel in enumerate(vessel_coords):

            if roi_type in ['B', 'C']:
                idx_in_zone = np.where(zonal_mask[vessel[:,0], vessel[:,1]])[0]
                if idx_in_zone.shape[0] < min_pixels_per_vessel:
                    continue
                else:
                    vessel = vessel[idx_in_zone]
            vessel_count += 1
            zonal_vessels.append(vessel)
                         
            # Work out length of current vessel
            vessel = vessel.T
            N = len(vessel[0])
            v_length = _curve_length(vessel[0], vessel[1])
            c_length = _chord_length(vessel[0], vessel[1])
            tcc += v_length / c_length
                    
            # tcurve is simply the pixel length of the vessel
            tcurve += v_length
            
            # td measures curve_chord_ratio for subvessel segments per inflection point 
            # and cumulatively add them, and scale by number of inflections and overall curve length
            # formula comes from https://ieeexplore.ieee.org/document/1279902
            td += tortuosity_density(vessel[0], vessel[1], v_length)

        # Normalise tortuosity density and tortuosity distance by vessel_count
        td = td/vessel_count
        tcc = tcc/vessel_count
    
        # This is measuring the same thing as average_width computed in global_metrics, but should be smaller as 
        # individual vessel segments exclude branching points in their calculation
        all_vessel_widths = _calculate_vessel_widths(zonal_mask, zonal_vessels)
        local_caliber = np.mean(all_vessel_widths)
        
        # collect outputs
        slo_dict[roi_type]["tortuosity_density"] = td
        slo_dict[roi_type]['tortuosity_distance'] = tcc
        slo_dict[roi_type]["average_local_calibre"] = local_caliber*scale

        # Do not calculate CRAE/CRVE if binary vessels.
        #print('CRAE/CRVE')
        if (vessel_type != "binary") & (roi_type in ['B', 'C']):   
        
            # calculate the CRAE/CRVE with Knudtson calibre
            vtype = vessel_type[0].upper()
            sorted_vessel_widths_average = sorted(all_vessel_widths)[-6:]
            N_vessels = len(sorted_vessel_widths_average)
        
            # Error handle if detected less than 6 vessels, must be even number
            if N_vessels < 6:
                msg1 = f'        WARNING: Less than 6 vessels detected in zone. Please check segmentation. Returning -1 for CR{vtype}E.'
                msg2 = f'                 Note that this means AVR cannot be computed for this image'
                slo_dict[roi_type]["CRAE_Knudtson"] = -1
                slo_dict[roi_type]["CRVE_Knudtson"] = -1
        
                # log to user
                print(msg1)
                print(msg2)
                logging_list.append(msg1)
                logging_list.append(msg2)
        
            #  Compute calibre, taking into account number of available vessels
            else:
                
                w_first_artery_Knudtson_1, w_first_vein_Knudtson_1 = Knudtson_cal(sorted_vessel_widths_average[0],
                                                                                  sorted_vessel_widths_average[5])
                
                w_first_artery_Knudtson_2, w_first_vein_Knudtson_2 = Knudtson_cal(sorted_vessel_widths_average[1],
                                                                                  sorted_vessel_widths_average[4])
                    
                w_first_artery_Knudtson_3, w_first_vein_Knudtson_3 = Knudtson_cal(sorted_vessel_widths_average[2],
                                                                                  sorted_vessel_widths_average[3])
                
                CRAE_first_round = sorted([w_first_artery_Knudtson_1,
                                           w_first_artery_Knudtson_2,
                                           w_first_artery_Knudtson_3])
                CRVE_first_round = sorted([w_first_vein_Knudtson_1,
                                           w_first_vein_Knudtson_2,
                                           w_first_vein_Knudtson_3])
                
                if vessel_type=='artery': 
                    w_second_artery_Knudtson_1, w_second_vein_Knudtson_1 = Knudtson_cal(CRAE_first_round[0],
                                                                                        CRAE_first_round[2])  
                
                    CRAE_second_round = sorted([w_second_artery_Knudtson_1,CRAE_first_round[1]])
                    CRAE_Knudtson,_ = Knudtson_cal(CRAE_second_round[0],CRAE_second_round[1])
                    slo_dict[roi_type]["CRAE_Knudtson"] = CRAE_Knudtson*scale
                    slo_dict[roi_type]["CRVE_Knudtson"] = -1
                
                else:
                    w_second_artery_Knudtson_1, w_second_vein_Knudtson_1 = Knudtson_cal(CRVE_first_round[0],
                                                                                        CRVE_first_round[2])  
                
                    CRVE_second_round = sorted([w_second_vein_Knudtson_1,CRVE_first_round[1]])
                    _,CRVE_Knudtson = Knudtson_cal(CRVE_second_round[0],CRVE_second_round[1])
                    slo_dict[roi_type]["CRAE_Knudtson"] = -1
                    slo_dict[roi_type]["CRVE_Knudtson"] = CRVE_Knudtson*scale
                
        else:
            slo_dict[roi_type]["CRAE_Knudtson"] = -1
            slo_dict[roi_type]["CRVE_Knudtson"] = -1
        
        
    return slo_dict