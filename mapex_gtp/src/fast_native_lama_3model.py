"""Three-model cropped LaMa predictor, with native 500-cell planner padding retained.
This is a NEW numerical variant; not decision-parity-equivalent to original big-G inference.
"""
import os,time,json
import numpy as np
import torch
from lama_pred_utils import convert_obsimg_to_model_input
def centered_padding(input_n,divisor=16):
    total=(divisor-int(input_n)%divisor)%divisor
    return total//2,total-total//2
def align_small_map_to_big(obs_full,small_array,big_shape,native_pad=500,
                           fill_value=1.0):
    """Center-place tiny model response into original native global predictor frame.
    Original 500-cell outer perimeter has no physical environment geometry.
    """
    a=np.asarray(obs_full)
    if a.ndim!=2 or a.shape[0]<=2*native_pad or a.shape[1]<=2*native_pad:
        raise ValueError("native padded obs invalid")
    h,w=a.shape[0]-2*native_pad,a.shape[1]-2*native_pad
    pt,pb=centered_padding(h)
    pl,pr=centered_padding(w)
    expected=(h+pt+pb,w+pl+pr)
    if np.shape(small_array)!=expected:raise ValueError("tiny tensor mismatch")
    full_pt,_=centered_padding(a.shape[0])
    full_pl,_=centered_padding(a.shape[1])
    y0=native_pad+full_pt-pt
    x0=native_pad+full_pl-pl
    assert y0>=0 and x0>=0
    assert y0+expected[0]<=big_shape[0] and x0+expected[1]<=big_shape[1]
    out=np.full(big_shape,float(fill_value),dtype=np.float32)
    out[y0:y0+expected[0],x0:x0+expected[1]]=np.asarray(small_array,dtype=np.float32)
    assert np.array_equal(out[native_pad+full_pt:native_pad+full_pt+h,
                              native_pad+full_pl:native_pad+full_pl+w],
                          np.asarray(small_array,dtype=np.float32)[pt:pt+h,pl:pl+w])
    return out,{"small_shape":expected,"global_shape":tuple(big_shape),
                "small_pad_top":pt,"small_pad_left":pl,
                "global_pad_top":full_pt,"global_pad_left":full_pl,
                "paste_top":y0,"paste_left":x0,
                "core_global_top":native_pad+full_pt,
                "core_global_left":native_pad+full_pl,"core_shape":(h,w)}
def predict_three_models_cropped(obs_full,models,transform,padded_obs,
                                  device="cpu",native_pad=500):
    if len(models)!=3:raise ValueError("Must use exactly G1,G2,G3")
    source=np.asarray(obs_full)
    h,w=source.shape
    cropped=np.ascontiguousarray(source[native_pad:h-native_pad,native_pad:w-native_pad])
    if cropped.size==0 or not np.isin(cropped,[0.,.5,1.]).all():
        raise ValueError("native raw observed core invalid")
    image=np.stack((cropped,cropped,cropped),axis=2)
    batch,mask=convert_obsimg_to_model_input(image,transform,device)
    before=time.perf_counter()
    with torch.no_grad():
        outputs=[]
        for model in models:
            p=model(batch)
            x=p["inpainted"][0,0].detach()
            outputs.append(x)
        pred=torch.stack(outputs,axis=0)
        mean_small=torch.mean(pred,dim=0).cpu().numpy().astype(np.float32)
        var_small=torch.var(pred,dim=0).cpu().numpy().astype(np.float32)
    elapsed=time.perf_counter()-before
    prob_small=np.clip(np.nan_to_num(mean_small,nan=1.,posinf=1.,neginf=0.),0,1)
    variance_small=np.maximum(0,np.nan_to_num(var_small,nan=0.,posinf=0.,neginf=0.))
    big_shape=np.asarray(padded_obs).shape
    mean_big,geometry=align_small_map_to_big(source,prob_small,big_shape,native_pad,fill_value=1)
    var_big,_=align_small_map_to_big(source,variance_small,big_shape,native_pad,fill_value=0)
    # Use ensemble mean for occupancy instead of the removed separate big-G predictor.
    # Conservative 1=occupied outside measured ROI: no synthetic information gain.
    pred_occ=(mean_big>=0.5).astype(np.float32)
    # Known measurements have priority over predictions.
    y,x=geometry["core_global_top"],geometry["core_global_left"]
    sub=pred_occ[y:y+cropped.shape[0],x:x+cropped.shape[1]]
    sub[cropped==1]=1
    sub[cropped==0]=0
    # The mean is NOT force overwritten because native scorer's probability
    # semantics use the raw ensemble mean. This is an explicit numerical change.
    info={"model_count":3,"checkpoint_names":["G1","G2","G3"],
          "big_G_model_removed":True,"model_tensor_shape":tuple(batch["image"].shape),
          "forecast_time_s":round(elapsed,5),"overlay":geometry,
          "outer_pred_occupied":1.0,"outer_mean_occupied_probability":1.0,
          "outer_variance":0.0,
          "pred_occ_method":"ensemble_mean >= 0.5; known obs overrides",
          "numeric_parity_with_original":False}
    return pred_occ,mean_big,torch.from_numpy(var_big),info
