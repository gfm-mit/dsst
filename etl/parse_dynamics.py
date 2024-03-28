import numpy as np
import scipy.signal
import shutil
import pandas as pd
import matplotlib.pyplot as plt

def convolve_and_pool_norm2(x, k, w, k2=[.5, .5]):
  nda_k = np.array(k)[:, np.newaxis]
  conv = scipy.signal.convolve2d(x, nda_k, 'same', fillvalue=np.nan)
  norm2 = np.sum(conv**2, axis=-1)
  avg = np.roll(np.convolve(norm2, k2, 'same'), -1)
  return avg / w

def nanpad_convolve(x, k, w):
  nda_k = np.array(k)[:, np.newaxis]
  conv = scipy.signal.convolve2d(x, nda_k, 'same', fillvalue=np.nan)
  return conv / w

def curvature(dynamics):
  a_cross_v = np.cross(dynamics['a_bar'], dynamics['v_bar'])
  v_bar3 = np.linalg.norm(dynamics['v_bar'], axis=-1) ** 3
  with np.errstate(divide='ignore', invalid='ignore'):
    curvature = a_cross_v / v_bar3
  curvature[v_bar3 == 0] = 0
  with np.errstate(divide='ignore', invalid='ignore'):
    log_curvature = np.sign(curvature) * (1+np.log10(np.abs(curvature)))
  curvature[np.abs(curvature) > 1] = log_curvature[np.abs(curvature) > 1]
  return curvature

def dynamics(stroke):
  retval = dict(
      v_bar=nanpad_convolve(stroke["x y".split()], [1, 0, -1], 2),
      v_mag2=convolve_and_pool_norm2(stroke["x y".split()], [1, -1], 1),
      a_bar=nanpad_convolve(stroke["x y".split()], [1, -2, 1], 2),
      # a has an odd kernel, no need to average magnitudes
      dv_mag2=convolve_and_pool_norm2(stroke["x y".split()], [1, -1], 1, [.5, -.5]),
      j_bar=nanpad_convolve(stroke["x y".split()], [1, -2, 0, 2, -1], 12),
      j_mag2=convolve_and_pool_norm2(stroke["x y".split()], [1, -3, 3, -1], 72),
  )
  retval['cw'] = curvature(retval)
  retval['a_mag2'] = np.sum(retval['a_bar']**2, axis=-1)
  if stroke.shape[0] < 3:
    for k in retval.keys():
      shape = retval[k].shape
      shape = [stroke.shape[0]] if len(shape) == 1 else [stroke.shape[0], shape[1]]
      retval[k] = np.full(shape, np.nan)
  munged = pd.DataFrame({
     k: retval[k]
      for k in "v_mag2 a_mag2 dv_mag2 cw j_mag2".split()
      if k in retval
  }, index=stroke.index.rename("row_number"))
  return munged

def convert(raw):
  dyn = raw.groupby("box").apply(dynamics).reset_index().set_index("row_number")
  for k in "symbol task x y t".split():
    dyn[k] = raw[k]
  # magic numbers from old undocumented code, look roughly right
  for k, v in dict(
    v_mag2=1.5e4,
    a_mag2=3e5,
    j_mag2=5e6,
    #dv_mag2=4e4,
  ).items():
    dyn[k] = np.log(1e-4 + dyn[k] * v)
  # eyeballed from the data
  dyn.dv_mag2 = np.select(
    [dyn.dv_mag2 > 1e-9, dyn.dv_mag2 < -1e-9],
    [9 + np.log10(dyn.dv_mag2), -9 - np.log10(-dyn.dv_mag2)],
    default=0) / 2
  return dyn