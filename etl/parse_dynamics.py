import numpy as np
import scipy.signal
import shutil

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
      v_bar=nanpad_convolve(stroke, [1, 0, -1], 2),
      v_mag2=convolve_and_pool_norm2(stroke, [1, -1], 1),
      a_bar=nanpad_convolve(stroke, [1, -2, 1], 2),
      # a has an odd kernel, no need to average magnitudes
      dv_mag2=convolve_and_pool_norm2(stroke, [1, -1], 1, [.5, -.5]),
      j_bar=nanpad_convolve(stroke, [1, -2, 0, 2, -1], 12),
      j_mag2=convolve_and_pool_norm2(stroke, [1, -3, 3, -1], 72),
  )
  retval['cw'] = curvature(retval)
  retval['a_mag2'] = np.sum(retval['a_bar']**2, axis=-1)
  if stroke.shape[0] < 3:
    for k in retval.keys():
      shape = retval[k].shape
      shape = [stroke.shape[0]] if len(shape) == 1 else [stroke.shape[0], shape[1]]
      retval[k] = np.full(shape, np.nan)
  return retval

def get_pause(raw):
  strokes = raw.groupby("t_min").t.max().sort_index()
  pause = strokes.index.values[1:] - strokes.values[:-1]
  pause = pd.Series(pause, index=strokes.index.values[1:])
  pause.loc[0] = 0
  pause = pause.sort_index()
  return pause

def convert():
  try:
    shutil.rmtree('TS')
  except FileNotFoundError:
    pass
  for f in tqdm(list(Path('DIGITS').glob('*.csv'))):
    raw = pd.read_csv(f)
    pause = get_pause(raw)
    pkey = f.name.replace('.csv', '')
    for (symbol, task, iteration), g in raw.groupby('symbol task iteration'.split()):
      accum = []
      if True:
        for t_min, g2 in g.groupby('t_min'):
          d = dynamics(g2["x y".split()])
          try:
            munged = pd.DataFrame({
                k: d[k]
                for k in "v_mag2 a_mag2 dv_mag2 cw j_mag2".split()
                if k in d
            })
          except ValueError:
            assert False, (g2, d)
          munged["pause"] = pause.loc[t_min]
          munged.v_mag2 *= 1.5e4
          munged.a_mag2 *= 3e5
          munged.dv_mag2 *= 4e4
          munged.j_mag2 *= 5e6
          munged.pause = np.log10(1+munged.pause) - 4.1
      accum += [munged]
      munged = pd.concat(accum)
      f_out = Path('TS') / pkey / str(symbol) / task / "{}.npy".format(iteration)
      f_out.parent.mkdir(exist_ok=True, parents=True)
      np.save(f_out, munged.values)
convert()