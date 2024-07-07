import models.base
import models.cnn_1d
import models.cnn_1d_atrous
import models.cnn_1d_butterfly
import models.cnn_1d_fft
import models.cnn_2d
import models.linear_bc
import models.linear_bnc
import models.rnn_lstm
import models.seq_cnn
import models.seq_tcn
import models.attn_falcon
import models.transformer

def lookup_model(name):
  if name == "cnn":
    return models.cnn_1d.Cnn
  elif name == "atrous":
    return models.cnn_1d_atrous.Cnn
  elif name == "butterfly":
    return models.cnn_1d_butterfly.Cnn
  elif name == "2d":
    return models.cnn_2d.Cnn
  elif name == "0d":
    return models.linear_bc.Linear
  elif name == "linear":
    return models.linear_bnc.Linear
  elif name == "lstm":
    return models.rnn_lstm.Rnn
  elif name == "transformer":
    return models.transformer.Transformer
  elif name == "falcon":
    return models.attn_falcon.Transformer
  elif name == "causal":
    return models.seq_cnn.Cnn
  elif name == "fft":
    return models.cnn_1d_fft.Cnn
  elif name == "tcn":
    return models.seq_tcn.Cnn
  assert False, name

def get_all_1d_models():
  return [
    models.linear_bnc.Linear,
    #models.cnn_1d.Cnn,
    #models.cnn_1d_atrous.Cnn,
    models.cnn_1d_butterfly.Cnn,
    models.rnn_lstm.Rnn,
    models.transformer.Transformer,
    models.attn_falcon.Transformer,
    models.seq_cnn.Cnn,
    models.cnn_1d_fft.Cnn,
  ]