import models.base
import models.cnn_1d
import models.cnn_1d_atrous
import models.cnn_1d_butterfly
import models.cnn_2d
import models.linear_bc
import models.linear_bnc
import models.rnn_lstm
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
  assert False

def get_all_1d_models():
  return [
    models.linear_bnc.Linear,
    models.cnn_1d.Cnn,
    models.cnn_1d_atrous.Cnn,
    models.cnn_1d_butterfly.Cnn,
    models.rnn_lstm.Rnn,
    models.transformer.Transformer,
  ]