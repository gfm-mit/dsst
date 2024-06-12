import gtorch.models.base
import gtorch.models.cnn_1d
import gtorch.models.cnn_1d_atrous
import gtorch.models.cnn_1d_butterfly
import gtorch.models.cnn_2d
import gtorch.models.linear_bc
import gtorch.models.linear_bnc
import gtorch.models.rnn_lstm
import gtorch.models.transformer
import gtorch.models.transformer_fft

def lookup_model(name):
  if name == "cnn":
    return gtorch.models.cnn_1d.Cnn
  elif name == "atrous":
    return gtorch.models.cnn_1d_atrous.Cnn
  elif name == "butterfly":
    return gtorch.models.cnn_1d_butterfly.Cnn
  elif name == "2d":
    return gtorch.models.cnn_2d.Cnn
  elif name == "0d":
    return gtorch.models.linear_bc.Linear
  elif name == "linear":
    return gtorch.models.linear_bnc.Linear
  elif name == "lstm":
    return gtorch.models.rnn_lstm.Rnn
  elif name == "transformer":
    return gtorch.models.transformer.Transformer
  elif name == "fft":
    return gtorch.models.transformer_fft.Transformer
  assert False

def get_all_1d_models():
  return [
    gtorch.models.linear_bnc.Linear,
    gtorch.models.cnn_1d.Cnn,
    gtorch.models.cnn_1d_atrous.Cnn,
    gtorch.models.cnn_1d_butterfly.Cnn,
    gtorch.models.rnn_lstm.Rnn,
    gtorch.models.transformer.Transformer,
    gtorch.models.transformer_fft.Transformer,
  ]