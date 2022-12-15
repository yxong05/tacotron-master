from .tacotron import Tacotron


def create_model(name):
  if name == 'tacotron':
    return Tacotron()
  else:
    raise Exception('Unknown model: ' + name)
