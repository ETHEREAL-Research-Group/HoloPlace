import torch as th


class OnnxablePolicy(th.nn.Module):
  def __init__(self, extractor, action_net, value_net):
    super().__init__()
    self.extractor = extractor
    self.action_net = action_net
    self.value_net = value_net

  def forward(self, observation):
    # NOTE: You may have to process (normalize) observation in the correct
    #       way before using this. See `common.preprocessing.preprocess_obs`
    action_hidden, value_hidden = self.extractor(observation)
    return self.action_net(action_hidden), self.value_net(value_hidden)


def export_model(policy, observation_space, path='test2.onnx') -> None:
  onnxable_model = OnnxablePolicy(
      policy.mlp_extractor, policy.action_net, policy.value_net
  )

  observation_size = observation_space.shape
  dummy_input = th.randn(*observation_size,)
  th.onnx.export(
      onnxable_model,
      th.ones(dummy_input.shape, dtype=th.float32,
              device=policy.device),
      path,
      opset_version=9,
      input_names=['input'],
      output_names = ['output']
  )
