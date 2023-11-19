from sbi.utils.user_input_checks import wrap_as_pytorch_simulator, get_batch_loop_simulator
from sbi.utils.user_input_checks_utils import MultipleIndependent
from sbi.utils.user_input_checks import process_prior
from torch.distributions import Distribution
from torch import Tensor, tensor, Size
from pymob.sim import priors

def prepare_simulator_for_sbi(user_simulator, prior):
    assert callable(user_simulator), "Simulator must be a function."

    pytorch_simulator = wrap_as_pytorch_simulator(
        user_simulator, prior, False
    )

    simulator = get_batch_loop_simulator(pytorch_simulator)

    return simulator


def prepare_sbi_prior(prior):
    if isinstance(prior, Distribution):
        prior_ = prior

    if isinstance(prior, list):
        prior_ = priors.distribution(prior)

    else:
        NotImplementedError(prior)

    processed_prior, _, _ = process_prior(prior=prior_)

    n_samples = 2
    x = processed_prior.sample((n_samples,))

    lp = processed_prior.log_prob(x)
    lp.shape == x.shape[:-1]

    assert isinstance(x, Tensor), "sample must be a tensor"
    assert x.shape == Size([n_samples, len(prior_.keys)]), "sample shape is not correct"
    
    return processed_prior