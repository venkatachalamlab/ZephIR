from ..utils.utils import *

from ..models.zephir import ZephIR
from ..zephod.model import ZephOD


def build_models(
    container,
    dimmer_ratio,
    grid_shape,
    fovea_sigma,
    n_chunks,):
    """Compile PyTorch models for tracking.

    Models are built according to user arguments. Zephod is only built when a
    compatible model is available.

    :param container: variable container, needs to contain: dataset, allow_rotation,
    dev, img_shape, n_frame, shape_n
    :param dimmer_ratio: Zephir parameter, dims descriptor edges relative to center
    :param grid_shape: Zephir parameter, descriptor size
    :param fovea_sigma: Zephir parameter, size of foveated region
    :param n_chunks: Zephir parameter, number of chunks to divide forward pass into
    :return: container (updated entry for: grid_shape), zephir, zephod
    """
    # pull variables from container
    dataset = container.get('dataset')
    allow_rotation = container.get('allow_rotation')
    dev = container.get('dev')
    img_shape = container.get('img_shape')
    n_frame = container.get('n_frame')
    shape_n = container.get('shape_n')
    
    z_stacks = min(img_shape[0], grid_shape[0])
    grid_shape = (z_stacks - (z_stacks+1) % 2, grid_shape[1], grid_shape[2])
    grid_spacing = tuple(np.array(grid_shape) / np.array(img_shape))
    
    # models and loss function compiled
    print(f'\nCompiling models and loss function...')
    model_kwargs = {
        'allow_rotation': allow_rotation,
        'dimmer_ratio': dimmer_ratio,
        'fovea_sigma': fovea_sigma,
        'grid_shape': grid_shape,
        'grid_spacing': grid_spacing,
        'n_chunks': n_chunks,
        'n_frame': n_frame,
        'shape_n': shape_n,
        'ftr_ratio': 0.6,
        'ret_stride': 2,
    }
    zephir = ZephIR(**model_kwargs)

    # compiling Zephod models from existing kwargs and state_dict
    zephod_path = dataset / 'zephod.pt'
    if not zephod_path.is_file():
        zephod_path = Path(__file__).parent.parent / 'zephod' / 'models.pt'
    if zephod_path.is_file():
        try:
            zephod_checkpoint = torch.load(str(zephod_path))
        except RuntimeError:
            print('*** CUDA NOT AVAILABLE! Mapping Zephod parameters to CPU...')
            zephod_checkpoint = torch.load(str(zephod_path), map_location=torch.device('cpu'))
        zephod = ZephOD(**zephod_checkpoint['model_kwargs'])
        zephod.load_state_dict(zephod_checkpoint['state_dict'])
        zephod.to(dev)
    else:
        print('******* ERROR: Zephod checkpoint file not found!')
        zephod = None

    # push variables to container
    container.update({
        'grid_shape': grid_shape,
    })

    # push models to checkpoint
    update_checkpoint(dataset, {
        'zephir_kwargs': model_kwargs,
        'zephir': zephir,
        'zephod': zephod
    })

    return container, zephir, zephod
