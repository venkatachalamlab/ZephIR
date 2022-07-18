from ..utils.utils import *


class Container:
    """Zephir variable container.

    Store and handle variables and properties that are shared across Zephir methods.
    """

    def __init__(self, dataset, **kwargs):

        self.dataset = Path(dataset)
        self.metadata = get_metadata(dataset)

        self.props = {
            'dataset': self.dataset,
            'img_shape': (self.metadata['shape_z'],
                          self.metadata['shape_y'],
                          self.metadata['shape_x']),
            'shape_t': self.metadata['shape_t'],
        }

        self.update(kwargs)

    def get(self, key: str):
        """Fetch variable with given name.

        :param key: name of variable to fetch.
        :return: specified variable
        """
        if key in self.props:
            return self.props[key]
        else:
            print(f'*** ERROR: variable:{key} not found in container!')
            return None

    def update(self, props: dict):
        """Update value for existing variable or add as new variable to container.props.

        :param props: dictionary of names and variables to store.
        """
        for key, value in props.items():
            self.props[key] = value
        update_checkpoint(self.dataset, {'container': self})
        return

