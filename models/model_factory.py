from models.EGLGE import EGLGE
from models.cocge.co_cge import COCGE
from models.ThreeBranchPrompt import ThreeBranchPromptModel

def get_model(config, attributes=None, classes=None, offset=None, dset=None):
    if config.model_name == 'EGLGE':
        model = EGLGE(config, dset=dset)
    elif config.model_name == 'co-cge':
        model = COCGE(config, dset=dset)
    elif config.model_name == 'ThreeBranch':
        model = ThreeBranchPromptModel(config, dset=dset)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Model Name {:s}.".format(
                config.model_name
            )
        )

    return model
