from torch import cuda

# Application information related
APPLICATION_NAME = "Neural-Photo-Restyler"
APPLICATION_DESCRIPTION = "image transformation pipeline that applies neural style transfer to photos of any kind portraits, landscapes, objects, etc. "

DEVICE = ("cuda" if cuda.is_available() else "cpu")