"""Minimal stand-in for Pythonista's `ui`: just enough for textures."""
import io

from PIL import Image as _PIL


class Image(object):
    def __init__(self, pil=None):
        self.pil = pil

    @classmethod
    def from_data(cls, data, scale=1.0):
        return cls(_PIL.open(io.BytesIO(data)).convert('RGBA'))

    @classmethod
    def named(cls, path):
        return cls(_PIL.open(path).convert('RGBA'))

    @property
    def size(self):
        return self.pil.size


def get_screen_size():
    return (390, 844)
