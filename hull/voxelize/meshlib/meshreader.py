import os
from .defaultreader import DefaultReader
from .stlreader import StlReader
from .objreader import ObjReader


class MeshReader(DefaultReader):
    """

    @type _type_reader: dict[str, any]
    @type _reader: DefaultReader
    """
    _type_reader = {
        ".stl": StlReader,
        ".obj": ObjReader,
        ".zip": ObjReader,
    }

    def __init__(self):
        self._reader = DefaultReader()

    def read_archive(self, file_path):
        """

        @type file_path: str
        @rtype: None
        """
        # todo: test for stl or obj archive
        assert os.path.exists(file_path), "Bad file path: {}".format(file_path)
        assert os.path.isfile(file_path), "Not a file: {}".format(file_path)
        file_extension = os.path.splitext(os.path.basename(file_path))[1].lower()
        assert file_extension == ".zip", "Bad file type: {}".format(file_extension)
        del self._reader
        self._reader = self._type_reader[file_extension]()
        self._reader.read_archive(file_path)

    def read(self, file_path):
        """

        @type file_path: str
        @rtype: None
        """
        assert os.path.exists(file_path), "Bad file path: {}".format(file_path)
        assert os.path.isfile(file_path), "Not a file: {}".format(file_path)
        file_extension = os.path.splitext(os.path.basename(file_path))[1].lower()
        assert file_extension in self._type_reader, "Unknown file type: {}".format(file_extension)
        del self._reader
        self._reader = self._type_reader[file_extension]()
        self._reader.read(file_path)

    def get_facets(self, name=None):
        """

        @rtype: collections.Iterable[((float, float, float), (float, float, float), (float, float, float))]
        """
        return self._reader.get_facets()

    def has_triangular_facets(self):
        """

        @rtype: bool
        """
        return self._reader.has_triangular_facets()
