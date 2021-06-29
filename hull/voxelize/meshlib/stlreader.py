import numpy as np
import os
from struct import unpack
from .defaultreader import DefaultReader


class StlReader(DefaultReader):
    """
    @type _facets: dict[str, list[tuple[tuple[float]]]]
    @type _norms: dict[str, list[tuple[float]]]
    """

    def __init__(self):
        self._facets = {}
        self._norms = {}

    @staticmethod
    def read_binary(file_path):
        """
        Created on Thu Nov 19 06:37:35 2013

        @author: Sukhbinder Singh

        Reads a Binary file and
        Returns Header,Points,Normals,Vertex1,Vertex2,Vertex3

        Source: http://sukhbinder.wordpress.com/2013/11/28/binary-stl-file-reader-in-python-powered-by-numpy/

        @type file_path: str
        @rtype:
        """
        fp = open(file_path, 'rb')
        header = fp.read(80)
        nn = fp.read(4)
        number_of_facets = unpack('i', nn)[0]
        record_dtype = np.dtype([
            ('normals', np.float32, (3,)),
            ('Vertex1', np.float32, (3,)),
            ('Vertex2', np.float32, (3,)),
            ('Vertex3', np.float32, (3,)),
            ('atttr', '<i2', (1,))
        ])
        data = np.fromfile(fp, dtype=record_dtype, count=number_of_facets)
        fp.close()

        normals = data['normals']
        vertex_1 = data['Vertex1']
        vertex_2 = data['Vertex2']
        vertex_3 = data['Vertex3']

        # p = np.append(vertex_1, vertex_2, axis=0)
        # p = np.append(p, vertex_3, axis=0)  # list(v1)
        # points = np.array(list(set(tuple(p1) for p1 in p)))

        return header, normals, vertex_1, vertex_2, vertex_3

    @staticmethod
    def parse_askii_verticle(input_stream):
        """
        'vertex 0.0 0.0 0.0'

        @param input_stream:
        @rtype: (float, float, float)
        """
        _, verticle_x, verticle_y, verticle_z = input_stream.readline().strip().split(' ')
        return float(verticle_x), float(verticle_y), float(verticle_z),

    @staticmethod
    def parse_askii_triangle(input_stream):
        """
        'vertex 0.0 0.0 0.0' x3

        @param input_stream:
        @rtype: ((float, float, float), (float, float, float), (float, float, float))
        """
        assert input_stream.readline().strip().startswith("outer loop")
        triangle = (
            StlReader.parse_askii_verticle(input_stream),
            StlReader.parse_askii_verticle(input_stream),
            StlReader.parse_askii_verticle(input_stream))
        assert input_stream.readline().strip().startswith("endloop")
        return triangle

    @staticmethod
    def parse_askii_list_of_facets(input_stream):
        """
        'facet normal 0.0 -1.0 0.0'
        'outer loop'
        'vertex 0.0 0.0 0.0' x3
        'endloop'
        'endfacet'

        @param input_stream:
        @rtype: collections.Iterable[((float, float, float), ((float, float, float), (float, float, float), (float, float, float)))]
        """
        line = input_stream.readline().strip()
        while not line.startswith("endsolid"):
            _, _, normal_x, normal_y, normal_z = line.split(' ')
            triangle = StlReader.parse_askii_triangle(input_stream)
            assert input_stream.readline().strip().startswith("endfacet")
            yield (normal_x, normal_y, normal_z), triangle
            line = input_stream.readline().strip()

    @staticmethod
    def parse_askii_solids(input_stream):
        """
        'solid cube_corner'
        'facet normal 0.0 -1.0 0.0'
        'outer loop'
        'vertex 0.0 0.0 0.0' x3
        'endloop'
        'endfacet'
        'endsolid'

        @param input_stream:
        @rtype: collections.Iterable[(str, collections.Iterable[((float, float, float), ((float, float, float), (float, float, float), (float, float, float)))]])]
        """
        line = input_stream.readline()
        while line:
            line = line.strip()
            assert line.startswith("solid"), line
            _, name = line.split(' ', 1)
            # print(line)
            yield name, StlReader.parse_askii_list_of_facets(input_stream)
            line = input_stream.readline()
        input_stream.close()

    @staticmethod
    def read_askii_stl(file_path):
        """

        @type file_path: str
        @rtype: collections.Iterable[(str, collections.Iterable[((float, float, float), ((float, float, float), (float, float, float), (float, float, float)))]])]
        """
        assert os.path.exists(file_path), "Bad path: {}".format(file_path)
        return StlReader.parse_askii_solids(open(file_path, 'r'))

    @staticmethod
    def _is_ascii_stl(file_path):
        """

        @type file_path: str

        @rtype: bool
        """
        with open(file_path, 'rb') as input_data:
            line = input_data.readline()
            if line.startswith(b'solid'):
                return True
            else:
                return False

    def read(self, file_path):
        """

        @type file_path: str
        @rtype: None
        """
        del self._facets
        del self._norms
        self._facets = {}
        self._norms = {}
        if StlReader._is_ascii_stl(file_path):
            for name, facets in StlReader.read_askii_stl(file_path):
                assert name not in self._facets, "Objects in file are not unique"
                self._facets[name] = []
                self._norms[name] = []
                for normal, (v1, v2, v3) in facets:
                    self._facets[name].append((v1, v2, v3))
                    self._norms[name].append(normal)
        else:
            head, n, v1, v2, v3 = StlReader.read_binary(file_path)
            self._facets["obj"] = []
            self._norms["obj"] = []
            for norm, vertex_1, vertex_2, vertex_3 in zip(n, v1, v2, v3):
                # yield (tuple(i), tuple(j), tuple(k))
                self._facets["obj"].append((vertex_1, vertex_2, vertex_3))
                self._norms["obj"].append(norm)

    def get_names(self):
        """
        @rtype: collections.Iterable[str]
        """
        return self._facets.keys()

    def get_facets(self, name=None):
        """

        @rtype: collections.Iterable[((float, float, float), (float, float, float), (float, float, float))]
        """
        if name:
            assert name in self._facets, "Unknown object: {}".format(name)
            for facet in self._facets[name]:
                yield facet
        else:
            assert name is None, "Unknown object: {}".format(name)
            for name in self._facets:
                for facet in self._facets[name]:
                    yield facet

    def has_triangular_facets(self):
        """

        @rtype: bool
        """
        # todo: is this always the case?
        return True
