import tempfile
import zipfile
import shutil
import sys
import os
from .defaultreader import DefaultReader


class MeshGroup(object):
    """
    # group name
        g [group name]
    # Vertices
        v 0.123 0.234 0.345 1.0
    # Texture coordinates
        vt 0.500 1 [0]
    # Vertex normals
        vn 0.707 0.000 0.707
    # Parameter space vertices
        vp 0.310000 3.210000 2.100000
    # Polygonal face element
        f 6/4/1 3/5/3 7/6/5
    # usemtl Material__3

    @type _material_library_file_path: str
    @type _tmp_material: str
    @type _vertex_indices: list[list[int, int, int]]
    @type _texture_indices: list[list[int, int, int]]
    @type _normal_indices: list[list[int, int, int]]
    @type _use_material: dict[str, list[int]]
    """
    def __init__(self, material_library_file_path=None):
        """
        @type material_library_file_path: str
        """
        self._material_library_file_path = material_library_file_path
        self._tmp_material = None
        self._vertex_indices = []
        self._texture_indices = []
        self._normal_indices = []
        self._use_material = {None: []}

    def __iter__(self):
        """
        access to block config from the class

        @rtype:
        """
        for face_element in self._vertex_indices:
            yield face_element

    def items_vertex(self):
        for element in self._vertex_indices:
            yield element

    def items_texture(self):
        for element in self._texture_indices:
            yield element

    def items_normal(self):
        for element in self._normal_indices:
            yield element

    def polygons_to_triangles(self, vertex_indice, texture_indice, normal_indice):
        for u in range(len(vertex_indice)-2):
            for v in range(u+1, len(vertex_indice)-1):
                for w in range(v+1, len(vertex_indice)):
                    # print(u, v, w)
                    self._vertex_indices.append([vertex_indice[u], vertex_indice[v], vertex_indice[w]])
                    if len(texture_indice):
                        self._texture_indices.append([texture_indice[u], texture_indice[v], texture_indice[w]])
                    if len(normal_indice):
                        self._normal_indices.append([normal_indice[u], normal_indice[v], normal_indice[w]])

    def parse_f(self, line):
        vertex_indice = []
        texture_indice = []
        normal_indice = []
        for entry in line.split(' '):
            values = entry.split('/')
            vertex_indice.append(int(values[0]))
            if len(values) > 1:
                try:
                    texture_indice.append(int(values[1]))
                except ValueError:
                    pass
                if len(values) > 2:
                    try:
                        normal_indice.append(int(values[2]))
                    except ValueError:
                        pass
        if len(vertex_indice) > 3:
            self.polygons_to_triangles(vertex_indice, texture_indice, normal_indice)
            return
        self._vertex_indices.append(vertex_indice)
        if len(texture_indice):
            self._texture_indices.append(texture_indice)
        if len(normal_indice):
            self._normal_indices.append(normal_indice)
        self._use_material[self._tmp_material].append(len(self._vertex_indices))

    def parse_usemtl(self, line):
        """
        @type line: str
        """
        self._tmp_material = line
        self._use_material[self._tmp_material] = []

    def has_triangular_facets(self):
        if len(self._vertex_indices) == 0:
            return False
        return len(self._vertex_indices[0]) == 3


class MeshObject(object):
    """
    # o object name
    # mtllib Scaffold.mtl
    # g group name

    @type _groups: dict[str, MeshGroup]
    @type _tmp_material_library_file_path: str
    @type _vertices: list[list[float, float, float, float]]
    @type _texture_coordinates: list[list[float, float, float]]
    @type _vertex_normals: list[list[float, float, float]
    @type _parameter_space_vertices: list[list[int, int, int]]
    """
    def __init__(self, material_library_file_path=None):
        """
        """
        self._groups = {}
        self._tmp_material_library_file_path = material_library_file_path
        self._tmp_material = None
        self._vertices = []
        self._texture_coordinates = []
        self._vertex_normals = []
        self._parameter_space_vertices = []

    def parse_g(self, line):
        """
        @type line: str
        @rtype: MeshGroup
        """
        assert line not in self._groups, "Groups are not unique: {}".format(line)
        self._groups[line] = MeshGroup(self._tmp_material_library_file_path)
        return self._groups[line]

    def parse_v(self, line):
        """
        @type line: str
        """
        values = [float(value.strip()) for value in line.split(' ')]
        self._vertices.append(values)

    def parse_vt(self, line):
        """
        @type line: str
        """
        values = [float(value) for value in line.split(' ')]
        self._texture_coordinates.append(values)

    def parse_vn(self, line):
        """
        @type line: str
        """
        values = tuple([float(value) for value in line.split(' ')])
        self._vertex_normals.append(values)

    def parse_vp(self, line):
        """
        @type line: str
        """
        values = [int(value) for value in line.split(' ')]
        self._parameter_space_vertices.append(values)

    def parse_mtllib(self, line):
        """
        @type line: str
        """
        self._tmp_material_library_file_path = line

    def get_facets(self):
        """

        @rtype: collections.Iterable[((float, float, float), (float, float, float), (float, float, float))]
        """
        for name, group in self._groups.items():
            for indice in group.items_vertex():
                yield (
                    self._vertices[indice[0]-1],
                    self._vertices[indice[1]-1],
                    self._vertices[indice[2]-1])

    def get_texture_facets(self):
        """

        @rtype: collections.Iterable[((float, float, float), (float, float, float), (float, float, float))]
        """
        for name, group in self._groups.items():
            for indice in group.items_texture():
                yield (
                    self._texture_coordinates[indice[0]-1],
                    self._texture_coordinates[indice[1]-1],
                    self._texture_coordinates[indice[2]-1])

    def get_normals(self):
        """

        @rtype: collections.Iterable[((float, float, float), (float, float, float), (float, float, float))]
        """
        for name, group in self._groups.items():
            for indice in group.items_normal():
                yield (
                    self._vertex_normals[indice[0]-1],
                    self._vertex_normals[indice[1]-1],
                    self._vertex_normals[indice[2]-1])

    def has_triangular_facets(self):
        list_of_lookups = [group.has_triangular_facets() for name, group in self._groups.items()]
        return all(list_of_lookups)


class ObjReader(DefaultReader):
    """
    @type _tmp_dir: str
    @type _objects: dict[str, MeshObject]
    @type _tmp_material_library_file_path: str
    """
    def __init__(self):
        self._tmp_dir = None
        self._objects = {}
        self._directory_textures = None
        self._tmp_material_library_file_path = None

    def __exit__(self, type, value, traceback):
        if self._tmp_dir and os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir)
        self.tmp_dir = None

    def __del__(self):
        if self._tmp_dir and os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir)
        self.tmp_dir = None

    def read(self, file_path):
        assert os.path.exists(file_path), "Bad file path: '{}'".format(file_path)
        self._objects = {}
        self._tmp_material_library_file_path = None
        current_object = None
        current_group = None
        with open(file_path) as input_stream:
            for line in input_stream:
                line = line.rstrip()
                if line.startswith('#') or not len(line):
                    continue
                line_split = line.split(' ', 1)
                if len(line_split) == 1:
                    sys.stderr.write("[ObjReader] WARNING: Bad line: {}\n".format(line_split[0]))
                    continue
                key, data = line_split
                data = data.strip()
                key = key.lower()
                if key == 'mtllib':
                    if current_object:
                        current_object.parse_mtllib(data)
                    else:
                        self.parse_mtllib(data)
                    continue
                if not current_object and key != 'o':
                    name = os.path.splitext(os.path.basename(file_path))[0]
                    current_object = self.parse_o(name)
                if not current_group and (key == 'f' or key == 'usemtl'):
                    name = os.path.splitext(os.path.basename(file_path))[0]
                    current_group = current_object.parse_g(name)
                if key == 'o':
                    current_object = self.parse_o(data)
                    continue
                if key == 'g':
                    current_group = current_object.parse_g(data)
                    continue
                if key == 'v':
                    current_object.parse_v(data)
                    continue
                if key == 'vt':
                    current_object.parse_vt(data)
                    continue
                if key == 'vn':
                    current_object.parse_vn(data)
                    continue
                if key == 'vp':
                    current_object.parse_vp(data)
                    continue
                if key == 'f':
                    current_group.parse_f(data)
                    continue
                if key == 'usemtl':
                    current_group.parse_usemtl(data)
                    continue
                else:
                    sys.stderr.write("[ObjReader] WARNING: Unknown key: {}\n".format(key))
                    continue

    @staticmethod
    def _get_obj_file_path(directory):
        list_of_dir = os.listdir(directory)
        for item in list_of_dir:
            if item. endswith('obj'):
                return os.path.join(directory, item)
        return None

    def read_archive(self, file_path):
        """

        @param file_path:
        @return:
        """
        # deal with temporary directory
        self._tmp_dir = tempfile.mkdtemp(prefix="{}_".format("ObjReader"))

        # deal with input directory
        assert zipfile.is_zipfile(file_path), "Not a zip archive: '{}'".format(file_path)
        directory_input = tempfile.mkdtemp(dir=self._tmp_dir)

        with zipfile.ZipFile(file_path, "r") as read_handler:
            read_handler.extractall(directory_input)

        directory_items = os.listdir(directory_input)
        while len(directory_items) == 1:
            if os.path.isdir(os.path.join(directory_input, directory_items[0])):
                directory_input = os.path.join(directory_input, directory_items[0])
                directory_items = os.listdir(directory_input)

        if os.path.exists(os.path.join(directory_input, 'source')):
            list_of_dir = os.listdir(directory_input)
            if len(list_of_dir) and list_of_dir[0].endswith(".zip"):
                #  source contains another zip file, most with likely copies of textures
                file_path = os.path.join(directory_input, 'source', list_of_dir[0])
                assert zipfile.is_zipfile(file_path), "Not a zip archive: '{}'".format(file_path)
                directory_input = tempfile.mkdtemp(dir=self._tmp_dir)

                with zipfile.ZipFile(file_path, "r") as read_handler:
                    read_handler.extractall(directory_input)

        if os.path.exists(os.path.join(directory_input, 'lod0')):
            directory_input = os.path.join(directory_input, 'lod0')

        directory_source = directory_input
        directory_textures = directory_input
        if os.path.exists(os.path.join(directory_input, 'source')):
            directory_source = os.path.join(directory_input, 'source')
        if os.path.exists(os.path.join(directory_input, 'textures')):
            directory_textures = os.path.join(directory_input, 'textures')
        self._directory_textures = directory_textures
        file_path_obj = ObjReader._get_obj_file_path(directory_source)
        assert isinstance(file_path_obj, str), "obj file not found."
        self.read(file_path_obj)

    def parse_o(self, line):
        """
        @type line: str
        @rtype: MeshObject
        """
        assert line not in self._objects, "Objects are not unique: {}".format(line)
        self._objects[line] = MeshObject(self._tmp_material_library_file_path)
        return self._objects[line]

    def parse_mtllib(self, line):
        self._tmp_material_library_file_path = line

    def get_facets(self, name=None):
        """

        @rtype: collections.Iterable[((float, float, float), (float, float, float), (float, float, float))]
        """
        if name:
            assert name in self._objects, "Unknown object: {}".format(name)
            for element in self._objects[name].get_facets():
                yield element
        else:
            assert name is None, "Unknown object: {}".format(name)
            for name, mesh_object in self._objects.items():
                for element in mesh_object.get_facets():
                    yield element

    def get_texture_facets(self, name=None):
        """

        @rtype: collections.Iterable[((float, float, float), (float, float, float), (float, float, float))]
        """
        if name:
            assert name in self._objects, "Unknown object: {}".format(name)
            for element in self._objects[name].get_texture_facets():
                yield element
        else:
            assert name is None, "Unknown object: {}".format(name)
            for name, mesh_object in self._objects.items():
                for element in mesh_object.get_texture_facets():
                    yield element

    def get_normals(self, name=None):
        """

        @rtype: collections.Iterable[((float, float, float), (float, float, float), (float, float, float))]
        """
        if name:
            assert name in self._objects, "Unknown object: {}".format(name)
            for element in self._objects[name].get_normals():
                yield element
        else:
            assert name is None, "Unknown object: {}".format(name)
            for name, mesh_object in self._objects.items():
                for element in mesh_object.get_normals():
                    yield element

    def get_names(self):
        """
        @rtype: collections.Iterable[str]
        """
        repr(self._objects.keys())

    def has_triangular_facets(self):
        """
        @rtype: bool
        """
        list_of_lookups = [mesh_object.has_triangular_facets() for name, mesh_object in self._objects.items()]
        # print(list_of_lookups)
        return all(list_of_lookups)
