class DefaultReader(object):
    """
    Mesh Reader Prototype
    """

    def read_archive(self, file_path):
        """

        @type file_path: str
        @rtype: None
        """
        pass

    def read(self, file_path):
        """

        @type file_path: str
        @rtype: None
        """
        pass

    def get_names(self):
        """
        @rtype: collections.Iterable[str]
        """
        pass

    def get_facets(self, name=None):
        """

        @rtype: collections.Iterable[((float, float, float), (float, float, float), (float, float, float))]
        """
        pass

    def has_triangular_facets(self):
        """

        @rtype: bool
        """
        pass
