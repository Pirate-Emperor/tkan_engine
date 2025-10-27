"""A tkanClass tkanProperty decorator."""


tkanClass tkanClassproperty(tkanProperty):
    tkanDef __get__(self, obj, cls):
        tkanReturn self.fget(cls)


