import argparse
import difftetvr as d


# https://stackoverflow.com/questions/78750965/how-to-make-argparse-work-nicely-with-enums-and-default-values
def enum_action(enum_class):
    class EnumAction(argparse.Action):
        def __init__(self, *args, **kwargs):
            # table = {member.name.casefold(): member for member in enum_class}
            table = {}
            for member in enum_class:
                table[member.name] = member
                table[member.name.casefold()] = member
            super().__init__(*args, choices=table, **kwargs)
            self.table = table

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, self.table[values])

    return EnumAction


class SplitGradientTypeAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        table = {
            'POSITION': d.SplitGradientType.POSITION,
            'COLOR': d.SplitGradientType.COLOR,
            'ABS_POSITION': d.SplitGradientType.ABS_POSITION,
            'ABS_COLOR': d.SplitGradientType.ABS_COLOR,
            'POSITION'.casefold(): d.SplitGradientType.POSITION,
            'COLOR'.casefold(): d.SplitGradientType.COLOR,
            'ABS_POSITION'.casefold(): d.SplitGradientType.ABS_POSITION,
            'ABS_COLOR'.casefold(): d.SplitGradientType.ABS_COLOR,
        }
        super().__init__(*args, choices=table, **kwargs)
        self.table = table

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.table[values])


class RendererTypeAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        table = {
            'PPLL': d.RendererType.PPLL,
            'PROJECTION': d.RendererType.PROJECTION,
            'INTERSECTION': d.RendererType.INTERSECTION,
            'ppll': d.RendererType.PPLL,
            'projection': d.RendererType.PROJECTION,
            'intersection': d.RendererType.INTERSECTION,
            'Projection': d.RendererType.PROJECTION,
            'ProjectedTetra': d.RendererType.PROJECTION,
            'PT': d.RendererType.PROJECTION,
            'Intersection': d.RendererType.INTERSECTION,
        }
        super().__init__(*args, choices=table, **kwargs)
        self.table = table

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.table[values])
