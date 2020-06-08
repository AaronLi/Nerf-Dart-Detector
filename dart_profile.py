import collections
import json

import colour_conversions

ColourIsolationSpec = collections.namedtuple("ColourIsolationSpec",
                                             ['colour_low', 'colour_high', 'dilations', 'erosions'])


def read_colour_isolation_specs(spec_dict: dict) -> ColourIsolationSpec:
    """
    Reads a dict containing colour isolation info and returns a ColourIsolationSpec instance that contains the data
    :param spec_dict: the parameters to put in the specs
    :return:the outputted spec object containing the same data as the dict
    """
    out = ColourIsolationSpec(
        colour_low=colour_conversions.hsv360_100_100_to_hsv180_255_255(spec_dict['low']),
        colour_high=colour_conversions.hsv360_100_100_to_hsv180_255_255(spec_dict['high']),
        erosions=spec_dict['erosions'],
        dilations=spec_dict['dilations']
    )
    return out


class DartProfile:
    """
    DartProfiles hold the parameters that allow the dart finding algorithm to find a dart of a certian colour.
    A dart profile has two components of type ColourIsolationSpec:
    DartProfile.body and DartProfile.tip
    these contain the details required to extract a dart from the image
    """
    def __init__(self, dart_name, tip_low=(0, 0, 0), tip_high=(0, 0, 0), body_low=(0, 0, 0), body_high=(0, 0, 0), tip_erosions=0,
                 tip_dilations=0, body_erosions=0, body_dilations=0, identification_colour=(0, 0, 0)) -> None:
        """
        Constructor for DartProfile
        :param tip_low: The lower limit for possible tip colours
        :param tip_high: The upper limit for possible tip colours
        :param body_low:  The lower limit for possible body colours
        :param body_high:  The upper limit for possible body colours
        """
        self.identification_colour = identification_colour
        self.dart_name = dart_name
        self.body = ColourIsolationSpec(body_low, body_high, body_dilations, body_erosions)
        self.tip = ColourIsolationSpec(tip_low, tip_high, tip_dilations, tip_erosions)

    @staticmethod
    def read_from_file(json_file_in):
        json_info = json.load(json_file_in)
        disabled = json_info.get('disabled')

        if disabled:
            print('Disabled: ', json_info['name'])
            return

        out = DartProfile(json_info['name'])
        body_data = json_info['body']
        tip_data = json_info['tip']

        out.body = read_colour_isolation_specs(body_data)
        out.tip = read_colour_isolation_specs(tip_data)
        out.identification_colour = colour_conversions.hsv360_100_100_to_hsv180_255_255(json_info['id_colour'])

        return out

    def __str__(self) -> str:
        return f'"{self.dart_name}": colour: {self.identification_colour} {{body: {self.body.__str__()} tip: {self.tip.__str__()}}}'

    def __repr__(self) -> str:
        return f'{super().__repr__()}: {self.__str__()}'

