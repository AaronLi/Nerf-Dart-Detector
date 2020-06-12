import enum

class GPIOTYPE(enum.Enum):
    POWER = enum.auto()
    PWM = enum.auto()
    BASIC = enum.auto()

class GPIOPin:
    pass


class RaspberryPi:
    def __init__(self):
        pass