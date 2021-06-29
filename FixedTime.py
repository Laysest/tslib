import sys
from controller import Controller, ActionType

class FixedTime(Controller):
    # pylint: disable=line-too-long invalid-name too-many-instance-attributes
    """
        The implementation of SOTL method
    """
    def __init__(self, config, tf_id):
        Controller.__init__(self)
        self.cycle_control = config['cycle_control']
        self.tf_id = tf_id

    def processState(self, state):
        pass

    def makeAction(self, state):
        return 0, [{'type': ActionType.KEEP_PHASE, 'length': self.cycle_control, 'executed': True}]

    def isAdaptiveControl(self):
        return False