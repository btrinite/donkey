import time
import donkeycar as dk

# Import the ADS1x15 module.
import Adafruit_ADS1x15

GAIN = 2/3
CELLS_DEFAULT_VOLTAGE = 3.7
CELLS_THRESH_VOLTAGE = 3

import logging
logger = logging.getLogger('donkey.battery')

class BatteryController(object):
    '''
    Battery Ctrl
    '''

    def __init__(self,
                 verbose = False, nbCells = 2
                 ):

        self.verbose = verbose
        self.running = True
        self.nbCells = nbCells
        self.adc = Adafruit_ADS1x15.ADS1115()
        self.batteryLevel = [CELLS_DEFAULT_VOLTAGE] * self.nbCells

    def update(self):
        '''
        poll for emergency events
        '''

        while self.running:
            values = [0]*4
            for i in range(self.nbCells):
                self.batteryLevel[i] = self.adc.read_adc(i, gain=GAIN)
                logger.debug('Cell {} Voltage {:01.2f}'.format(i, self.batteryLevel[i]))



    def run_threaded(self):
        return self.batteryLevel

    def run(self):
        raise Exception("We expect for this part to be run with the threaded=True argument.")
        return False

    def shutdown(self):
        self.running = False
        time.sleep(0.5)

