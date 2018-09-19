#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it. 

Usage:
    manage.py (drive) [--model=<model>] [--js|--tx]
    manage.py (train) [--tub=<tub1,tub2,..tubn>]  (--model=<model>) [--base_model=<base_model>] [--no_cache]

Options:
    -h --help        Show this screen.
    --tub TUBPATHS   List of paths to tubs. Comma separated. Use quotes to use wildcards. ie "~/tubs/*"
    --js             Use physical joystick.
"""
import os
import logging
import sys, multiprocessing
# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    filename='data/donkey.log',
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

logger = logging.getLogger('donkey.manage')

from docopt import docopt

import donkeycar as dk

# import parts
from donkeycar.parts.camera import Webcam, PiCamera
from donkeycar.parts.transform import Lambda
from donkeycar.parts.keras import KerasCategorical, KerasLinear

from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
from donkeycar.parts.datastore import TubHandler, TubGroup
from donkeycar.parts.controller import LocalWebController, FPVWebController, JoystickController, TxController, TxActionController
from donkeycar.parts.emergency import EmergencyController
from donkeycar.parts.throttle_in_line import ThrottleInLine
from donkeycar.parts.battery import BatteryController

from sys import platform


import signal
import time

#ctr is global
ctr = None

def drive(cfg, model_path=None, use_joystick=False, use_tx=False):
    '''
    Start the drive loop
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a timely manner.
    Parts may have named outputs and inputs. The framework handles passing named outputs
    to parts requesting the same named input.
    '''

    # Initialize car
    logger.info("Init Vehicle main object")
    V = dk.vehicle.Vehicle()

    logger.info("Init Cam part")
    if cfg.USE_WEB_CAMERA:
        cam = Webcam(resolution=cfg.CAMERA_RESOLUTION)
    else:
        cam = PiCamera(resolution=cfg.CAMERA_RESOLUTION)
        
    V.add(cam, outputs=['cam/image_array'], threaded=True)

    logger.info("Init Controller part")
    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
        # modify max_throttle closer to 1.0 to have more power
        # modify steering_scale lower than 1.0 to have less responsive steering
        ctr = JoystickController(max_throttle=cfg.JOYSTICK_MAX_THROTTLE,
                                 steering_scale=cfg.JOYSTICK_STEERING_SCALE,
                                 auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE,
                                 throttle_axis = cfg.JOYSTICK_THROTTLE_AXIS,
                                 steering_axis = cfg.JOYSTICK_STEERING_AXIS,
                                 btn_mode = cfg.JOYSTICK_DRIVING_MODE_BUTTON,
                                 btn_record_toggle = cfg.JOYSTICK_RECORD_TOGGLE_BUTTON,
                                 btn_inc_max_throttle = cfg.JOYSTICK_INCREASE_MAX_THROTTLE_BUTTON,
                                 btn_dec_max_throttle = cfg.JOYSTICK_DECREASE_MAX_THROTTLE_BUTTON,
                                 btn_inc_throttle_scale = cfg.JOYSTICK_INCREASE_THROTTLE_SCALE_BUTTON,
                                 btn_dec_throttle_scale = cfg.JOYSTICK_DECREASE_THROTTLE_SCALE_BUTTON,
                                 btn_inc_steer_scale = cfg.JOYSTICK_INCREASE_STEERING_SCALE_BUTTON,
                                 btn_dec_steer_scale = cfg.JOYSTICK_DECREASE_STEERING_SCALE_BUTTON,
                                 btn_toggle_const_throttle = cfg.JOYSTICK_TOGGLE_CONSTANT_THROTTLE_BUTTON,
                                 verbose = cfg.JOYSTICK_VERBOSE
                                 )
    elif use_tx or cfg.USE_TX_AS_DEFAULT:
        ctr = TxController(throttle_tx_min = cfg.TX_THROTTLE_MIN,
                           throttle_tx_max = cfg.TX_THROTTLE_MAX,
                           steering_tx_min = cfg.TX_STEERING_MIN,
                           steering_tx_max = cfg.TX_STEERING_MAX,
                           throttle_tx_thresh = cfg.TX_THROTTLE_TRESH,
                           ch_aux_tx_thresh = cfg.TX_CH_AUX_TRESH,
                           verbose = cfg.TX_VERBOSE
                           )
    else:        
        #This web controller will create a web server that is capable
        #of managing steering, throttle, and modes, and more.
        ctr = LocalWebController()

    V.add(ctr,
          inputs=['cam/image_array', 'pilot/annoted_img'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording', 'ch5', 'ch6'],
          threaded=True)


    if use_tx or cfg.USE_TX_AS_DEFAULT:
        actionctr = TxActionController()
        V.add(actionctr,
          inputs=['user/mode', 'ch5', 'ch6'],
          outputs=['user/mode', 'flag'],
          threaded=True)

    if cfg.USE_THROTTLEINLINE:
        logger.info("Init throttleInLine part")
        throttleinline = ThrottleInLine(cfg.THROTTLEINLINE_ANGLE_MIN, cfg.THROTTLEINLINE_ANGLE_MAX)
        V.add(throttleinline,
                inputs=['cam/image_array'],
                outputs=['pilot/throttle_boost', 'pilot/annoted_img'],
                threaded=True)

    logger.info("Init emergency part")
    emergencyCtrl = EmergencyController()

    V.add(emergencyCtrl,
          inputs=['user/mode'],
          outputs=['user/mode'],
          threaded=True)
    # See if we should even run the pilot module.
    # This is only needed because the part run_condition only accepts boolean
    def pilot_condition(mode):
        if mode == 'user':
            return False
        else:
            return True

    logger.info("Init pilot part")
    pilot_condition_part = Lambda(pilot_condition)
    V.add(pilot_condition_part, inputs=['user/mode'], outputs=['run_pilot'])

    logger.info("Init Model part")
    # Run the pilot if the mode is not user and not Tx.
    kl = KerasCategorical()
    #kl = KerasLinear()
    if model_path:
        if (os.path.exists(model_path)):
            logger.info("IA : Load integrated model")
            kl.load(model_path)
        else:
            # Model reconstruction from JSON file
            logger.info("IA : Load Weights + Model Architecture model")
            kl.load2(model_path)

    V.add(kl, inputs=['cam/image_array'],
        outputs=['pilot/angle', 'pilot/throttle'],
        run_condition='run_pilot')

    # Choose what inputs should change the car.
    def drive_mode(mode,
                   user_angle, user_throttle,
                   pilot_angle, pilot_throttle, throttle_boost):
        if mode == 'user':
            return user_angle, user_throttle

        else:
            if cfg.USE_THROTTLEINLINE:
                if throttle_boost:
                    pilot_throttle = pilot_throttle*cfg.THROTTLEINLINE_BOOST_FACTOR
                    logger.debug("Apply Boost")
            if mode == 'local_angle':
                return pilot_angle, user_throttle
            else:
                logger.debug('drive_mode: Pilot return angle={:01.2f} throttle={:01.2f}'.format(pilot_angle, pilot_throttle))
                return pilot_angle, pilot_throttle

    drive_mode_part = Lambda(drive_mode)
    V.add(drive_mode_part,
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle', 'pilot/throttle_boost'],
          outputs=['angle', 'throttle'])

    if cfg.USE_PWM_ACTUATOR:
        logger.info("Init Actuator part")
        steering_controller = PCA9685(cfg.STEERING_CHANNEL)

        steering = PWMSteering(controller=steering_controller,
                            left_pulse=cfg.STEERING_LEFT_PWM,
                            right_pulse=cfg.STEERING_RIGHT_PWM)

        throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL)
        throttle = PWMThrottle(controller=throttle_controller,
                            max_pulse=cfg.THROTTLE_FORWARD_PWM,
                            zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                            min_pulse=cfg.THROTTLE_REVERSE_PWM,
                            kick_pulse=cfg.THROTTLE_KICK_PULSE,
                            min_spd_pulse=cfg.THROTTLE_MIN_SPD_PULSE,
                            constant_mode=cfg.THROTTLE_CONSTANT_MODE)

        V.add(steering, inputs=['angle'])
        V.add(throttle, inputs=['throttle', 'user/mode'])

    if cfg.BATTERY_USE_MONITOR:
        logger.info("Init Battery Monitor part")
        battery_controller = BatteryController (nbCells=cfg.BATTERY_NCELLS)
        V.add(battery_controller, outputs = ['battery'], threaded=True)

    # add tub to save data
    inputs = ['cam/image_array', 'user/angle', 'user/throttle', 'user/mode', 'pilot/angle', 'pilot/throttle', 'flag']
    types = ['image_array', 'float', 'float', 'str', 'numpy.float32', 'numpy.float32', 'str']

    logger.info("Init Tub Handler part")
    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=inputs, types=types)
    V.add(tub, inputs=inputs, run_condition='recording')

    if use_tx or cfg.USE_TX_AS_DEFAULT:
        fpv = FPVWebController()
        V.add(fpv,
                inputs=['cam/image_array', 'pilot/annoted_img', 'user/angle', 'user/throttle', 'user/mode', 'pilot/angle', 'pilot/throttle', 'pilot/throttle_boost'],
                threaded=True)        

    logger.info("Start main loop")

    # run the vehicle for 20 seconds
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS)

    print("You can now go to <your pi ip address>:8887 to drive your car.")


def train(cfg, tub_names, model_name, base_model=None):
    '''
    use the specified data in tub_names to train an artifical neural network
    saves the output trained model as model_name
    '''
    X_keys = ['cam/image_array']
    y_keys = ['user/angle', 'user/throttle']

    def rt(record):
        record['user/angle'] = dk.utils.linear_bin(record['user/angle'])
        return record

    kl = KerasCategorical()
    #kl = KerasLinear()
    print(base_model)
    if base_model is not None:
        base_model = os.path.expanduser(base_model)
        kl.load(base_model)

    print('tub_names', tub_names)
    if not tub_names:
        tub_names = os.path.join(cfg.DATA_PATH, '*')

    tubgroup = TubGroup(tub_names)
    train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys, record_transform=rt,
                                                    batch_size=cfg.BATCH_SIZE,
                                                    train_frac=cfg.TRAIN_TEST_SPLIT)

    model_path = os.path.expanduser(model_name)

    total_records = len(tubgroup.df)
    total_train = int(total_records * cfg.TRAIN_TEST_SPLIT)
    total_val = total_records - total_train
    print('train: %d, validation: %d' % (total_train, total_val))
    steps_per_epoch = total_train // cfg.BATCH_SIZE
    print('steps_per_epoch', steps_per_epoch)

    kl.train(train_gen,
             val_gen,
             saved_model_path=model_path,
             steps=steps_per_epoch,
             train_split=cfg.TRAIN_TEST_SPLIT)

def softExit():
        if (ctr  != None):
            ctr.gracefull_shutdown()
        time.sleep(0.2)
        logger.info ('Exit')
        os._exit(1)        

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self,signum, frame):
        self.kill_now = True
        softExit()

def log_exception(*args):
    logger.info ('Got exception %s' % (args,))
    softExit()

#sys.excepthook = log_exception

if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()

#    killer = GracefulKiller()

    if args['drive']:
        logger.info("Start in drive mode")
        drive(cfg, model_path=args['--model'], use_joystick=args['--js'], use_tx=args['--tx'])

    elif args['train']:
        tub = args['--tub']
        model = args['--model']
        base_model = args['--base_model']
        cache = not args['--no_cache']
        train(cfg, tub, model, base_model=base_model)
    
#    while True:
#        time.sleep(1)
#        if killer.kill_now:
#            break






