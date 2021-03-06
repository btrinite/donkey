#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Dec 06 15:29:44 2017

@author: benoit.trinite

txwcontroller.py


"""



import array
import time
import struct
import os
from threading import Thread
import donkeycar as dk
from sys import platform

from donkeycar.parts.configctrl import myConfig, CONFIG2LEVEL

import logging

#if platform != "darwin":
import serial

def map_range(x, X_min, X_max, Y_min, Y_max):
    '''
    Linear mapping between two ranges of values
    '''
    if (x<X_min):
        x=X_min
    if (x>X_max):
        x=X_max
    X_range = X_max - X_min
    Y_range = Y_max - Y_min
    XY_ratio = X_range/Y_range

    y = ((x-X_min) / XY_ratio + Y_min)

    return y

class Txserial():
    '''
    An interface to a Tx through serial link
    '''
    counter = 0

    def __init__(self, logger):
        self.ser = None
        self.lastLocalTs = 0
        self.lastDistTs = 0
        self.logger = logger
        self.status="-"
        self.th_pulse=1500
        self.st_pulse=1500

    def init(self):
        # Open serial link
        try:
            self.ser = serial.Serial(

                port=myConfig['TX']['TX_SERIAL'],
                baudrate = myConfig['TX']['TX_SERIAL_SPEED'],
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=1
            )
            self.logger.info('{} initialized'.format(myConfig['TX']['TX_SERIAL'])) 
            self.ledStatus("init")       

        except Exception as e:
           self.logger.info('Serial port not initialized '+str(e))

        return True

    def poll(self, mode, vehicle_armed=None):
        '''
        query the state of the joystick, returns button which was pressed, if any,
        and axis which was moved, if any. button_state will be None, 1, or 0 if no changes,
        pressed, or released. axis_val will be a float from -1 to +1. button and axis will
        be the string label determined by the axis map in init.
        '''
         
        steering_tx = -1
        throttle_tx = 0
        ch5_tx = 0
        ch2_tx = 0
        ch6_tx = 0
        ts = 0
        speedometer = 0
        sensor_left = 0
        sensor_right = 0
        msg=""

        if (Txserial.counter%10 == 0):
            if (vehicle_armed==None or vehicle_armed==False):
                self.ledStatus("disarmed")       
            else:
                self.ledStatus(mode)       
        Txserial.counter += 1

        if (self.ser == None):
            self.init()

        try:
            while (True):
                if (self.ser.in_waiting<=0):
                    continue
                if self.ser.in_waiting > 50:
                    self.logger.debug('poll: Serial buffer underrun {} ... flushing'.format(str(self.ser.in_waiting)))
                    self.ser.reset_input_buffer()
                    continue
                msg=self.ser.readline().decode('iso-8859-1').strip()
                if (len(msg))<=0:
                    continue
                if (msg.startswith('Debug')):
                    self.logger.debug('poll Debug msg {}'.format(msg))
                    continue
                if (msg.startswith('RX,')):
                    self.logger.debug('poll Tx msg {}'.format(msg))
                    ts, throttle_tx, steering_tx, ch2_tx, ch5_tx, ch6_tx, speedometer, sensor_left, sensor_right = map(int, msg.partition(',')[2].split(','))
                    break
                else:
                    continue

        except Exception as e:
            self.logger.info('poll: Exception while parsing msg '+str(e))
            if (str(e).startswith("Serial port not initialized")):
            	self.ser.close()
            	self.ser = None
            	self.logger.info('port Closed')
            else:
                pass
        except:
            pass

        now=time.clock()*1000
        if (steering_tx == -1):
            self.logger.debug('poll: No Rx signal , forcing idle position')
            return 1500,1500,1500,1300,1300,0,0,0

        self.lastLocalTs = now
        self.lastDistTs = ts
        self.logger.debug('poll Tx params: ts {} steering_tx= {:05.0f} throttle_tx= {:05.0f} ch2={:05.0f} speedometer= {:03.0f} sensor_left= {:05.0f} sensor_right= {:05.0f}'.format(ts, steering_tx, throttle_tx, ch2_tx, speedometer, sensor_left, sensor_right))


        return throttle_tx, steering_tx, ch2_tx, ch5_tx, ch6_tx, speedometer, sensor_left, sensor_right

    def send_cmd (self):
        cmd='{},{},{}\n'.format (self.th_pulse, self.st_pulse, self.status)
        if (self.ser != None):
            self.logger.debug('Command sent : {} '.format(cmd))
            self.ser.write(cmd.encode())

    def ledStatus (self, status):
        if (status!=None):
            self.status=status
            self.send_cmd()
    
    def set_pulse(self, pulse, pulse_ch=0):
        if (pulse_ch==0):
            self.th_pulse = pulse
        if (pulse_ch==1):
            self.st_pulse = pulse

        self.send_cmd()

   

class TxController(object):
    '''
    Tx client using access to local serial input
    '''

    def __init__(self, poll_delay=0.0,
                 auto_record_on_throttle=True,
                 verbose = False
                 ):

        self.logger = logging.getLogger(myConfig['DEBUG']['PARTS']['TXCTRL']['NAME'])
        self.logger.setLevel(CONFIG2LEVEL[myConfig['DEBUG']['PARTS']['TXCTRL']['LEVEL']])
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = 'user'
        self.poll_delay = poll_delay
        self.running = True
        self.speedometer = 0
        self.lane = 0
        self.ch5 = False
        self.ch6 = False
        self.sensor_left = 0
        self.sensor_right = 0
        self.recording = False
        self.auto_record_on_throttle = auto_record_on_throttle
        self.tx = None
        self.vehicle_armed = None
        self.verbose = verbose

    def on_throttle_changes(self):
        '''
        turn on recording when non zero throttle in the user mode.
        '''
        if self.auto_record_on_throttle:
            self.recording = (self.throttle != 0.0 and self.mode == 'user')

    def init_tx(self):
        '''
        attempt to init Tx
        '''
        try:
            self.tx = Txserial(self.logger)
            self.tx.init()
        except FileNotFoundError:
            print(" Unable to init Tx receiver.")
            self.tx = None
        return self.tx is not None


    def update(self):
        '''
        poll a Tx for input events
        '''

        #wait for Tx to be online
        while self.running and not self.init_tx():
            time.sleep(5)

        while self.running:
            throttle_tx, steering_tx, ch2_tx, ch5_tx, ch6_tx, speedometer, sensor_left, sensor_right = self.tx.poll(self.mode, self.vehicle_armed)
            if throttle_tx > myConfig['TX']['TX_THROTTLE_TRESH']:
                self.throttle = map_range(throttle_tx, myConfig['TX']['TX_THROTTLE_MIN'], myConfig['TX']['TX_THROTTLE_MAX'], -1, 1)
            else:
                self.throttle = 0
            self.on_throttle_changes()
            self.angle = 0-map_range(steering_tx, myConfig['TX']['TX_STEERING_MIN'], myConfig['TX']['TX_STEERING_MAX'], -1, 1)
            self.speedometer = 1-map_range(speedometer,myConfig['TX']['TX_SPEEDOMETER_MIN'], myConfig['TX']['TX_SPEEDOMETER_MAX'], 0, 1)
            self.lane = int(round(map_range(ch2_tx,myConfig['TX']['TX_LANE_MIN'], myConfig['TX']['TX_LANE_MAX'], 0, 2)))
            self.sensor_left = sensor_left
            self.sensor_right = sensor_right
            if (ch5_tx > myConfig['TX']['TX_CH_AUX_TRESH']+100):
                self.ch5 = True

            if (ch5_tx < myConfig['TX']['TX_CH_AUX_TRESH']-100):
                self.ch5 = False

            if (ch6_tx > myConfig['TX']['TX_CH_AUX_TRESH']+100):
                self.ch6 = True

            if (ch6_tx < myConfig['TX']['TX_CH_AUX_TRESH']-100):
                self.ch6 = False

            self.logger.debug('Mapped TX param : angle={:01.2f} throttle={:01.2f} speed={:01.2f} lane={:d} ch5={:d} ch6={:d}'.format (self.angle, self.throttle, self.speedometer, self.lane, self.ch5, self.ch6))
            time.sleep(self.poll_delay)

    def run_threaded(self, mode=None, vehicle_armed=None, img_arr=None, annoted_img=None):
        self.mode = mode
        self.vehicle_armed = vehicle_armed
        if (annoted_img is not None):
            self.img_arr = annoted_img
        else:
            self.img_arr = img_arr
        dk.perfmon.LogEvent('TXCtrl-Poll')
        return self.angle, self.throttle, self.recording, self.lane, self.ch5, self.ch6, self.speedometer, self.sensor_left, self.sensor_right

    def run(self, img_arr=None, img_annoted=None):
        raise Exception("We expect for this part to be run with the threaded=True argument.")
        return False

    def set_pulse(self, pulse, pulse_ch=0):
        if (self.tx != None):
            self.tx.set_pulse (pulse, pulse_ch)

    def shutdown(self):
        self.running = False
        time.sleep(0.5)

    def gracefull_shutdown(self):
        self.tx.ledStatus('init')


       
